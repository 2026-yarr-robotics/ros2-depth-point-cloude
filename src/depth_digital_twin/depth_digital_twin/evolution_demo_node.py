"""evolution_demo_node — replay the *evolution* of the depth digital twin as a
set of switchable visualisation methods, for recording GIFs in RViz.

The node reads the exo (and hand) RGB-D + YOLO detections of a recorded
sequence and renders ONE detection method at a time into a common `world`
(= robot base) frame. A small Tk window (`evolution_control`) cycles the active
method with a "Next method ▶" button; each method is meant to be recorded as a
separate GIF, in order, to tell the development story:

  M0   Raw cloud, ArUco OFF ..... whole-scene cloud in RAW camera coordinates
                                  (correction NOT applied) → floats off the
                                  table/robot
  M0.5 Raw cloud, ArUco ON ...... same cloud through camera→world(=robot base)
                                  → snaps onto the table, aligned to the robot
  1  Raw point-cloud OBB ........ no erode / no filtering → big, noisy box
  2  Point-cloud cone fit ....... fit the known cup frustum in the cloud (heavy)
  3  Exo + Hand fusion .......... exo alone is blind to some cups; add the
                                  wrist (hand) camera, whose world pose comes
                                  from joint-FK, and fuse by inverse covariance
  4  RGB rim fit ................ silhouette/contour fit (precise but heavy)
  5  Erode + shape-prior AABB ... erode + filter → clean cloud → known-cup
                                  axis-aligned box (a body-of-revolution cup has
                                  no yaw → AABB), used as a prior to SELECT the
                                  cup to grab

Every method reuses the SAME measurement maths the production pipeline uses
(`cup_geometry`, `point_cloud_node` helpers) — this is a faithful reproduction,
not a mock, but rendered cleanly (no KF/tracking smear) so the stages read
clearly on camera.

World frame requirement: the cups can only be expressed in the robot base frame
once the exo ArUco marker is detected. The node waits for the
`world ← exo_color_optical_frame` TF (published by `world_origin_node` after it
locks the marker on the early warm-up frames) and, the first time it appears,
calls the player's `~/begin_range` service so playback jumps to the target
window (e.g. 1333..1666) and loops there.
"""
from __future__ import annotations

import numpy as np
import rclpy
import cv2
import message_filters
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time as RclTime
from std_msgs.msg import ColorRGBA, Header, Int32, String
from std_srvs.srv import Trigger
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
import tf2_ros

from depth_digital_twin.intrinsics import load_intrinsics
from depth_digital_twin.cup_geometry import (
    fit_silhouette_xy, edge_snap_fit, cone_silhouette_px, largest_contour,
    xy_cov_from_px, fuse_xy, ray_through_point)
from depth_digital_twin.point_cloud_node import (
    _compute_box_world, _fit_cup_axis_xy, _cup_frustum_geometry,
    _make_pointcloud2, _mad_keep_indices, _filter_spatial_density,
    _box_corners, _pack_rgb, _fixed_cup_centroid)
from depth_digital_twin_msgs.msg import SegmentedObjectArray


# (tag, short English title, Korean one-liner). Order = the evolution story.
# M0/M0.5 bookend the ArUco world calibration: M0 plots the cloud in raw camera
# coordinates (correction NOT applied) so it floats off the table/robot; M0.5
# applies camera→world(=robot base) so it snaps into place.
METHODS = [
    ('M0',   'Raw Cloud (ArUco OFF)', '보정 전 · 카메라 좌표'),
    ('M0.5', 'Raw Cloud (ArUco ON)', '보정 적용 · world=robot base'),
    ('M1', 'Raw Point-Cloud OBB', '필터 없음 · 크고 노이즈'),
    ('M2', 'Point-Cloud Cone Fit', '컵 형상 탐색 · 무거움'),
    ('M3', 'Exo + Hand Fusion', 'hand 관절 FK 보정'),
    ('M4', 'RGB Rim Fit', '정밀하나 배포엔 무거움'),
    ('M5', 'Erode + Shape-Prior AABB', '형상 prior로 타깃 컵 선택'),
]

UPRIGHT = {'upright-cup', 'cup', 'upright_cup'}
FALLEN = {'fallen-cup', 'fallen_cup'}


def _C(r, g, b, a=1.0) -> ColorRGBA:
    return ColorRGBA(r=float(r), g=float(g), b=float(b), a=float(a))


def _quat_to_R(qx, qy, qz, qw) -> np.ndarray:
    n = qx * qx + qy * qy + qz * qz + qw * qw
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s * (qy * qy + qz * qz), s * (qx * qy - qz * qw),
         s * (qx * qz + qy * qw)],
        [s * (qx * qy + qz * qw), 1 - s * (qx * qx + qz * qz),
         s * (qy * qz - qx * qw)],
        [s * (qx * qz - qy * qw), s * (qy * qz + qx * qw),
         1 - s * (qx * qx + qy * qy)],
    ], dtype=np.float64)


class EvolutionDemoNode(Node):
    def __init__(self) -> None:
        super().__init__('evolution_demo_node')
        gp = self.declare_parameter

        gp('intrinsics_exo_path', '')
        gp('intrinsics_hand_path', '')
        gp('world_frame', 'world')
        gp('exo_frame', 'exo_color_optical_frame')
        gp('hand_frame', 'hand_color_optical_frame')
        gp('depth_unit', 0.001)
        gp('z_min', 0.1)
        gp('z_max', 6.0)
        gp('downsample', 2)

        gp('exo_rgb_topic', '/camera_exo/color/image_raw')
        gp('exo_depth_topic', '/camera_exo/aligned_depth_to_color/image_raw')
        gp('exo_detections_topic', '/digital_twin/detections_exo')
        gp('hand_rgb_topic', '/camera_hand/color/image_raw')
        gp('hand_depth_topic', '/camera_hand/aligned_depth_to_color/image_raw')
        gp('hand_detections_topic', '/digital_twin/detections_hand')

        gp('points_topic', '/demo/points')
        gp('markers_topic', '/demo/markers')
        gp('overlay_topic', '/demo/overlay')
        gp('status_topic', '~/status')

        gp('cup_top_diameter_m', 0.054)
        gp('cup_bottom_diameter_m', 0.078)
        gp('cup_height_m', 0.095)
        gp('cup_polygon_segments', 24)
        gp('table_z_m', 0.0)

        gp('mask_erode_px', 3)
        gp('depth_gradient_max_m', 0.06)
        gp('box_outlier_mad_k', 3.0)
        gp('spatial_density_radius_m', 0.03)
        gp('spatial_density_min_neighbors', 5)
        gp('rim_sigma_px', 2.0)
        gp('cluster_xy_m', 0.06)          # exo<->hand same-cup gate (m)
        # M0 / M0.5: whole-scene cloud sampling (not per-cup). Coarse stride
        # keeps RViz light; max_z focuses on the near workspace so the
        # uncorrected/corrected contrast is a clean slab, not the whole room.
        gp('full_cloud_stride', 8)
        gp('full_cloud_max_z', 2.0)

        gp('start_method', 1)
        gp('approx_sync_slop', 0.08)
        gp('render_period_s', 0.0)        # >0 throttles heavy renders
        gp('player_begin_range_srv', '/sequence_player_node/begin_range')

        def P(n):
            return self.get_parameter(n).value

        ix = str(P('intrinsics_exo_path'))
        ih = str(P('intrinsics_hand_path'))
        if not ix:
            raise RuntimeError('intrinsics_exo_path is required')
        self.intr_exo = load_intrinsics(ix)
        self.intr_hand = load_intrinsics(ih) if ih else None

        self.world_frame = str(P('world_frame'))
        self.exo_frame = str(P('exo_frame'))
        self.hand_frame = str(P('hand_frame'))
        self.depth_unit = float(P('depth_unit'))
        self.z_min = float(P('z_min'))
        self.z_max = float(P('z_max'))
        self.downsample = max(1, int(P('downsample')))
        self.full_stride = max(1, int(P('full_cloud_stride')))
        self.full_max_z = float(P('full_cloud_max_z'))

        self.r_top = float(P('cup_top_diameter_m')) * 0.5
        self.r_bot = float(P('cup_bottom_diameter_m')) * 0.5
        self.cup_h = float(P('cup_height_m'))
        self.n_seg = int(P('cup_polygon_segments'))
        self.table_z = float(P('table_z_m'))

        self.erode_px = int(P('mask_erode_px'))
        self.grad_max = float(P('depth_gradient_max_m'))
        self.mad_k = float(P('box_outlier_mad_k'))
        self.dens_r = float(P('spatial_density_radius_m'))
        self.dens_n = int(P('spatial_density_min_neighbors'))
        self.rim_sigma_px = float(P('rim_sigma_px'))
        self.cluster_xy = float(P('cluster_xy_m'))
        self.slop = float(P('approx_sync_slop'))
        self.render_period = float(P('render_period_s'))

        self.method = int(np.clip(int(P('start_method')), 1, len(METHODS)))
        self.bridge = CvBridge()
        self._hand = None            # latest (rgb, depth, det) bundle
        self._begin_called = False
        self._last_render = None     # rclpy Time of last render

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.pub_points = self.create_publisher(
            PointCloud2, str(P('points_topic')), 1)
        self.pub_markers = self.create_publisher(
            MarkerArray, str(P('markers_topic')), 1)
        self.pub_overlay = self.create_publisher(
            Image, str(P('overlay_topic')), 1)
        self.pub_status = self.create_publisher(
            String, str(P('status_topic')), 1)

        self.create_service(Trigger, '~/next_method', self._srv_next)
        self.create_service(Trigger, '~/prev_method', self._srv_prev)
        self.create_subscription(Int32, '~/set_method', self._on_set, 1)

        self._begin_cli = self.create_client(
            Trigger, str(P('player_begin_range_srv')))

        # exo drives rendering; hand is cached for method 3.
        exo = [message_filters.Subscriber(self, Image, str(P('exo_rgb_topic'))),
               message_filters.Subscriber(self, Image,
                                          str(P('exo_depth_topic'))),
               message_filters.Subscriber(self, SegmentedObjectArray,
                                          str(P('exo_detections_topic')))]
        self.exo_sync = message_filters.ApproximateTimeSynchronizer(
            exo, queue_size=30, slop=self.slop)
        self.exo_sync.registerCallback(self._on_exo)

        hand = [message_filters.Subscriber(self, Image,
                                           str(P('hand_rgb_topic'))),
                message_filters.Subscriber(self, Image,
                                           str(P('hand_depth_topic'))),
                message_filters.Subscriber(self, SegmentedObjectArray,
                                           str(P('hand_detections_topic')))]
        self.hand_sync = message_filters.ApproximateTimeSynchronizer(
            hand, queue_size=30, slop=self.slop)
        self.hand_sync.registerCallback(self._on_hand)

        # Periodic status so the Tk panel shows the current method even before
        # any frame is rendered.
        self.create_timer(0.3, self._publish_status)
        self.get_logger().info(
            f'evolution_demo_node ready — method {self.method}/{len(METHODS)}')

    # ── method switching ────────────────────────────────────────────────────
    def _set_method(self, m: int) -> None:
        self.method = int(np.clip(m, 1, len(METHODS)))
        self._publish_status()
        self.get_logger().info(f'method → {self._title()}')

    def _srv_next(self, _req, res):
        self._set_method(self.method % len(METHODS) + 1)
        res.success, res.message = True, self._title()
        return res

    def _srv_prev(self, _req, res):
        self._set_method((self.method - 2) % len(METHODS) + 1)
        res.success, res.message = True, self._title()
        return res

    def _on_set(self, msg: Int32) -> None:
        self._set_method(int(msg.data))

    def _title(self) -> str:
        i, en, ko = METHODS[self.method - 1]
        return f'[{i}] {en} — {ko}'

    def _publish_status(self) -> None:
        self.pub_status.publish(String(data=self._title()))

    # ── input plumbing ──────────────────────────────────────────────────────
    def _on_hand(self, rgb, depth, det) -> None:
        self._hand = (rgb, depth, det)

    def _lookup(self, frame: str, stamp):
        """world ← frame at `stamp` (fall back to latest). Returns (R, t)."""
        for t in (RclTime.from_msg(stamp), RclTime()):
            try:
                tf = self.tf_buffer.lookup_transform(
                    self.world_frame, frame, t, timeout=Duration(seconds=0.0))
                q = tf.transform.rotation
                tr = tf.transform.translation
                R = _quat_to_R(q.x, q.y, q.z, q.w)
                return R, np.array([tr.x, tr.y, tr.z], dtype=np.float64)
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException):
                continue
        return None

    def _maybe_begin_range(self) -> None:
        if self._begin_called:
            return
        self._begin_called = True
        if self._begin_cli.service_is_ready():
            self._begin_cli.call_async(Trigger.Request())
            self.get_logger().info(
                'ArUco world locked — requested player begin_range '
                '(jump to target window).')
        else:
            self.get_logger().warn(
                'world locked but player begin_range service not ready; '
                'playback stays in warm-up loop.')

    def _cup_dets(self, det: SegmentedObjectArray):
        """Yield (mask_u8, bbox, class_name, upright, score) for each cup."""
        out = []
        for o in det.objects:
            name = str(o.class_name).lower()
            if name not in UPRIGHT and name not in FALLEN:
                continue
            try:
                mask = self.bridge.imgmsg_to_cv2(o.mask, 'mono8')
            except Exception:
                continue
            out.append({
                'mask': mask, 'name': name, 'upright': name in UPRIGHT,
                'score': float(o.score),
                'bbox': (int(o.x_min), int(o.y_min), int(o.x_max),
                         int(o.y_max))})
        return out

    @staticmethod
    def _depth_m(depth_img, unit):
        return depth_img.astype(np.float32) * unit

    def _base_valid(self, depth_m):
        return (depth_m > self.z_min) & (depth_m < self.z_max)

    def _grad_ok(self, depth_m):
        gx = cv2.Sobel(depth_m, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(depth_m, cv2.CV_32F, 0, 1, ksize=3)
        return cv2.magnitude(gx, gy) < self.grad_max

    def _cloud(self, mask, depth_m, base_valid, R, t, bgr, intr, *,
               erode_px=0, grad_ok=None, ds=None, const_rgb=None):
        """Deproject a (optionally eroded/gradient-gated) mask to world points.
        Returns (xyz(N,3), rgb_packed(N,)) or (None, None)."""
        mb = mask > 0
        if erode_px > 0:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))
            mb = cv2.erode(mb.astype(np.uint8), k) > 0
        obj = mb & base_valid
        if grad_ok is not None:
            obj = obj & grad_ok
        oy, ox = np.where(obj)
        if oy.size < 8:
            return None, None
        ds = self.downsample if ds is None else ds
        if ds > 1:
            oy, ox = oy[::ds], ox[::ds]
        if oy.size < 8:
            return None, None
        oz = depth_m[oy, ox].astype(np.float64)
        uv = np.stack([ox, oy], axis=1).astype(np.float64)
        pc = intr.deproject_array(uv, oz)
        pw = (R @ pc.T).T + t
        if const_rgb is not None:
            rr = np.full(oy.size, const_rgb[0], np.uint8)
            gg = np.full(oy.size, const_rgb[1], np.uint8)
            bb = np.full(oy.size, const_rgb[2], np.uint8)
            rgb = _pack_rgb(rr, gg, bb)
        else:
            px = bgr[oy, ox]
            rgb = _pack_rgb(px[:, 2], px[:, 1], px[:, 0])
        return pw, rgb

    # ── render driver ───────────────────────────────────────────────────────
    def _on_exo(self, rgb_msg, depth_msg, det_msg) -> None:
        R_we = self._lookup(self.exo_frame, rgb_msg.header.stamp)
        if R_we is None:
            return                      # world not locked yet — wait
        self._maybe_begin_range()

        if self.render_period > 0.0 and self._last_render is not None:
            dt = (self.get_clock().now() - self._last_render).nanoseconds * 1e-9
            if dt < self.render_period:
                return
        self._last_render = self.get_clock().now()

        bgr = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
        depth_m = self._depth_m(depth, self.depth_unit)
        base_valid = self._base_valid(depth_m)
        cups = self._cup_dets(det_msg)
        overlay = bgr.copy()

        self._mid = 0
        markers = [self._delete_all()]
        pts_list, rgb_list = [], []

        R, t = R_we
        try:
            m = self.method
            if m == 1:                   # M0: cloud WITHOUT ArUco correction
                self._m0_nocal(depth_m, base_valid, R, t, bgr,
                               markers, pts_list, rgb_list, overlay)
            elif m == 2:                 # M0.5: same cloud WITH correction
                self._m0_cal(depth_m, base_valid, R, t, bgr,
                             markers, pts_list, rgb_list, overlay)
            elif m == 3:                 # M1
                self._m_raw(cups, depth_m, base_valid, R, t, bgr,
                            markers, pts_list, rgb_list, overlay)
            elif m == 4:                 # M2
                self._m_cone(cups, depth_m, base_valid, R, t, bgr,
                             markers, pts_list, rgb_list, overlay)
            elif m == 5:                 # M3
                self._m_fusion(cups, depth_m, base_valid, R_we, bgr,
                               markers, pts_list, rgb_list, overlay)
            elif m == 6:                 # M4
                self._m_rim(cups, depth_m, base_valid, R, t, bgr,
                            markers, pts_list, rgb_list, overlay)
            else:                        # M5
                self._m_prior(cups, depth_m, base_valid, R, t, bgr,
                              markers, pts_list, rgb_list, overlay)
        except Exception as e:           # never let one bad frame kill the node
            self.get_logger().warn(f'render error (method {self.method}): {e}',
                                   throttle_duration_sec=2.0)

        markers.append(self._title_marker())
        self._publish(markers, pts_list, rgb_list, overlay, rgb_msg.header.stamp)

    def _publish(self, markers, pts_list, rgb_list, overlay, stamp) -> None:
        hdr = Header(stamp=self.get_clock().now().to_msg(),
                     frame_id=self.world_frame)
        if pts_list:
            xyz = np.vstack(pts_list).astype(np.float32)
            rgb = np.concatenate(rgb_list).astype(np.float32)
        else:
            xyz = np.zeros((0, 3), np.float32)
            rgb = np.zeros((0,), np.float32)
        self.pub_points.publish(_make_pointcloud2(hdr, xyz, rgb))
        self.pub_markers.publish(MarkerArray(markers=markers))
        self._banner(overlay)
        msg = self.bridge.cv2_to_imgmsg(overlay, 'bgr8')
        msg.header.stamp = stamp
        msg.header.frame_id = self.exo_frame
        self.pub_overlay.publish(msg)
        self._publish_status()

    # ── M0 / M0.5: whole-scene cloud, ArUco correction OFF vs ON ────────────
    def _full_cloud(self, depth_m, base_valid, bgr, R, t, *, apply_world):
        """Whole near-workspace cloud. apply_world=False keeps RAW camera-optical
        coordinates (ArUco correction NOT applied): published as frame_id=world,
        RViz plots them straight onto the world axes so the cloud floats off the
        table/robot. apply_world=True maps them through camera→world(=robot base)
        so they snap into place — the camera pose is known, we just choose not to
        apply it in M0."""
        obj = base_valid & (depth_m < self.full_max_z)
        oy, ox = np.where(obj)
        if oy.size < 8:
            return None, None
        if self.full_stride > 1:
            oy, ox = oy[::self.full_stride], ox[::self.full_stride]
        if oy.size < 8:
            return None, None
        oz = depth_m[oy, ox].astype(np.float64)
        uv = np.stack([ox, oy], axis=1).astype(np.float64)
        pc = self.intr_exo.deproject_array(uv, oz)      # camera optical frame
        pw = (R @ pc.T).T + t if apply_world else pc
        px = bgr[oy, ox]
        return pw, _pack_rgb(px[:, 2], px[:, 1], px[:, 0])

    def _m0_nocal(self, depth_m, valid, R, t, bgr, markers, P, Rg, ov):
        pw, rgb = self._full_cloud(depth_m, valid, bgr, R, t, apply_world=False)
        if pw is not None:
            P.append(pw)
            Rg.append(rgb)
        markers.append(self._text([0.0, 0.0, 0.15],
                                  'ArUco OFF: camera-frame cloud',
                                  _C(1.0, 0.4, 0.3), h=0.04))
        cv2.putText(ov, 'M0: ArUco correction OFF - raw camera-frame cloud',
                    (12, ov.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (90, 120, 255), 2)

    def _m0_cal(self, depth_m, valid, R, t, bgr, markers, P, Rg, ov):
        pw, rgb = self._full_cloud(depth_m, valid, bgr, R, t, apply_world=True)
        if pw is not None:
            P.append(pw)
            Rg.append(rgb)
        markers.append(self._text([0.0, 0.0, 0.15],
                                  'ArUco ON: world = robot base',
                                  _C(0.3, 1.0, 0.4), h=0.04))
        cv2.putText(ov, 'M0.5: ArUco applied - cloud mapped to world(base)',
                    (12, ov.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (120, 230, 120), 2)

    # ── method 1: raw cloud + loose OBB ──────────────────────────────────────
    def _m_raw(self, cups, depth_m, valid, R, t, bgr, markers, P, Rg, ov):
        for c in cups:
            pw, rgb = self._cloud(c['mask'], depth_m, valid, R, t, bgr,
                                  self.intr_exo, erode_px=0, grad_ok=None)
            if pw is None:
                continue
            P.append(pw)
            Rg.append(rgb)
            box = _compute_box_world(pw, standing_ratio=0.8,
                                     min_elongation=1.5, force_aabb=False)
            if box is None:
                continue
            center, Rb, size, _ = box
            markers.append(self._box(center, Rb, size, _C(1.0, 0.25, 0.2),
                                     line_w=0.004))
            markers.append(self._text(
                center + [0, 0, float(size[2]) * 0.5 + 0.04],
                f'raw OBB {size[0]*100:.0f}x{size[1]*100:.0f}x'
                f'{size[2]*100:.0f}cm', _C(1, 0.5, 0.4)))
            self._draw_mask_outline(ov, c['mask'], (60, 60, 255))

    # ── method 2: cone fit in the (eroded) cloud ───────────────────────────
    def _m_cone(self, cups, depth_m, valid, R, t, bgr, markers, P, Rg, ov):
        # Cone fitting needs the mask-boundary mixed pixels gone, so erode like
        # the production path (mask_erode_px) before searching for the shape —
        # otherwise the least-squares cone chases the noisy rim. (method 1 stays
        # un-eroded to show the raw noise; the heavy part here is the LS fit.)
        for c in cups:
            pw, rgb = self._cloud(c['mask'], depth_m, valid, R, t, bgr,
                                  self.intr_exo, erode_px=self.erode_px,
                                  grad_ok=None)
            if pw is None:
                continue
            P.append(pw)
            Rg.append(rgb)
            if not c['upright']:
                continue
            fit = _fit_cup_axis_xy(pw, top_d=2 * self.r_top,
                                   bot_d=2 * self.r_bot, height=self.cup_h)
            if fit is None:
                continue
            cx, cy, z_base, rmse = fit
            markers.append(self._frustum(cx, cy, z_base, _C(0.1, 1.0, 0.6),
                                         line_w=0.004))
            markers.append(self._text(
                [cx, cy, z_base + self.cup_h + 0.04],
                f'cone fit rms={rmse*1000:.0f}mm', _C(0.3, 1, 0.6)))
            self._draw_mask_outline(ov, c['mask'], (120, 220, 120))

    # ── method 3: exo + hand inverse-covariance fusion ──────────────────────
    def _m_fusion(self, cups, depth_m, valid, R_we, bgr, markers, P, Rg, ov):
        R, t = R_we
        # exo observations (+ blue cloud).
        exo_obs = []
        for c in cups:
            if not c['upright']:
                continue
            pw, rgb = self._cloud(c['mask'], depth_m, valid, R, t, bgr,
                                  self.intr_exo, erode_px=1,
                                  const_rgb=(60, 120, 255))
            if pw is not None:
                P.append(pw)
                Rg.append(rgb)
            est = self._rim_estimate(c['mask'], depth_m, valid, R, t,
                                     self.intr_exo, bgr)
            if est is not None:
                exo_obs.append(est)

        # hand observations (+ orange cloud), via the FK-driven hand TF.
        hand_obs = []
        hand_ok = False
        if self._hand is not None and self.intr_hand is not None:
            hrgb, hdepth, hdet = self._hand
            R_wh = self._lookup(self.hand_frame, hrgb.header.stamp)
            if R_wh is not None:
                hand_ok = True
                Rh, th = R_wh
                hbgr = self.bridge.imgmsg_to_cv2(hrgb, 'bgr8')
                hdm = self._depth_m(
                    self.bridge.imgmsg_to_cv2(hdepth, 'passthrough'),
                    self.depth_unit)
                hvalid = self._base_valid(hdm)
                for c in self._cup_dets(hdet):
                    if not c['upright']:
                        continue
                    pw, rgb = self._cloud(c['mask'], hdm, hvalid, Rh, th, hbgr,
                                          self.intr_hand, erode_px=1,
                                          const_rgb=(255, 150, 40))
                    if pw is not None:
                        P.append(pw)
                        Rg.append(rgb)
                    est = self._rim_estimate(c['mask'], hdm, hvalid, Rh, th,
                                             self.intr_hand, hbgr)
                    if est is not None:
                        hand_obs.append(est)

        used_hand = [False] * len(hand_obs)
        for e in exo_obs:
            # nearest unused hand obs within the same-cup gate.
            j, best = -1, self.cluster_xy
            for k, h in enumerate(hand_obs):
                if used_hand[k]:
                    continue
                d = float(np.linalg.norm(e['xy'] - h['xy']))
                if d < best:
                    best, j = d, k
            z = e['z_base']
            # exo-only estimate: red frustum + (wide) covariance ellipse.
            markers.append(self._frustum(e['xy'][0], e['xy'][1], z,
                                         _C(1.0, 0.3, 0.3, 0.9), line_w=0.003))
            markers.append(self._ellipse(e['xy'], z, e['cov'],
                                         _C(1.0, 0.3, 0.3)))
            if j >= 0:
                used_hand[j] = True
                h = hand_obs[j]
                fused = fuse_xy([(e['xy'], e['cov']), (h['xy'], h['cov'])])
                xy_f, cov_f, _ = fused
                markers.append(self._ellipse(h['xy'], z, h['cov'],
                                             _C(1.0, 0.6, 0.1)))
                markers.append(self._frustum(xy_f[0], xy_f[1], z,
                                             _C(0.1, 1.0, 0.3), line_w=0.005))
                markers.append(self._ellipse(xy_f, z, cov_f, _C(0.1, 1.0, 0.3)))
                markers.append(self._text(
                    [xy_f[0], xy_f[1], z + self.cup_h + 0.05],
                    '[F] exo+hand', _C(0.4, 1.0, 0.5)))
            else:
                markers.append(self._text(
                    [e['xy'][0], e['xy'][1], z + self.cup_h + 0.05],
                    'exo only (uncertain)', _C(1.0, 0.5, 0.5)))
        # Cups only the hand sees (exo is blind to them) — the whole point of
        # adding the wrist camera. Render them as orange hand-only frustums.
        for k, h in enumerate(hand_obs):
            if used_hand[k]:
                continue
            z = h['z_base']
            markers.append(self._frustum(h['xy'][0], h['xy'][1], z,
                                         _C(1.0, 0.6, 0.1), line_w=0.004))
            markers.append(self._ellipse(h['xy'], z, h['cov'], _C(1.0, 0.6, 0.1)))
            markers.append(self._text(
                [h['xy'][0], h['xy'][1], z + self.cup_h + 0.05],
                'hand only (exo blind)', _C(1.0, 0.7, 0.2)))
        note = ('exo(blue)+hand(orange) → [F] fused'
                if hand_ok else 'hand FK unavailable — exo only')
        cv2.putText(ov, note, (12, ov.shape[0] - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (120, 255, 120) if hand_ok else (120, 120, 255), 2)

    # ── method 4: RGB rim/silhouette fit ────────────────────────────────────
    def _m_rim(self, cups, depth_m, valid, R, t, bgr, markers, P, Rg, ov):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        for c in cups:
            if not c['upright']:
                continue
            est = self._rim_estimate(c['mask'], depth_m, valid, R, t,
                                     self.intr_exo, bgr, gray=gray,
                                     keep_fit=True)
            if est is None:
                continue
            x, y, z = est['xy'][0], est['xy'][1], est['z_base']
            markers.append(self._frustum(x, y, z, _C(0.1, 0.9, 1.0),
                                         line_w=0.004))
            markers.append(self._text(
                [x, y, z + self.cup_h + 0.04],
                f"rim iou={est['iou']:.2f} rms={est['rms']:.1f}px",
                _C(0.3, 0.9, 1.0)))
            # 2D overlay: observed contour (green) + fitted silhouette (cyan).
            cnt = largest_contour(c['mask'])
            if cnt is not None:
                cv2.polylines(ov, [cnt.astype(np.int32)], True, (80, 230, 80), 1)
            sil = cone_silhouette_px(
                x, y, z, r_top=self.r_top, r_bot=self.r_bot, height=self.cup_h,
                K=self.intr_exo.K, dist=self.intr_exo.dist, R_wc=R, t_wc=t)
            if sil is not None:
                cv2.polylines(ov, [sil.astype(np.int32)], True, (255, 230, 40), 2)

    # ── method 5: erode + clean cloud + known-shape OBB + target select ─────
    def _m_prior(self, cups, depth_m, valid, R, t, bgr, markers, P, Rg, ov):
        grad_ok = self._grad_ok(depth_m)
        fits = []
        for c in cups:
            if not c['upright']:
                continue
            pw, rgb = self._cloud(c['mask'], depth_m, valid, R, t, bgr,
                                  self.intr_exo, erode_px=self.erode_px,
                                  grad_ok=grad_ok)
            if pw is None or len(pw) < 16:
                continue
            keep = _mad_keep_indices(pw, self.mad_k)
            if keep is not None:
                pw = pw[keep]
                rgb = rgb[keep]
            dk = _filter_spatial_density(pw, self.dens_r, self.dens_n)
            pw, rgb = pw[dk], rgb[dk]
            if len(pw) < 16:
                continue
            P.append(pw)
            Rg.append(rgb)
            fc = _fixed_cup_centroid(pw, cup_radius=self.r_bot, base_pct=5.0)
            if fc is None:
                continue
            cx, cy, z_base, _ = fc
            fits.append((cx, cy, z_base, c['mask']))

        if not fits:
            return
        # Target = the cup nearest the robot base origin (most reachable).
        d = [float(np.hypot(cx, cy)) for cx, cy, _, _ in fits]
        tgt = int(np.argmin(d))
        for i, (cx, cy, z_base, mask) in enumerate(fits):
            is_t = (i == tgt)
            col = _C(0.1, 1.0, 0.3) if is_t else _C(0.3, 0.6, 1.0)
            # Known-cup-sized box used as the grasp prior. Axis-aligned (AABB):
            # an upright cup is a body of revolution, so there is no meaningful
            # yaw — identity rotation is the correct box, not an arbitrary OBB.
            size = np.array([2 * self.r_bot, 2 * self.r_bot, self.cup_h])
            center = np.array([cx, cy, z_base + self.cup_h * 0.5])
            markers.append(self._box(center, np.eye(3), size, col,
                                     line_w=0.006 if is_t else 0.003))
            markers.append(self._frustum(cx, cy, z_base, col,
                                         line_w=0.005 if is_t else 0.002))
            markers.append(self._text(
                [cx, cy, z_base + self.cup_h + 0.05],
                'TARGET ▶ pick' if is_t else 'candidate', col))
            self._draw_mask_outline(
                ov, mask, (80, 255, 120) if is_t else (255, 160, 80))

    def _detected_z_base(self, z: np.ndarray) -> float:
        """Robust cup-base elevation from the DETECTED cloud z: low percentile
        within a band around the cloud median (so table / neighbour bleed below
        the cup is excluded), clipped to >= table_z. This is the measured z
        methods 3/4 render at, instead of pinning every frustum to z=0."""
        mz = float(np.median(z))
        band = z[np.abs(z - mz) < 0.06]          # +/- 6 cm around the cup body
        if band.size == 0:
            band = z
        return max(self.table_z, float(np.percentile(band, 5.0)))

    # ── silhouette estimate shared by methods 3 & 4 ─────────────────────────
    def _rim_estimate(self, mask, depth_m, valid, R, t, intr, bgr, *,
                      gray=None, keep_fit=False):
        pw, _ = self._cloud(mask, depth_m, valid, R, t, bgr, intr, erode_px=1)
        if pw is None or len(pw) < 12:
            return None
        xy0 = (float(np.median(pw[:, 0])), float(np.median(pw[:, 1])))
        # z_base from the DETECTED depth, not a flat table assumption: clip the
        # frustum base to the cup's measured z so stacked / elevated cups sit at
        # their real height instead of every frustum pinned to z=0.
        z0 = self._detected_z_base(pw[:, 2])
        fit = fit_silhouette_xy(
            mask, K=intr.K, dist=intr.dist, R_wc=R, t_wc=t, r_top=self.r_top,
            r_bot=self.r_bot, height=self.cup_h, z_base=z0, xy0=xy0,
            fit_boundary_offset=True)
        if not fit.get('ok'):
            return None
        x, y = float(fit['x']), float(fit['y'])
        if gray is not None:
            snap = edge_snap_fit(
                gray, K=intr.K, dist=intr.dist, R_wc=R, t_wc=t, r_top=self.r_top,
                r_bot=self.r_bot, height=self.cup_h, z_base=z0, xy0=(x, y),
                min_grad=8.0)
            if snap.get('ok'):
                x, y = float(snap['x']), float(snap['y'])
        p = np.array([x, y, z0])
        _, ray = ray_through_point(p, t)
        rng = float(np.linalg.norm(p - t))
        sig = self.rim_sigma_px
        iou = float(fit.get('iou', 0.0))
        if iou < 0.3:
            sig *= 2.0
        cov = xy_cov_from_px(sig, rng, intr.fx, ray)
        return {'xy': np.array([x, y]), 'cov': cov, 'z_base': z0, 'ray': ray,
                'iou': iou, 'rms': float(fit.get('rms_px', 0.0))}

    # ── marker builders ─────────────────────────────────────────────────────
    def _next_id(self) -> int:
        self._mid += 1
        return self._mid

    def _base_marker(self, mtype) -> Marker:
        m = Marker()
        m.header.frame_id = self.world_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'demo'
        m.id = self._next_id()
        m.type = mtype
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        return m

    def _delete_all(self) -> Marker:
        m = Marker()
        m.header.frame_id = self.world_frame
        m.ns = 'demo'
        m.action = Marker.DELETEALL
        return m

    def _box(self, center, R, size, color, line_w=0.004) -> Marker:
        m = self._base_marker(Marker.LINE_LIST)
        m.scale.x = line_w
        m.color = color
        cor = _box_corners(np.asarray(center, float), np.asarray(R, float),
                           np.asarray(size, float))
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7),
                 (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
        for a, b in edges:
            m.points.append(Point(x=float(cor[a, 0]), y=float(cor[a, 1]),
                                  z=float(cor[a, 2])))
            m.points.append(Point(x=float(cor[b, 0]), y=float(cor[b, 1]),
                                  z=float(cor[b, 2])))
        return m

    def _frustum(self, cx, cy, z_base, color, line_w=0.004) -> Marker:
        g = _cup_frustum_geometry(float(cx), float(cy), top_d=2 * self.r_top,
                                  bot_d=2 * self.r_bot, height=self.cup_h,
                                  floor_z=float(z_base), n_seg=self.n_seg)
        m = self._base_marker(Marker.LINE_LIST)
        m.scale.x = line_w
        m.color = color

        def loop(arr):
            for i in range(len(arr) - 1):
                for k in (i, i + 1):
                    m.points.append(Point(x=float(arr[k, 0]),
                                          y=float(arr[k, 1]),
                                          z=float(arr[k, 2])))
        loop(g['top_loop'])
        loop(g['bot_loop'])
        for a, b in g['generatrix']:
            m.points.append(Point(x=float(a[0]), y=float(a[1]), z=float(a[2])))
            m.points.append(Point(x=float(b[0]), y=float(b[1]), z=float(b[2])))
        return m

    def _ellipse(self, xy, z, cov, color) -> Marker:
        m = self._base_marker(Marker.LINE_STRIP)
        m.scale.x = 0.003
        m.color = color
        try:
            w, V = np.linalg.eigh(np.asarray(cov, float))
        except np.linalg.LinAlgError:
            return m
        a = float(np.sqrt(max(w[0], 1e-9)))
        b = float(np.sqrt(max(w[1], 1e-9)))
        # cap so a degenerate cov doesn't draw a screen-filling ring.
        a, b = min(a, 0.3), min(b, 0.3)
        for th in np.linspace(0, 2 * np.pi, 41):
            p = V[:, 0] * (a * np.cos(th)) + V[:, 1] * (b * np.sin(th))
            m.points.append(Point(x=float(xy[0] + p[0]),
                                  y=float(xy[1] + p[1]), z=float(z)))
        return m

    def _text(self, pos, text, color, h=0.03) -> Marker:
        m = self._base_marker(Marker.TEXT_VIEW_FACING)
        m.pose.position = Point(x=float(pos[0]), y=float(pos[1]),
                                z=float(pos[2]))
        m.scale.z = h
        m.color = color
        # RViz TEXT_VIEW_FACING mangles whitespace, so collapse every run of
        # spaces/newlines to a single '_' → each label is one connected token.
        m.text = '_'.join(str(text).split())
        return m

    def _title_marker(self) -> Marker:
        i, en, ko = METHODS[self.method - 1]
        m = self._text([0.45, 0.0, 0.65], f'{i}  {en}\n{ko}',
                       _C(1, 1, 1), h=0.05)
        return m

    # ── 2D overlay helpers ──────────────────────────────────────────────────
    def _draw_mask_outline(self, img, mask, color) -> None:
        cnt = largest_contour(mask)
        if cnt is not None:
            cv2.polylines(img, [cnt.astype(np.int32)], True, color, 1)

    def _banner(self, img) -> None:
        i, en, ko = METHODS[self.method - 1]
        cv2.rectangle(img, (0, 0), (img.shape[1], 34), (20, 20, 20), -1)
        cv2.putText(img, f'{i}  {en}', (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 230, 255), 2)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = EvolutionDemoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
