"""point_cloud_node — fuse RGB + aligned depth + segmentation into:

* `/digital_twin/points`     — coloured PointCloud2 in the `world` frame
* `/digital_twin/boxes`      — MarkerArray of per-object 3D position boxes
                              (CUBE + LINE_LIST outline + TEXT label)
* `/digital_twin/box_debug`  — RGB image with the projected 3D boxes drawn on top

3D box estimation is tailored to cup-like objects: a standing cup is a
near-symmetric cylinder, so PCA yaw is unstable and the box is published
axis-aligned. A fallen cup has a clear elongation in the XY plane, so PCA on
the horizontal projection is used to recover its orientation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point as MsgPoint, Quaternion
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray

import message_filters
import tf2_ros

from depth_digital_twin.intrinsics import load_intrinsics
from depth_digital_twin_msgs.msg import SegmentedObjectArray


# Edge index pairs for the 12 edges of a box given the 8-corner layout used
# by `_box_corners` below (bottom face 0..3, top face 4..7).
_BOX_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def _palette(i: int) -> tuple[float, float, float]:
    base = [
        (0.95, 0.26, 0.21),
        (0.30, 0.69, 0.31),
        (0.13, 0.59, 0.95),
        (1.00, 0.76, 0.03),
        (0.61, 0.15, 0.69),
        (0.00, 0.74, 0.83),
    ]
    return base[i % len(base)]


def _pack_rgb(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pack uint8 RGB into a float32 (PointCloud2 'rgb' field convention)."""
    rgb_uint = ((r.astype(np.uint32) << 16)
                | (g.astype(np.uint32) << 8)
                | b.astype(np.uint32))
    return np.frombuffer(rgb_uint.astype(np.uint32).tobytes(), dtype=np.float32)


class PointCloudNode(Node):
    def __init__(self) -> None:
        super().__init__('point_cloud_node')

        self.declare_parameter('intrinsics_path', '')
        self.declare_parameter('rgb_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic',
                               '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('detections_topic', '/digital_twin/detections')
        self.declare_parameter('points_topic', '/digital_twin/points')
        self.declare_parameter('boxes_topic', '/digital_twin/boxes')
        self.declare_parameter('box_debug_topic', '/digital_twin/box_debug')
        self.declare_parameter('depth_debug_topic', '/digital_twin/depth_debug')
        self.declare_parameter('camera_frame', 'camera_color_optical_frame')
        self.declare_parameter('world_frame', 'world')
        self.declare_parameter('depth_unit', 0.001)
        self.declare_parameter('downsample', 2)
        self.declare_parameter('z_min', 0.1)
        self.declare_parameter('z_max', 4.0)
        self.declare_parameter('box_line_width', 0.0015)
        self.declare_parameter('box_alpha', 0.25)
        self.declare_parameter('box_standing_ratio', 0.8)
        self.declare_parameter('box_min_elongation', 1.5)
        self.declare_parameter('box_force_aabb', False)
        # Pixels to erode the YOLO mask before sampling depth points for the
        # 3D box. YOLO seg boundaries are noisy and depth at object edges is
        # frequently a mixed/foreground+background pixel. 0 disables.
        self.declare_parameter('mask_erode_px', 3)
        # Depth-Laplacian threshold (m) for the mixed-pixel filter. A pixel is
        # rejected when |∇²z| exceeds this — i.e. it sits on a depth discontinuity
        # (cup silhouette vs background table) where the stereo correlator
        # averaged two surfaces and the deprojected point lands outside the
        # true geometry. 0 disables.
        self.declare_parameter('depth_gradient_max_m', 0.015)
        # Per-axis MAD-based outlier filter on the world-frame point cluster
        # before fitting the box. Drops points whose deviation from the median
        # on any axis exceeds k * 1.4826 * MAD (k≈3 ⇒ 3σ for Gaussian noise).
        # Catches single-pixel depth spikes (specular/transparent/mixed pixel)
        # that otherwise inflate the AABB. 0 disables.
        self.declare_parameter('box_outlier_mad_k', 3.0)
        self.declare_parameter('approx_sync_slop', 0.05)
        self.declare_parameter('objects_only', True)

        # ----- Cup model (truncated-cone prior; standing only) -----
        self.declare_parameter('cup_top_diameter_m', 0.054)
        self.declare_parameter('cup_bottom_diameter_m', 0.078)
        self.declare_parameter('cup_height_m', 0.095)
        self.declare_parameter('cup_polygon_segments', 24)
        self.declare_parameter('cup_smoothing_alpha', 0.3)
        self.declare_parameter('cup_track_keepalive_frames', 10)
        self.declare_parameter('cup_fit_residual_max', 0.02)
        self.declare_parameter('cup_class_names', ['cup'])

        # Accumulating-window pipeline. Per-frame depth is too noisy at one
        # shot; we ingest into per-track buffers and only fit + publish at
        # `window_period_s` cadence. With more samples per cluster the MAD
        # filter removes flicker outliers far more reliably and the cup-axis
        # fit becomes stable.
        self.declare_parameter('window_period_s', 0.5)

        # Mirror the floor-patch parameters (owned by world_origin_node, shared
        # via the /**: scope in params.yaml) so we can draw the same patch
        # rectangle on /digital_twin/depth_debug for visual sanity-checking.
        self.declare_parameter('window_radius', 30)
        self.declare_parameter('window_center_x_px', -1)
        self.declare_parameter('window_center_y_px', -1)

        # ArUco marker axes overlay on box_debug (uses calibrated TF, not real-time detect)
        self.declare_parameter('aruco_overlay', True)
        self.declare_parameter('world_marker_length_m', 0.05)

        path = Path(self.get_parameter('intrinsics_path').value)
        if not path.is_file():
            raise FileNotFoundError(f'intrinsics_path not found: {path}')
        self.intr = load_intrinsics(path)
        self.K = self.intr.K

        self.camera_frame: str = self.get_parameter('camera_frame').value
        self.world_frame: str = self.get_parameter('world_frame').value
        self.depth_unit: float = float(self.get_parameter('depth_unit').value)
        self.downsample: int = max(1, int(self.get_parameter('downsample').value))
        self.z_min: float = float(self.get_parameter('z_min').value)
        self.z_max: float = float(self.get_parameter('z_max').value)
        self.box_line_w: float = float(self.get_parameter('box_line_width').value)
        self.box_alpha: float = float(self.get_parameter('box_alpha').value)
        self.standing_ratio: float = float(self.get_parameter('box_standing_ratio').value)
        self.min_elongation: float = float(self.get_parameter('box_min_elongation').value)
        self.force_aabb: bool = bool(self.get_parameter('box_force_aabb').value)
        self.mask_erode_px: int = max(0, int(self.get_parameter('mask_erode_px').value))
        self.depth_grad_max: float = max(
            0.0, float(self.get_parameter('depth_gradient_max_m').value))
        self.outlier_mad_k: float = max(
            0.0, float(self.get_parameter('box_outlier_mad_k').value))
        self.objects_only: bool = bool(self.get_parameter('objects_only').value)
        self.cup_top_d: float = float(self.get_parameter('cup_top_diameter_m').value)
        self.cup_bot_d: float = float(self.get_parameter('cup_bottom_diameter_m').value)
        self.cup_h: float = float(self.get_parameter('cup_height_m').value)
        self.cup_n_seg: int = max(8, int(self.get_parameter('cup_polygon_segments').value))
        self.cup_alpha: float = float(self.get_parameter('cup_smoothing_alpha').value)
        self.cup_keepalive: int = int(self.get_parameter('cup_track_keepalive_frames').value)
        self.cup_resid_max: float = float(self.get_parameter('cup_fit_residual_max').value)
        self.cup_class_names: set[str] = {
            s.lower() for s in self.get_parameter('cup_class_names').value}
        self.patch_radius: int = max(1, int(self.get_parameter('window_radius').value))
        self.patch_cx_px: int = int(self.get_parameter('window_center_x_px').value)
        self.patch_cy_px: int = int(self.get_parameter('window_center_y_px').value)
        self.window_period_s: float = max(
            1e-3, float(self.get_parameter('window_period_s').value))

        # Tracks keyed by Ultralytics ByteTrack instance id (forwarded via
        # SegmentedObject.instance_id from detection_node). Per-track state:
        #   class_name              str
        #   center_xy               np.ndarray (EMA-smoothed cup centre)
        #   z_base                  float | None  (smoothed; None until first fit)
        #   points_buf, colors_buf  list[np.ndarray] — accumulated within window
        #   miss                    int — windows without any new points
        #   last_state              dict | None — last successful fit / render
        #   last_score, last_display_name, last_residual — for the label
        self._tracks: dict[int, dict] = {}
        self._last_published_ids: set[int] = set()
        self._window_start_stamp = None  # rclpy.time.Time, set on first frame

        slop: float = float(self.get_parameter('approx_sync_slop').value)

        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        latched = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)

        self.points_pub = self.create_publisher(
            PointCloud2, self.get_parameter('points_topic').value, 5)
        self.boxes_pub = self.create_publisher(
            MarkerArray, self.get_parameter('boxes_topic').value, latched)
        self.box_debug_pub = self.create_publisher(
            Image, self.get_parameter('box_debug_topic').value, 1)
        self.depth_debug_pub = self.create_publisher(
            Image, self.get_parameter('depth_debug_topic').value, 1)

        rgb_sub = message_filters.Subscriber(
            self, Image, self.get_parameter('rgb_topic').value)
        depth_sub = message_filters.Subscriber(
            self, Image, self.get_parameter('depth_topic').value)
        det_sub = message_filters.Subscriber(
            self, SegmentedObjectArray, self.get_parameter('detections_topic').value)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub, det_sub], queue_size=10, slop=slop)
        self.sync.registerCallback(self._on_synced)
        self.get_logger().info('point_cloud_node ready (waiting for synced frames)')

        # ── ArUco axis overlay (TF-based, not real-time detection) ────────
        self._aruco_overlay = bool(self.get_parameter('aruco_overlay').value)
        self._aruco_axis_len = float(
            self.get_parameter('world_marker_length_m').value) * 0.8

    # ------------------------------------------------------------------
    def _on_synced(self, rgb_msg: Image, depth_msg: Image,
                   det_msg: SegmentedObjectArray) -> None:
        """Per-frame: ingest detections into per-track buffers + emit live
        debug images. Heavy work (filter, fit, marker/cloud publish) is
        deferred to `_finalize_window` which fires every `window_period_s`."""
        try:
            tf = self.tf_buffer.lookup_transform(
                self.world_frame, self.camera_frame, rclpy.time.Time())
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            self.get_logger().warn(
                f'TF {self.world_frame}<-{self.camera_frame} not available yet',
                throttle_duration_sec=2.0)
            return

        # tf is target=world, source=camera ⇒ p_world = R_wc @ p_cam + t_wc.
        t_wc = np.array([tf.transform.translation.x,
                         tf.transform.translation.y,
                         tf.transform.translation.z], dtype=np.float64)
        q = tf.transform.rotation
        R_wc = _quat_to_rot(q.x, q.y, q.z, q.w)

        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        h, w = depth.shape[:2]
        if rgb.shape[:2] != (h, w):
            self.get_logger().warn_once(
                f'RGB ({rgb.shape[:2]}) != depth ({h, w}); requires aligned depth')
            return

        z = depth.astype(np.float32) * self.depth_unit
        valid = (z > self.z_min) & (z < self.z_max)
        if self.depth_grad_max > 0.0:
            # Mixed-pixel reject: |∇²z| at object silhouettes is large because
            # the stereo correlator averaged the cup surface with the background.
            # Killing those pixels collapses the inflated "ring" outside the cup.
            gz = cv2.Laplacian(z, cv2.CV_32F, ksize=3)
            valid &= np.abs(gz) < self.depth_grad_max

        union_mask = np.zeros((h, w), dtype=bool)
        per_object_masks: list[tuple[object, np.ndarray]] = []
        for obj in det_msg.objects:
            m = self.bridge.imgmsg_to_cv2(obj.mask, desired_encoding='mono8')
            if m.shape[:2] != (h, w):
                continue
            mb = m > 0
            union_mask |= mb
            per_object_masks.append((obj, mb))

        # Per-frame: depth debug stream is independent of detection success.
        self._publish_depth_debug(depth_msg, z, valid, union_mask)

        # Per-frame: ingest each detection's world points into its track's
        # accumulating buffer. No fitting / cloud / marker publish here — the
        # window timer below batches that into one ~window_period_s update.
        for obj, mb in per_object_masks:
            mb_box = self._erode_mask(mb)
            if mb_box.sum() < 32:
                mb_box = mb
            obj_valid = mb_box & valid
            if obj_valid.sum() < 32:
                continue
            oy, ox = np.where(obj_valid)
            oz = z[oy, ox]
            ocx_c = (ox.astype(np.float32) - self.intr.cx) * oz / self.intr.fx
            ocy_c = (oy.astype(np.float32) - self.intr.cy) * oz / self.intr.fy
            obj_cam = np.stack([ocx_c, ocy_c, oz], axis=1)
            obj_world = (R_wc @ obj_cam.T).T + t_wc
            if obj_world.shape[0] < 16:
                continue
            bgr = rgb[oy, ox]
            obj_rgb_packed = _pack_rgb(bgr[:, 2], bgr[:, 1], bgr[:, 0])

            inst_id = int(getattr(obj, 'instance_id', -1))
            if inst_id < 0:
                # ByteTrack hasn't promoted this detection yet — skip rather
                # than mint a synthetic id that would collide with future
                # tracker ids.
                continue
            class_name = (obj.class_name or '').lower()
            centroid_xy = np.median(obj_world[:, :2], axis=0)
            tid = inst_id
            track = self._tracks.get(tid)
            if track is None:
                track = {
                    'class_name': class_name,
                    'center_xy': np.asarray(centroid_xy, dtype=np.float64).copy(),
                    'z_base': None,
                    'points_buf': [],
                    'colors_buf': [],
                    'miss': 0,
                    'last_state': None,
                    'last_score': 0.0,
                    'last_display_name': obj.class_name or class_name,
                    'last_residual': 0.0,
                }
                self._tracks[tid] = track
            track['points_buf'].append(obj_world)
            track['colors_buf'].append(obj_rgb_packed)
            track['last_score'] = float(obj.score)
            track['last_display_name'] = obj.class_name

        # Per-frame box debug overlay using the LAST window's fit (frozen
        # box / frustum between updates is expected — they refresh every
        # window_period_s).
        debug_img = rgb.copy()
        n_drawn = 0
        for tid, track in self._tracks.items():
            ls = track.get('last_state')
            if ls is None:
                continue
            colour = _palette(tid - 1)
            self._draw_box_overlay(
                debug_img, ls['center'], ls['R'], ls['size'], ls['top_world'],
                colour, ls['label'], R_wc, t_wc)
            if ls.get('frustum') is not None:
                self._draw_frustum_overlay(
                    debug_img, ls['frustum'], colour, R_wc, t_wc)
            n_drawn += 1
        self._annotate_status(debug_img, n_drawn)
        self._draw_aruco_axes(debug_img)
        self._publish_debug(debug_img, rgb_msg.header)

        # Window check — finalize after window_period_s elapsed.
        now = self.get_clock().now()
        if self._window_start_stamp is None:
            self._window_start_stamp = now
            return
        elapsed = (now - self._window_start_stamp).nanoseconds * 1e-9
        if elapsed >= self.window_period_s:
            self._finalize_window(rgb_msg.header.stamp)
            self._window_start_stamp = now

    # ------------------------------------------------------------------
    def _finalize_window(self, stamp) -> None:
        """End-of-window: aggregate each track's accumulated points, MAD-filter,
        re-fit the cup pose (or OBB fallback), then publish the union of all
        tracks' filtered points and a fresh marker set. This is the only path
        that publishes /points and /boxes."""
        alive_xyz: list[np.ndarray] = []
        alive_rgb: list[np.ndarray] = []

        for tid in list(self._tracks.keys()):
            track = self._tracks[tid]
            buf_pts = track['points_buf']
            buf_cols = track['colors_buf']
            track['points_buf'] = []
            track['colors_buf'] = []

            if not buf_pts:
                track['miss'] += 1
                if track['miss'] > self.cup_keepalive:
                    self._tracks.pop(tid, None)
                continue

            all_pts = np.vstack(buf_pts)
            all_rgb = np.concatenate(buf_cols)
            # MAD filter on the aggregated cluster — much more robust than
            # per-frame because median + MAD have many more samples to anchor.
            keep = _mad_keep_indices(all_pts, self.outlier_mad_k)
            if keep is not None:
                all_pts = all_pts[keep]
                all_rgb = all_rgb[keep]
            if all_pts.shape[0] < 32:
                track['miss'] += 1
                if track['miss'] > self.cup_keepalive:
                    self._tracks.pop(tid, None)
                continue
            track['miss'] = 0

            ls = self._fit_and_render_state(tid, track, all_pts)
            if ls is not None:
                track['last_state'] = ls

            alive_xyz.append(all_pts.astype(np.float32))
            alive_rgb.append(all_rgb)

        # Combined cloud — every (filtered) accumulated point is plotted.
        if alive_xyz:
            cloud_xyz = np.vstack(alive_xyz)
            cloud_rgb = np.concatenate(alive_rgb).astype(np.float32)
        else:
            cloud_xyz = np.zeros((0, 3), dtype=np.float32)
            cloud_rgb = np.zeros((0,), dtype=np.float32)
        self.points_pub.publish(_make_pointcloud2(
            header=Header(stamp=stamp, frame_id=self.world_frame),
            xyz=cloud_xyz, rgb=cloud_rgb))

        # Marker emission — one update per window. DELETE for evicted tracks.
        markers = MarkerArray()
        alive_ids: set[int] = set()
        for tid, track in sorted(self._tracks.items()):
            ls = track.get('last_state')
            if ls is None:
                continue
            colour = _palette(tid - 1)
            self._append_box_markers(
                markers, tid, ls['center'], ls['R'], ls['size'],
                ls['top_world'], colour, ls['label'], stamp)
            if ls.get('frustum') is not None:
                self._append_cup_frustum_markers(
                    markers, tid, ls['frustum'], colour, stamp)
            alive_ids.add(tid)
        stale = self._last_published_ids - alive_ids
        for tid in stale:
            self._append_delete_markers(markers, tid, stamp)
        self._last_published_ids = alive_ids
        self.boxes_pub.publish(markers)

    # ------------------------------------------------------------------
    def _fit_and_render_state(self, tid: int, track: dict,
                              all_pts: np.ndarray):
        """Cup fit (with OBB fallback). EMA-smooth tracked centre + z_base.
        Return a dict with the geometry + label needed for marker / overlay
        emission, or None if neither cup nor OBB fit succeeds.

        Labels use underscore separators so each line is a single
        whitespace-free token — TEXT_VIEW_FACING markers spread spaces too
        wide and the label drifts horizontally across the screen otherwise.
        """
        class_name = track['class_name']
        cup_kind = (class_name in self.cup_class_names)
        cup_done = False
        cx_smooth = float(track['center_xy'][0])
        cy_smooth = float(track['center_xy'][1])
        z_base_smooth = track.get('z_base')
        residual = 0.0

        if cup_kind:
            fit = _fit_cup_axis_xy(
                all_pts, top_d=self.cup_top_d, bot_d=self.cup_bot_d,
                height=self.cup_h)
            if fit is not None and fit[3] <= self.cup_resid_max:
                cx_new, cy_new, z_base_new, residual = fit
                a = self.cup_alpha
                if z_base_smooth is None:
                    cx_smooth, cy_smooth, z_base_smooth = (
                        cx_new, cy_new, z_base_new)
                else:
                    cx_smooth = a * cx_new + (1.0 - a) * cx_smooth
                    cy_smooth = a * cy_new + (1.0 - a) * cy_smooth
                    z_base_smooth = a * z_base_new + (1.0 - a) * z_base_smooth
                track['center_xy'] = np.array(
                    [cx_smooth, cy_smooth], dtype=np.float64)
                track['z_base'] = z_base_smooth
                track['last_residual'] = residual
                cup_done = True

        if cup_done:
            cz = z_base_smooth + 0.5 * self.cup_h
            center = np.array([cx_smooth, cy_smooth, cz], dtype=np.float64)
            R_box = np.eye(3)
            d_max = max(self.cup_top_d, self.cup_bot_d)
            size = np.array([d_max, d_max, self.cup_h], dtype=np.float64)
            top_world = np.array(
                [cx_smooth, cy_smooth, z_base_smooth + self.cup_h],
                dtype=np.float64)
            frustum = _cup_frustum_geometry(
                cx_smooth, cy_smooth, top_d=self.cup_top_d,
                bot_d=self.cup_bot_d, height=self.cup_h,
                floor_z=z_base_smooth, n_seg=self.cup_n_seg)
            r_mm = residual * 1000.0
            line1 = f"#{tid}_{track['last_display_name']}_{track['last_score']:.2f}"
            line2 = (f"r={r_mm:.0f}mm_"
                     f"({cx_smooth:.2f},{cy_smooth:.2f},{top_world[2]:.2f})")
            label = line1.replace(' ', '_') + '\n' + line2.replace(' ', '_')
            return {
                'center': center, 'R': R_box, 'size': size,
                'top_world': top_world, 'frustum': frustum, 'label': label,
            }

        # Fallback: OBB / AABB on aggregated points (non-cup or fit failed).
        box = _compute_box_world(
            all_pts,
            standing_ratio=self.standing_ratio,
            min_elongation=self.min_elongation,
            force_aabb=self.force_aabb)
        if box is None:
            return None
        center, R_box, size, pose_label = box
        top_world = center + R_box @ np.array(
            [0.0, 0.0, float(size[2]) * 0.5], dtype=np.float64)
        track['center_xy'] = center[:2].astype(np.float64)
        line1 = (f"#{tid}_{track['last_display_name']}_"
                 f"{track['last_score']:.2f}_[{pose_label}]")
        line2 = (f"({top_world[0]:.2f},{top_world[1]:.2f},"
                 f"{top_world[2]:.2f})")
        label = line1.replace(' ', '_') + '\n' + line2.replace(' ', '_')
        return {
            'center': center, 'R': R_box, 'size': size,
            'top_world': top_world, 'frustum': None, 'label': label,
        }

    # ------------------------------------------------------------------
    def _append_box_markers(self, markers: MarkerArray, idx: int,
                            center: np.ndarray, R_box: np.ndarray,
                            size: np.ndarray, top_world: np.ndarray,
                            colour: tuple[float, float, float],
                            label: str, stamp) -> None:
        qx, qy, qz, qw = _rot_to_quat(R_box)

        cube = Marker()
        cube.header.frame_id = self.world_frame
        cube.header.stamp = stamp
        cube.ns = 'boxes'
        cube.id = idx
        cube.type = Marker.CUBE
        cube.action = Marker.ADD
        cube.pose.position = MsgPoint(
            x=float(center[0]), y=float(center[1]), z=float(center[2]))
        cube.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
        cube.scale.x = float(max(size[0], 1e-3))
        cube.scale.y = float(max(size[1], 1e-3))
        cube.scale.z = float(max(size[2], 1e-3))
        cube.color = ColorRGBA(
            r=colour[0], g=colour[1], b=colour[2], a=float(self.box_alpha))
        markers.markers.append(cube)

        outline = Marker()
        outline.header = cube.header
        outline.ns = 'box_outline'
        outline.id = idx
        outline.type = Marker.LINE_LIST
        outline.action = Marker.ADD
        outline.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        outline.scale.x = self.box_line_w
        outline.color = ColorRGBA(r=colour[0], g=colour[1], b=colour[2], a=1.0)
        corners = _box_corners(center, R_box, size)
        for a, b in _BOX_EDGES:
            outline.points.append(MsgPoint(
                x=float(corners[a, 0]), y=float(corners[a, 1]), z=float(corners[a, 2])))
            outline.points.append(MsgPoint(
                x=float(corners[b, 0]), y=float(corners[b, 1]), z=float(corners[b, 2])))
        markers.markers.append(outline)

        # Sphere marker at the box top-centre (pivot (0,0,+1) in box-local).
        top = Marker()
        top.header = cube.header
        top.ns = 'box_top'
        top.id = idx
        top.type = Marker.SPHERE
        top.action = Marker.ADD
        top.pose.position = MsgPoint(
            x=float(top_world[0]), y=float(top_world[1]), z=float(top_world[2]))
        top.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        top.scale.x = top.scale.y = top.scale.z = 0.02
        top.color = ColorRGBA(r=1.0, g=0.95, b=0.0, a=1.0)
        markers.markers.append(top)

        text = Marker()
        text.header = cube.header
        text.ns = 'box_labels'
        text.id = idx
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        # Stagger label Z by track id so coplanar cups don't overlap. Two
        # rows alternate (0.04, 0.09 m above the top centre).
        z_offset = 0.04 + 0.05 * ((idx - 1) % 2)
        text.pose.position = MsgPoint(
            x=float(top_world[0]),
            y=float(top_world[1]),
            z=float(top_world[2] + z_offset))
        text.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        text.scale.z = 0.025
        text.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        text.text = label
        markers.markers.append(text)

    # ------------------------------------------------------------------
    def _draw_box_overlay(self, img: np.ndarray, center: np.ndarray,
                          R_box: np.ndarray, size: np.ndarray,
                          top_world: np.ndarray,
                          colour: tuple[float, float, float], label: str,
                          R_wc: np.ndarray, t_wc: np.ndarray) -> None:
        corners_world = _box_corners(center, R_box, size)
        # world -> camera: p_c = R_wc^T @ (p_w - t_wc)
        corners_cam = (R_wc.T @ (corners_world - t_wc).T).T
        in_front = corners_cam[:, 2] > 0.05
        if not np.any(in_front):
            return
        z_safe = np.clip(corners_cam[:, 2], 1e-6, None)
        pix = (self.K @ corners_cam.T).T
        pix = pix[:, :2] / z_safe[:, None]
        pts = pix.astype(int)
        bgr_colour = (int(colour[2] * 255), int(colour[1] * 255), int(colour[0] * 255))
        for a, b in _BOX_EDGES:
            if not (in_front[a] and in_front[b]):
                continue
            cv2.line(img, tuple(pts[a]), tuple(pts[b]), bgr_colour, 2)

        # Project the top-centre point and draw a marker on the image.
        top_cam = R_wc.T @ (top_world - t_wc)
        top_label_anchor = None
        if top_cam[2] > 0.05:
            tp = (self.K @ top_cam) / max(float(top_cam[2]), 1e-6)
            tx, ty = int(round(tp[0])), int(round(tp[1]))
            cv2.circle(img, (tx, ty), 6, (0, 240, 255), -1)
            cv2.circle(img, (tx, ty), 7, (0, 0, 0), 1)
            top_label_anchor = (tx, ty)

        if top_label_anchor is not None:
            tx, ty = top_label_anchor
            cv2.putText(img, label, (tx + 8, max(0, ty - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 240, 255), 1,
                        lineType=cv2.LINE_AA)
        elif np.any(in_front):
            anchor = pts[in_front][np.argmin(pix[in_front, 1])]
            ax, ay = int(anchor[0]), max(0, int(anchor[1]) - 6)
            cv2.putText(img, label, (ax, ay),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, bgr_colour, 2)

    # ------------------------------------------------------------------
    def _append_cup_frustum_markers(self, markers: MarkerArray, idx: int,
                                    frustum: dict,
                                    colour: tuple[float, float, float],
                                    stamp) -> None:
        col = ColorRGBA(r=colour[0], g=colour[1], b=colour[2], a=1.0)

        def _loop_marker(ns: str, loop: np.ndarray) -> Marker:
            m = Marker()
            m.header.frame_id = self.world_frame
            m.header.stamp = stamp
            m.ns = ns
            m.id = idx
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            m.scale.x = self.box_line_w
            m.color = col
            for p in loop:
                m.points.append(MsgPoint(
                    x=float(p[0]), y=float(p[1]), z=float(p[2])))
            return m

        markers.markers.append(_loop_marker('cup_top_loop', frustum['top_loop']))
        markers.markers.append(_loop_marker('cup_bot_loop', frustum['bot_loop']))

        gen = Marker()
        gen.header.frame_id = self.world_frame
        gen.header.stamp = stamp
        gen.ns = 'cup_generatrix'
        gen.id = idx
        gen.type = Marker.LINE_LIST
        gen.action = Marker.ADD
        gen.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        gen.scale.x = self.box_line_w
        gen.color = col
        for p_top, p_bot in frustum['generatrix']:
            gen.points.append(MsgPoint(
                x=float(p_top[0]), y=float(p_top[1]), z=float(p_top[2])))
            gen.points.append(MsgPoint(
                x=float(p_bot[0]), y=float(p_bot[1]), z=float(p_bot[2])))
        markers.markers.append(gen)

    def _draw_frustum_overlay(self, img: np.ndarray, frustum: dict,
                              colour: tuple[float, float, float],
                              R_wc: np.ndarray, t_wc: np.ndarray) -> None:
        bgr_colour = (int(colour[2] * 255), int(colour[1] * 255), int(colour[0] * 255))

        def _project(loop_world: np.ndarray):
            cam = (R_wc.T @ (loop_world - t_wc).T).T
            ok = cam[:, 2] > 0.05
            zs = np.clip(cam[:, 2], 1e-6, None)
            pix = (self.K @ cam.T).T
            pix = pix[:, :2] / zs[:, None]
            return pix.astype(int), ok

        for loop in (frustum['top_loop'], frustum['bot_loop']):
            pts, ok = _project(loop)
            for k in range(len(pts) - 1):
                if ok[k] and ok[k + 1]:
                    cv2.line(img, tuple(pts[k]), tuple(pts[k + 1]),
                             bgr_colour, 1, lineType=cv2.LINE_AA)
        for p_top, p_bot in frustum['generatrix']:
            pair = np.stack([p_top, p_bot], axis=0)
            pts, ok = _project(pair)
            if ok[0] and ok[1]:
                cv2.line(img, tuple(pts[0]), tuple(pts[1]),
                         bgr_colour, 1, lineType=cv2.LINE_AA)

    def _append_delete_markers(self, markers: MarkerArray, idx: int,
                               stamp) -> None:
        for ns in ('boxes', 'box_outline', 'box_top', 'box_labels',
                   'cup_top_loop', 'cup_bot_loop', 'cup_generatrix'):
            d = Marker()
            d.header.frame_id = self.world_frame
            d.header.stamp = stamp
            d.ns = ns
            d.id = idx
            d.action = Marker.DELETE
            markers.markers.append(d)

    # ------------------------------------------------------------------
    def _erode_mask(self, mb: np.ndarray) -> np.ndarray:
        """Shrink the YOLO mask by `mask_erode_px` to drop edge pixels whose
        depth is unreliable (mixed foreground/background). No-op if disabled."""
        if self.mask_erode_px <= 0:
            return mb
        k = self.mask_erode_px
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        eroded = cv2.erode(mb.astype(np.uint8), kernel, iterations=1)
        return eroded > 0

    def _draw_aruco_axes(self, img: np.ndarray) -> None:
        """Project calibrated world (base) and ArUco marker frames onto the debug image.

        Both transforms are published once by world_origin_node at calibration
        time and looked up from the TF tree — no real-time marker detection needed.
        """
        if not self._aruco_overlay:
            return
        dist = getattr(self.intr, 'dist', None)

        def _project_label(tvec_3: np.ndarray, text: str, colour) -> None:
            z = float(tvec_3[2])
            if z < 0.01:
                return
            px = self.intr.K @ tvec_3 / z
            cv2.putText(img, text, (int(px[0]) + 8, int(px[1]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2, cv2.LINE_AA)

        # ── world frame = robot base origin ──────────────────────────────
        try:
            tf_w = self.tf_buffer.lookup_transform(
                self.camera_frame, self.world_frame, rclpy.time.Time())
            tw = tf_w.transform.translation
            qw = tf_w.transform.rotation
            tvec_w = np.array([[tw.x], [tw.y], [tw.z]], dtype=np.float64)
            if float(tw.z) > 0.01:
                R_cw = _quat_to_rot(qw.x, qw.y, qw.z, qw.w)
                rvec_w, _ = cv2.Rodrigues(R_cw)
                cv2.drawFrameAxes(img, self.intr.K, dist, rvec_w, tvec_w,
                                  self._aruco_axis_len * 1.5, thickness=3)
                _project_label(np.array([tw.x, tw.y, tw.z]),
                               'base', (255, 255, 255))
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            pass

        # ── aruco marker frame ────────────────────────────────────────────
        try:
            tf_a = self.tf_buffer.lookup_transform(
                self.camera_frame, 'aruco', rclpy.time.Time())
            ta = tf_a.transform.translation
            qa = tf_a.transform.rotation
            tvec_a = np.array([[ta.x], [ta.y], [ta.z]], dtype=np.float64)
            if float(ta.z) > 0.01:
                R_ca = _quat_to_rot(qa.x, qa.y, qa.z, qa.w)
                rvec_a, _ = cv2.Rodrigues(R_ca)
                cv2.drawFrameAxes(img, self.intr.K, dist, rvec_a, tvec_a,
                                  self._aruco_axis_len, thickness=2)
                _project_label(np.array([ta.x, ta.y, ta.z]),
                               'aruco', (0, 255, 255))
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            pass

    def _publish_clear_markers(self, stamp) -> None:
        clear = MarkerArray()
        d = Marker()
        d.header.frame_id = self.world_frame
        d.header.stamp = stamp
        d.action = Marker.DELETEALL
        clear.markers.append(d)
        self.boxes_pub.publish(clear)

    def _publish_debug(self, img: np.ndarray, src_header) -> None:
        msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        msg.header = src_header
        self.box_debug_pub.publish(msg)

    def _publish_depth_debug(self, depth_msg: Image, z_m: np.ndarray,
                             valid: np.ndarray, union_mask: np.ndarray) -> None:
        """JET-colormapped depth view for live debugging. Pixels outside
        [z_min, z_max] (or zero depth from the sensor) are blacked out;
        detection mask outlines are drawn in white so it's obvious whether
        depth is dropping out *inside* the object silhouette."""
        norm = np.zeros_like(z_m, dtype=np.uint8)
        if bool(valid.any()):
            zspan = max(self.z_max - self.z_min, 1e-6)
            scaled = np.clip((z_m - self.z_min) / zspan, 0.0, 1.0)
            norm[valid] = (scaled[valid] * 255.0).astype(np.uint8)
        color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        # Black-out invalids — applyColorMap on 0 gives dark blue otherwise,
        # which is hard to distinguish from "near the camera".
        color[~valid] = (0, 0, 0)
        if union_mask is not None and bool(union_mask.any()):
            edges = cv2.Canny((union_mask.astype(np.uint8) * 255), 50, 150)
            color[edges > 0] = (255, 255, 255)
        # Floor-patch overlay: rectangle + centre crosshair + median depth so
        # we can immediately see whether world_origin_node is sampling depth
        # over actual floor (and not, say, a hand or a chair leg).
        h, w = z_m.shape[:2]
        cx = w // 2 if self.patch_cx_px < 0 else self.patch_cx_px
        cy = h // 2 if self.patch_cy_px < 0 else self.patch_cy_px
        cx = int(np.clip(cx, 0, w - 1))
        cy = int(np.clip(cy, 0, h - 1))
        r = self.patch_radius
        x0, y0 = max(0, cx - r), max(0, cy - r)
        x1, y1 = min(w - 1, cx + r), min(h - 1, cy + r)
        cv2.rectangle(color, (x0, y0), (x1, y1), (0, 255, 255), 1)
        cv2.drawMarker(color, (cx, cy), (0, 255, 255),
                       markerType=cv2.MARKER_CROSS, markerSize=12, thickness=1)
        patch_z = z_m[y0:y1 + 1, x0:x1 + 1]
        patch_valid = valid[y0:y1 + 1, x0:x1 + 1]
        if bool(patch_valid.any()):
            med = float(np.median(patch_z[patch_valid]))
            patch_label = (f'patch {2 * r + 1}x{2 * r + 1}px @({cx},{cy}) '
                           f'med={med:.3f}m  valid={int(patch_valid.sum())}')
        else:
            patch_label = (f'patch {2 * r + 1}x{2 * r + 1}px @({cx},{cy}) '
                           f'NO VALID DEPTH')
        cv2.putText(color, patch_label, (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1,
                    cv2.LINE_AA)

        n_valid = int(valid.sum())
        n_total = int(valid.size)
        cv2.putText(
            color,
            f'depth valid {n_valid}/{n_total} ({100.0 * n_valid / max(n_total, 1):.0f}%) '
            f'  range [{self.z_min:.2f},{self.z_max:.2f}] m',
            (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            cv2.LINE_AA)
        msg = self.bridge.cv2_to_imgmsg(color, encoding='bgr8')
        msg.header = depth_msg.header
        self.depth_debug_pub.publish(msg)

    @staticmethod
    def _annotate_status(img: np.ndarray, n: int) -> None:
        colour = (0, 255, 0) if n else (0, 200, 255)
        cv2.putText(img, f'3d boxes={n}', (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)


# ----------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------
def _fit_cup_axis_xy(points: np.ndarray, *, top_d: float, bot_d: float,
                     height: float):
    """Algebraic LS fit of the cup axis (cx, cy) and base elevation z_base
    given a vertical truncated-cone prior.

    The cup may stand on any horizontal surface — table, shelf, floor — so
    we don't assume a global floor height. The cluster's robust 5th-percentile
    Z is treated as the cup base; r(z) interpolates between r_bot at z_base
    and r_top at z_base+height. Every surface point satisfies
        (x - cx)^2 + (y - cy)^2 = r(z)^2
    so expanding gives a linear system in (cx, cy, C=cx^2+cy^2):
        -2*cx*x - 2*cy*y + C = r(z)^2 - x^2 - y^2

    Returns (cx, cy, z_base, rmse_residual_m) or None if degenerate. The
    visible side alone is enough — radius variation along z constrains the
    centre even from a one-sided arc.
    """
    if points.shape[0] < 16 or height <= 1e-6:
        return None
    x = points[:, 0]
    y = points[:, 1]
    # Robust cup base: 5th percentile is resilient to a few stray low pixels
    # (e.g. table-edge bleed) without being pulled by upper noise.
    z_base = float(np.percentile(points[:, 2], 5.0))
    z_rel = np.clip(points[:, 2] - z_base, 0.0, height)
    r_bot = bot_d * 0.5
    r_top = top_d * 0.5
    r_z = r_bot + (r_top - r_bot) * (z_rel / height)
    A = np.column_stack([-2.0 * x, -2.0 * y, np.ones_like(x)])
    b = r_z ** 2 - x ** 2 - y ** 2
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy, _ = (float(v) for v in sol)
    if not (np.isfinite(cx) and np.isfinite(cy)):
        return None
    rho = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    rmse = float(np.sqrt(np.mean((rho - r_z) ** 2)))
    return cx, cy, z_base, rmse


def _cup_frustum_geometry(cx: float, cy: float, *, top_d: float, bot_d: float,
                          height: float, floor_z: float, n_seg: int) -> dict:
    """Pre-compute the world-frame vertices of the cup frustum wireframe used
    by the markers: closed top/bottom loops + a few vertical generatrix lines.
    """
    angles = np.linspace(0.0, 2.0 * np.pi, n_seg + 1)
    r_top = top_d * 0.5
    r_bot = bot_d * 0.5
    z_top = floor_z + height
    z_bot = floor_z
    top_loop = np.stack([
        cx + r_top * np.cos(angles),
        cy + r_top * np.sin(angles),
        np.full_like(angles, z_top),
    ], axis=1)
    bot_loop = np.stack([
        cx + r_bot * np.cos(angles),
        cy + r_bot * np.sin(angles),
        np.full_like(angles, z_bot),
    ], axis=1)
    n_gen = min(8, n_seg)
    gen_idx = np.linspace(0, n_seg, n_gen, endpoint=False).astype(int)
    pairs = [(top_loop[i].copy(), bot_loop[i].copy()) for i in gen_idx]
    return {'top_loop': top_loop, 'bot_loop': bot_loop, 'generatrix': pairs}


def _mad_keep_indices(points: np.ndarray, mad_k: float):
    """Boolean mask of points within `mad_k * 1.4826 * MAD` per axis. Returns
    None when the threshold is disabled (mad_k <= 0), the cluster is too
    small to compute a robust median, or filtering would discard so much
    data that downstream fitting becomes unstable."""
    if mad_k <= 0.0 or points.shape[0] < 16:
        return None
    med = np.median(points, axis=0)
    abs_dev = np.abs(points - med)
    mad = np.median(abs_dev, axis=0)
    threshold = mad_k * 1.4826 * np.maximum(mad, 1e-6)
    keep = np.all(abs_dev <= threshold, axis=1)
    if int(keep.sum()) < 16:
        return None
    return keep


def _filter_outliers(points: np.ndarray, mad_k: float) -> np.ndarray:
    """Drop points whose per-axis absolute deviation from the median exceeds
    `mad_k * 1.4826 * MAD` (≈ k·σ under Gaussian noise). Returns the filtered
    subset; falls back to the original cluster if filtering would leave too
    few points to fit a box."""
    if mad_k <= 0.0 or points.shape[0] < 16:
        return points
    med = np.median(points, axis=0)
    abs_dev = np.abs(points - med)
    mad = np.median(abs_dev, axis=0)
    # Avoid divide-by-zero on a perfectly flat axis.
    threshold = mad_k * 1.4826 * np.maximum(mad, 1e-6)
    keep = np.all(abs_dev <= threshold, axis=1)
    if int(keep.sum()) < 16:
        return points
    return points[keep]


def _compute_box_world(points: np.ndarray, *,
                       standing_ratio: float, min_elongation: float,
                       force_aabb: bool):
    """Estimate a 3D position box for a cluster of world-frame points.

    Returns (center(3,), R(3,3), size(3,), pose_label) or None.

    For cup-like targets: treat tall clusters (z extent dominates) as
    standing → axis-aligned box (yaw=0). Otherwise project to the XY plane
    and inspect PCA elongation; only commit to a yaw rotation when the
    principal axis is clearly longer than the secondary one.
    """
    if points.shape[0] < 32:
        return None

    pmin = points.min(axis=0)
    pmax = points.max(axis=0)
    extent = pmax - pmin
    z_ext = float(extent[2])
    h_ext = float(max(extent[0], extent[1]))
    aabb_center = (pmin + pmax) * 0.5

    if force_aabb or h_ext < 1e-6:
        return aabb_center, np.eye(3), extent, 'standing' if z_ext >= h_ext else 'unknown'

    if z_ext / max(h_ext, 1e-6) > standing_ratio:
        return aabb_center, np.eye(3), extent, 'standing'

    xy = points[:, :2]
    xy_centered = xy - xy.mean(axis=0)
    cov = np.cov(xy_centered.T)
    if not np.all(np.isfinite(cov)):
        return aabb_center, np.eye(3), extent, 'unknown'
    eigvals, eigvecs = np.linalg.eigh(cov)
    lam_major = float(eigvals[1])
    lam_minor = float(max(eigvals[0], 1e-12))
    elongation = (lam_major / lam_minor) ** 0.5

    if elongation < min_elongation:
        return aabb_center, np.eye(3), extent, 'unknown'

    principal = eigvecs[:, -1]
    yaw = float(np.arctan2(principal[1], principal[0]))
    cy, sy = np.cos(yaw), np.sin(yaw)
    R = np.array([[cy, -sy, 0.0],
                  [sy,  cy, 0.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)

    centroid = points.mean(axis=0)
    local = (R.T @ (points - centroid).T).T
    lmin = local.min(axis=0)
    lmax = local.max(axis=0)
    size = lmax - lmin
    center_world = centroid + R @ ((lmin + lmax) * 0.5)
    return center_world, R, size, 'fallen'


def _box_corners(center: np.ndarray, R: np.ndarray, size: np.ndarray) -> np.ndarray:
    h = np.asarray(size, dtype=np.float64) * 0.5
    s = np.array([
        [-1, -1, -1], [+1, -1, -1], [+1, +1, -1], [-1, +1, -1],
        [-1, -1, +1], [+1, -1, +1], [+1, +1, +1], [-1, +1, +1],
    ], dtype=np.float64) * h
    return (R @ s.T).T + np.asarray(center, dtype=np.float64)


def _quat_to_rot(x: float, y: float, z: float, w: float) -> np.ndarray:
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xx = x * x * s; yy = y * y * s; zz = z * z * s
    xy = x * y * s; xz = x * z * s; yz = y * z * s
    wx = w * x * s; wy = w * y * s; wz = w * z * s
    return np.array([
        [1 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1 - (xx + yy)],
    ], dtype=np.float64)


def _rot_to_quat(R: np.ndarray) -> tuple[float, float, float, float]:
    """Return quaternion (x, y, z, w) from a 3x3 rotation matrix."""
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    tr = m00 + m11 + m22
    if tr > 0:
        s = (tr + 1.0) ** 0.5 * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif (m00 > m11) and (m00 > m22):
        s = ((1.0 + m00 - m11 - m22) ** 0.5) * 2.0
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = ((1.0 + m11 - m00 - m22) ** 0.5) * 2.0
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = ((1.0 + m22 - m00 - m11) ** 0.5) * 2.0
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    return float(x), float(y), float(z), float(w)


def _make_pointcloud2(header: Header, xyz: np.ndarray, rgb: np.ndarray) -> PointCloud2:
    n = xyz.shape[0]
    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    point_step = 16
    buf = np.empty(n, dtype=np.dtype([
        ('x', np.float32), ('y', np.float32), ('z', np.float32), ('rgb', np.float32)
    ]))
    buf['x'] = xyz[:, 0]
    buf['y'] = xyz[:, 1]
    buf['z'] = xyz[:, 2]
    buf['rgb'] = rgb
    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = n
    msg.fields = fields
    msg.is_bigendian = False
    msg.point_step = point_step
    msg.row_step = point_step * n
    msg.is_dense = True
    msg.data = buf.tobytes()
    return msg


def main(args: Iterable[str] | None = None) -> None:
    rclpy.init(args=args)
    node = PointCloudNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
