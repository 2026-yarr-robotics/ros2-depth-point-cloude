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
        # Per-axis MAD-based outlier filter on the world-frame point cluster
        # before fitting the box. Drops points whose deviation from the median
        # on any axis exceeds k * 1.4826 * MAD (k≈3 ⇒ 3σ for Gaussian noise).
        # Catches single-pixel depth spikes (specular/transparent/mixed pixel)
        # that otherwise inflate the AABB. 0 disables.
        self.declare_parameter('box_outlier_mad_k', 3.0)
        self.declare_parameter('approx_sync_slop', 0.05)
        self.declare_parameter('objects_only', True)

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
        self.outlier_mad_k: float = max(
            0.0, float(self.get_parameter('box_outlier_mad_k').value))
        self.objects_only: bool = bool(self.get_parameter('objects_only').value)
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

    # ------------------------------------------------------------------
    def _on_synced(self, rgb_msg: Image, depth_msg: Image,
                   det_msg: SegmentedObjectArray) -> None:
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

        union_mask = np.zeros((h, w), dtype=bool)
        per_object_masks: list[tuple[object, np.ndarray]] = []
        for obj in det_msg.objects:
            m = self.bridge.imgmsg_to_cv2(obj.mask, desired_encoding='mono8')
            if m.shape[:2] != (h, w):
                continue
            mb = m > 0
            union_mask |= mb
            per_object_masks.append((obj, mb))

        sample_mask = valid & union_mask if self.objects_only else valid

        ys, xs = np.mgrid[0:h:self.downsample, 0:w:self.downsample]
        ys = ys.flatten(); xs = xs.flatten()
        sel = sample_mask[ys, xs]
        ys = ys[sel]; xs = xs[sel]

        # Always emit a debug image (even if no detections) so RViz keeps a feed.
        debug_img = rgb.copy()

        if ys.size == 0:
            empty = _make_pointcloud2(
                header=Header(stamp=rgb_msg.header.stamp, frame_id=self.world_frame),
                xyz=np.zeros((0, 3), dtype=np.float32),
                rgb=np.zeros((0,), dtype=np.float32))
            self.points_pub.publish(empty)
            self._publish_clear_markers(rgb_msg.header.stamp)
            self._annotate_status(debug_img, 0)
            self._publish_debug(debug_img, rgb_msg.header)
            return

        zs = z[ys, xs]
        cam_x = (xs.astype(np.float32) - self.intr.cx) * zs / self.intr.fx
        cam_y = (ys.astype(np.float32) - self.intr.cy) * zs / self.intr.fy
        pts_cam = np.stack([cam_x, cam_y, zs], axis=1)
        pts_world = (R_wc @ pts_cam.T).T + t_wc

        bgr = rgb[ys, xs]
        rgb_packed = _pack_rgb(bgr[:, 2], bgr[:, 1], bgr[:, 0])

        cloud_msg = _make_pointcloud2(
            header=Header(stamp=rgb_msg.header.stamp, frame_id=self.world_frame),
            xyz=pts_world.astype(np.float32),
            rgb=rgb_packed)
        self.points_pub.publish(cloud_msg)

        # Per-object 3D boxes.
        markers = MarkerArray()
        clear = Marker()
        clear.header.frame_id = self.world_frame
        clear.header.stamp = rgb_msg.header.stamp
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)

        n_drawn = 0
        for i, (obj, mb) in enumerate(per_object_masks):
            mb_box = self._erode_mask(mb)
            # Fallback: if erosion ate the whole mask (small object) use raw.
            if mb_box.sum() < 32:
                mb_box = mb
            obj_valid = mb_box & valid
            if obj_valid.sum() < 32:
                continue
            oy, ox = np.where(obj_valid)
            oz = z[oy, ox]
            ocx = (ox.astype(np.float32) - self.intr.cx) * oz / self.intr.fx
            ocy = (oy.astype(np.float32) - self.intr.cy) * oz / self.intr.fy
            obj_cam = np.stack([ocx, ocy, oz], axis=1)
            obj_world = (R_wc @ obj_cam.T).T + t_wc
            obj_world = _filter_outliers(obj_world, self.outlier_mad_k)

            box = _compute_box_world(
                obj_world,
                standing_ratio=self.standing_ratio,
                min_elongation=self.min_elongation,
                force_aabb=self.force_aabb)
            if box is None:
                continue
            center, R_box, size, pose_label = box
            # Top-centre of the box (pivot (0, 0, +1) in box-local, scaled).
            top_world = center + R_box @ np.array(
                [0.0, 0.0, float(size[2]) * 0.5], dtype=np.float64)
            colour = _palette(i)
            label_text = (f'{obj.class_name} {obj.score:.2f} [{pose_label}] '
                          f'T({top_world[0]:.2f},{top_world[1]:.2f},'
                          f'{top_world[2]:.2f})')

            self._append_box_markers(
                markers, i + 1, center, R_box, size, top_world, colour,
                label_text, rgb_msg.header.stamp)
            self._draw_box_overlay(
                debug_img, center, R_box, size, top_world, colour, label_text,
                R_wc, t_wc)
            n_drawn += 1

        self.boxes_pub.publish(markers)
        self._annotate_status(debug_img, n_drawn)
        self._publish_debug(debug_img, rgb_msg.header)

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
        # Place above the sphere so they don't visually overlap.
        text.pose.position = MsgPoint(
            x=float(top_world[0]),
            y=float(top_world[1]),
            z=float(top_world[2] + 0.05))
        text.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        text.scale.z = 0.04
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
    def _erode_mask(self, mb: np.ndarray) -> np.ndarray:
        """Shrink the YOLO mask by `mask_erode_px` to drop edge pixels whose
        depth is unreliable (mixed foreground/background). No-op if disabled."""
        if self.mask_erode_px <= 0:
            return mb
        k = self.mask_erode_px
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        eroded = cv2.erode(mb.astype(np.uint8), kernel, iterations=1)
        return eroded > 0

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

    @staticmethod
    def _annotate_status(img: np.ndarray, n: int) -> None:
        colour = (0, 255, 0) if n else (0, 200, 255)
        cv2.putText(img, f'3d boxes={n}', (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)


# ----------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------
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
