"""point_cloud_node — fuse RGB + aligned depth + segmentation into a coloured
PointCloud2 (entire scene) plus per-object 3D ConvexHull line markers, all
expressed in the `world` frame.
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Iterable

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray

import message_filters
import tf2_ros

from depth_digital_twin.intrinsics import load_intrinsics
from depth_digital_twin_msgs.msg import SegmentedObjectArray


def _palette(i: int) -> tuple[float, float, float]:
    """Stable distinct colour per index in [0,1]."""
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
        self.declare_parameter('hulls_topic', '/digital_twin/hulls')
        self.declare_parameter('camera_frame', 'camera_color_optical_frame')
        self.declare_parameter('world_frame', 'world')
        self.declare_parameter('depth_unit', 0.001)
        self.declare_parameter('downsample', 2)  # take every Nth pixel
        self.declare_parameter('z_min', 0.1)
        self.declare_parameter('z_max', 4.0)
        self.declare_parameter('hull_line_width', 0.005)  # m
        self.declare_parameter('approx_sync_slop', 0.05)  # s
        self.declare_parameter('objects_only', True)  # only publish points inside detected masks

        path = Path(self.get_parameter('intrinsics_path').value)
        if not path.is_file():
            raise FileNotFoundError(f'intrinsics_path not found: {path}')
        self.intr = load_intrinsics(path)

        self.camera_frame: str = self.get_parameter('camera_frame').value
        self.world_frame: str = self.get_parameter('world_frame').value
        self.depth_unit: float = float(self.get_parameter('depth_unit').value)
        self.downsample: int = max(1, int(self.get_parameter('downsample').value))
        self.z_min: float = float(self.get_parameter('z_min').value)
        self.z_max: float = float(self.get_parameter('z_max').value)
        self.hull_w: float = float(self.get_parameter('hull_line_width').value)
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
        self.hulls_pub = self.create_publisher(
            MarkerArray, self.get_parameter('hulls_topic').value, latched)

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

    def _on_synced(self, rgb_msg: Image, depth_msg: Image,
                   det_msg: SegmentedObjectArray) -> None:
        # Resolve camera->world transform; if not yet broadcast, drop the frame.
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

        t_cw = np.array([tf.transform.translation.x,
                         tf.transform.translation.y,
                         tf.transform.translation.z], dtype=np.float64)
        # Rotation is identity by design (world axes-aligned with camera).
        # We support a non-identity rotation for safety using quaternion-to-matrix.
        q = tf.transform.rotation
        R_cw = _quat_to_rot(q.x, q.y, q.z, q.w)

        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        h, w = depth.shape[:2]
        if rgb.shape[:2] != (h, w):
            self.get_logger().warn_once(
                f'RGB ({rgb.shape[:2]}) != depth ({h, w}); requires aligned depth')
            return

        z = depth.astype(np.float32) * self.depth_unit  # metres
        valid = (z > self.z_min) & (z < self.z_max)

        # Union of all detected object masks (used for both gating the cloud
        # and for per-object hull computation below).
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
        ys = ys.flatten()
        xs = xs.flatten()
        sel = sample_mask[ys, xs]
        ys = ys[sel]
        xs = xs[sel]
        if ys.size == 0:
            # No detected pixels this frame — publish an empty cloud so RViz
            # clears the previous render instead of holding a stale point set.
            empty = _make_pointcloud2(
                header=Header(stamp=rgb_msg.header.stamp, frame_id=self.world_frame),
                xyz=np.zeros((0, 3), dtype=np.float32),
                rgb=np.zeros((0,), dtype=np.float32))
            self.points_pub.publish(empty)
            # Still clear hull markers.
            clear = MarkerArray()
            d = Marker()
            d.header.frame_id = self.world_frame
            d.header.stamp = rgb_msg.header.stamp
            d.action = Marker.DELETEALL
            clear.markers.append(d)
            self.hulls_pub.publish(clear)
            return

        zs = z[ys, xs]
        # Deproject to camera frame.
        cam_x = (xs.astype(np.float32) - self.intr.cx) * zs / self.intr.fx
        cam_y = (ys.astype(np.float32) - self.intr.cy) * zs / self.intr.fy
        cam_z = zs
        pts_cam = np.stack([cam_x, cam_y, cam_z], axis=1)  # (N,3)

        # camera -> world : p_w = R_wc @ p_c + t_wc.
        # tf above is target=world, source=camera, so it is T_w<-c directly:
        # p_w = R @ p_c + t, where R = R_cw, t = t_cw.
        pts_world = (R_cw @ pts_cam.T).T + t_cw

        # Pack RGB (BGR -> RGB for the standard 'rgb' field).
        bgr = rgb[ys, xs]
        rgb_packed = _pack_rgb(bgr[:, 2], bgr[:, 1], bgr[:, 0])

        cloud_msg = _make_pointcloud2(
            header=Header(stamp=rgb_msg.header.stamp, frame_id=self.world_frame),
            xyz=pts_world.astype(np.float32),
            rgb=rgb_packed)
        self.points_pub.publish(cloud_msg)

        # Per-object hulls — recompute deprojection for each mask at full resolution.
        markers = MarkerArray()
        delete_all = Marker()
        delete_all.header.frame_id = self.world_frame
        delete_all.header.stamp = rgb_msg.header.stamp
        delete_all.action = Marker.DELETEALL
        markers.markers.append(delete_all)

        for i, (obj, mb) in enumerate(per_object_masks):
            obj_valid = mb & valid
            if obj_valid.sum() < 32:
                continue
            oy, ox = np.where(obj_valid)
            oz = z[oy, ox]
            ocx = (ox.astype(np.float32) - self.intr.cx) * oz / self.intr.fx
            ocy = (oy.astype(np.float32) - self.intr.cy) * oz / self.intr.fy
            obj_cam = np.stack([ocx, ocy, oz], axis=1)
            obj_world = (R_cw @ obj_cam.T).T + t_cw

            edges = _convex_hull_edges(obj_world)
            if edges is None:
                continue

            colour = _palette(i)
            m = Marker()
            m.header.frame_id = self.world_frame
            m.header.stamp = rgb_msg.header.stamp
            m.ns = 'hulls'
            m.id = i + 1
            m.type = Marker.LINE_LIST
            m.action = Marker.ADD
            m.pose.orientation.w = 1.0
            m.scale.x = self.hull_w
            m.color = ColorRGBA(r=colour[0], g=colour[1], b=colour[2], a=1.0)
            for a, b in edges:
                m.points.append(Point(x=float(a[0]), y=float(a[1]), z=float(a[2])))
                m.points.append(Point(x=float(b[0]), y=float(b[1]), z=float(b[2])))
            markers.markers.append(m)

            label = Marker()
            label.header = m.header
            label.ns = 'hull_labels'
            label.id = i + 1
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            centroid = obj_world.mean(axis=0)
            label.pose.position = Point(x=float(centroid[0]), y=float(centroid[1]),
                                        z=float(centroid[2]))
            label.pose.orientation.w = 1.0
            label.scale.z = 0.04
            label.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            label.text = f'{obj.class_name} {obj.score:.2f}'
            markers.markers.append(label)

        self.hulls_pub.publish(markers)


def _quat_to_rot(x: float, y: float, z: float, w: float) -> np.ndarray:
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xx = x * x * s
    yy = y * y * s
    zz = z * z * s
    xy = x * y * s
    xz = x * z * s
    yz = y * z * s
    wx = w * x * s
    wy = w * y * s
    wz = w * z * s
    return np.array([
        [1 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1 - (xx + yy)],
    ], dtype=np.float64)


def _convex_hull_edges(points: np.ndarray):
    """Return list of (p_a, p_b) edge tuples for the 3D convex hull of `points`,
    or None if the hull cannot be computed."""
    try:
        from scipy.spatial import ConvexHull, QhullError  # type: ignore
    except ImportError:
        return None
    if points.shape[0] < 4:
        return None
    try:
        hull = ConvexHull(points)
    except (QhullError, ValueError):
        return None
    edges = set()
    for simplex in hull.simplices:
        for a, b in ((simplex[0], simplex[1]),
                     (simplex[1], simplex[2]),
                     (simplex[2], simplex[0])):
            key = (a, b) if a < b else (b, a)
            edges.add(key)
    return [(points[a], points[b]) for a, b in edges]


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
