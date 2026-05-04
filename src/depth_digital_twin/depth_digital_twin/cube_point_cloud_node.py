"""cube_point_cloud_node — Cube R-CNN 3D boxes → coloured PointCloud2 +
Marker.CUBE MarkerArray (in `world` frame).

Per detection:
  * Take depth pixels inside the 2D bbox.
  * Deproject to camera frame using the loaded intrinsics.
  * Keep only points that fall inside the 3D OBB (so background is removed).
  * Transform the surviving points to the `world` frame (TF from
    world_origin_node) and append RGB.

Outputs:
  * `/digital_twin/cube_points`  — sensor_msgs/PointCloud2 (RGB, world frame)
  * `/digital_twin/cube_boxes`   — visualization_msgs/MarkerArray (CUBE +
    text label per detection, world frame)
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

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

from depth_digital_twin.cube_rcnn_predictor import points_inside_obb
from depth_digital_twin.intrinsics import load_intrinsics
from depth_digital_twin_msgs.msg import Cube3DDetectionArray


def _palette(i: int) -> tuple[float, float, float]:
    base = [
        (0.95, 0.26, 0.21), (0.30, 0.69, 0.31), (0.13, 0.59, 0.95),
        (1.00, 0.76, 0.03), (0.61, 0.15, 0.69), (0.00, 0.74, 0.83),
    ]
    return base[i % len(base)]


def _pack_rgb(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    rgb_uint = ((r.astype(np.uint32) << 16)
                | (g.astype(np.uint32) << 8)
                | b.astype(np.uint32))
    return np.frombuffer(rgb_uint.astype(np.uint32).tobytes(), dtype=np.float32)


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


def _make_pointcloud2(header: Header, xyz: np.ndarray, rgb: np.ndarray) -> PointCloud2:
    n = int(xyz.shape[0])
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
    if n:
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


def _quat_mul(q1: tuple[float, float, float, float],
              q2: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Hamilton product (x, y, z, w)."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


class CubePointCloudNode(Node):
    def __init__(self) -> None:
        super().__init__('cube_point_cloud_node')

        self.declare_parameter('intrinsics_path', '')
        self.declare_parameter('rgb_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic',
                               '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('detections_topic', '/digital_twin/cube_detections')
        self.declare_parameter('points_topic', '/digital_twin/cube_points')
        self.declare_parameter('boxes_topic', '/digital_twin/cube_boxes')
        self.declare_parameter('camera_frame', 'camera_color_optical_frame')
        self.declare_parameter('world_frame', 'world')
        self.declare_parameter('depth_unit', 0.001)
        self.declare_parameter('z_min', 0.1)
        self.declare_parameter('z_max', 4.0)
        self.declare_parameter('obb_inflate', 0.02)  # m, slack when filtering points
        self.declare_parameter('approx_sync_slop', 0.05)
        self.declare_parameter('box_alpha', 0.25)

        path = Path(self.get_parameter('intrinsics_path').value)
        if not path.is_file():
            raise FileNotFoundError(f'intrinsics_path not found: {path}')
        self.intr = load_intrinsics(path)

        self.camera_frame: str = self.get_parameter('camera_frame').value
        self.world_frame: str = self.get_parameter('world_frame').value
        self.depth_unit: float = float(self.get_parameter('depth_unit').value)
        self.z_min: float = float(self.get_parameter('z_min').value)
        self.z_max: float = float(self.get_parameter('z_max').value)
        self.obb_inflate: float = float(self.get_parameter('obb_inflate').value)
        slop: float = float(self.get_parameter('approx_sync_slop').value)
        self.box_alpha: float = float(self.get_parameter('box_alpha').value)

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

        rgb_sub = message_filters.Subscriber(
            self, Image, self.get_parameter('rgb_topic').value)
        depth_sub = message_filters.Subscriber(
            self, Image, self.get_parameter('depth_topic').value)
        det_sub = message_filters.Subscriber(
            self, Cube3DDetectionArray, self.get_parameter('detections_topic').value)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub, det_sub], queue_size=10, slop=slop)
        self.sync.registerCallback(self._on_synced)
        self.get_logger().info('cube_point_cloud_node ready (waiting for synced frames)')

    # ------------------------------------------------------------------
    def _on_synced(self, rgb_msg: Image, depth_msg: Image,
                   det_msg: Cube3DDetectionArray) -> None:
        try:
            tf = self.tf_buffer.lookup_transform(
                self.world_frame, self.camera_frame, rclpy.time.Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            self.get_logger().warn(
                f'TF {self.world_frame}<-{self.camera_frame} not ready',
                throttle_duration_sec=2.0)
            return

        t_wc = np.array([tf.transform.translation.x,
                         tf.transform.translation.y,
                         tf.transform.translation.z], dtype=np.float64)
        q_wc = (tf.transform.rotation.x, tf.transform.rotation.y,
                tf.transform.rotation.z, tf.transform.rotation.w)
        R_wc = _quat_to_rot(*q_wc)

        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        h, w = depth.shape[:2]
        if rgb.shape[:2] != (h, w):
            self.get_logger().warn_once(
                f'RGB ({rgb.shape[:2]}) != depth ({h, w}); aligned_depth required')
            return

        z = depth.astype(np.float32) * self.depth_unit
        valid = (z > self.z_min) & (z < self.z_max)

        all_xyz: list[np.ndarray] = []
        all_rgb: list[np.ndarray] = []
        markers = MarkerArray()
        clear = Marker()
        clear.header.frame_id = self.world_frame
        clear.header.stamp = rgb_msg.header.stamp
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)

        for i, det in enumerate(det_msg.detections):
            x0 = max(0, int(det.x_min)); y0 = max(0, int(det.y_min))
            x1 = min(w, int(det.x_max)); y1 = min(h, int(det.y_max))
            if x1 <= x0 or y1 <= y0:
                continue

            sub_valid = valid[y0:y1, x0:x1]
            if not np.any(sub_valid):
                # Still publish the box even if no depth points landed.
                _append_cube_marker(markers, det, R_wc, t_wc, q_wc, i,
                                    rgb_msg.header.stamp, self.world_frame,
                                    self.box_alpha)
                continue

            ys, xs = np.where(sub_valid)
            ys_full = ys + y0
            xs_full = xs + x0
            zs = z[ys_full, xs_full]
            cam_x = (xs_full.astype(np.float32) - self.intr.cx) * zs / self.intr.fx
            cam_y = (ys_full.astype(np.float32) - self.intr.cy) * zs / self.intr.fy
            pts_cam = np.stack([cam_x, cam_y, zs], axis=1)

            # OBB filter: centre/R/size are in camera frame.
            centre = np.array([det.center.x, det.center.y, det.center.z],
                              dtype=np.float64)
            R_box = _quat_to_rot(det.orientation.x, det.orientation.y,
                                 det.orientation.z, det.orientation.w)
            size = np.array([det.size.x, det.size.y, det.size.z],
                            dtype=np.float64)
            inside = points_inside_obb(pts_cam, centre, R_box, size,
                                       pad=self.obb_inflate)
            if np.any(inside):
                pts_cam = pts_cam[inside]
                ys_full = ys_full[inside]
                xs_full = xs_full[inside]
                pts_world = (R_wc @ pts_cam.T).T + t_wc
                bgr = rgb[ys_full, xs_full]
                rgb_packed = _pack_rgb(bgr[:, 2], bgr[:, 1], bgr[:, 0])
                all_xyz.append(pts_world.astype(np.float32))
                all_rgb.append(rgb_packed)

            _append_cube_marker(markers, det, R_wc, t_wc, q_wc, i,
                                rgb_msg.header.stamp, self.world_frame,
                                self.box_alpha)

        if all_xyz:
            xyz = np.concatenate(all_xyz, axis=0)
            rgb_arr = np.concatenate(all_rgb, axis=0)
        else:
            xyz = np.zeros((0, 3), dtype=np.float32)
            rgb_arr = np.zeros((0,), dtype=np.float32)
        cloud = _make_pointcloud2(
            header=Header(stamp=rgb_msg.header.stamp, frame_id=self.world_frame),
            xyz=xyz, rgb=rgb_arr)
        self.points_pub.publish(cloud)
        self.boxes_pub.publish(markers)


def _append_cube_marker(markers: MarkerArray,
                        det,
                        R_wc: np.ndarray,
                        t_wc: np.ndarray,
                        q_wc: tuple[float, float, float, float],
                        idx: int,
                        stamp,
                        frame_id: str,
                        alpha: float) -> None:
    # Centre in world frame.
    c_cam = np.array([det.center.x, det.center.y, det.center.z], dtype=np.float64)
    c_world = R_wc @ c_cam + t_wc

    # Composed rotation: q_world_box = q_world_camera ∘ q_camera_box.
    q_box = (det.orientation.x, det.orientation.y,
             det.orientation.z, det.orientation.w)
    q_world_box = _quat_mul(q_wc, q_box)

    colour = _palette(idx)

    box = Marker()
    box.header.frame_id = frame_id
    box.header.stamp = stamp
    box.ns = 'cube_boxes'
    box.id = idx + 1
    box.type = Marker.CUBE
    box.action = Marker.ADD
    box.pose.position = MsgPoint(x=float(c_world[0]),
                                 y=float(c_world[1]),
                                 z=float(c_world[2]))
    box.pose.orientation = Quaternion(
        x=float(q_world_box[0]), y=float(q_world_box[1]),
        z=float(q_world_box[2]), w=float(q_world_box[3]))
    box.scale.x = float(det.size.x)
    box.scale.y = float(det.size.y)
    box.scale.z = float(det.size.z)
    box.color = ColorRGBA(r=colour[0], g=colour[1], b=colour[2], a=float(alpha))
    markers.markers.append(box)

    # Outline marker — opaque line list along OBB edges for clarity.
    outline = Marker()
    outline.header = box.header
    outline.ns = 'cube_outline'
    outline.id = idx + 1
    outline.type = Marker.LINE_LIST
    outline.action = Marker.ADD
    outline.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    outline.scale.x = 0.0015
    outline.color = ColorRGBA(r=colour[0], g=colour[1], b=colour[2], a=1.0)

    # Build the 8 corners in world frame.
    h = np.array([det.size.x, det.size.y, det.size.z], dtype=np.float64) * 0.5
    s = np.array([
        [-1, -1, -1], [+1, -1, -1], [+1, +1, -1], [-1, +1, -1],
        [-1, -1, +1], [+1, -1, +1], [+1, +1, +1], [-1, +1, +1],
    ], dtype=np.float64) * h
    R_box_world = _quat_to_rot(*q_world_box)
    corners_world = (R_box_world @ s.T).T + c_world
    edges = ((0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7))
    for a, b in edges:
        outline.points.append(MsgPoint(x=float(corners_world[a, 0]),
                                       y=float(corners_world[a, 1]),
                                       z=float(corners_world[a, 2])))
        outline.points.append(MsgPoint(x=float(corners_world[b, 0]),
                                       y=float(corners_world[b, 1]),
                                       z=float(corners_world[b, 2])))
    markers.markers.append(outline)

    label = Marker()
    label.header = box.header
    label.ns = 'cube_labels'
    label.id = idx + 1
    label.type = Marker.TEXT_VIEW_FACING
    label.action = Marker.ADD
    label.pose.position = MsgPoint(
        x=float(c_world[0]),
        y=float(c_world[1]),
        z=float(c_world[2] + det.size.z * 0.5 + 0.03))
    label.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    label.scale.z = 0.05
    label.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
    label.text = f'{det.class_name} {det.score:.2f}'
    markers.markers.append(label)


def main(args: Iterable[str] | None = None) -> None:
    rclpy.init(args=args)
    node = CubePointCloudNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
