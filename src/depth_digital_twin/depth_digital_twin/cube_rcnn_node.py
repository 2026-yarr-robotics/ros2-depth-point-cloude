"""cube_rcnn_node — RGB → Cube R-CNN → Cube3DDetectionArray.

Outputs every detection's 3D oriented bounding box (centre, orientation, size)
in the source camera optical frame. A live debug image with both 2D bbox and
projected 3D box edges is also published, so the camera feed is always visible
in RViz regardless of detection success.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point as MsgPoint, Quaternion, Vector3
from rclpy.node import Node
from sensor_msgs.msg import Image

from depth_digital_twin.cube_rcnn_predictor import (
    CubeRcnnPredictor, obb_corners, rotation_matrix_to_quaternion,
)
from depth_digital_twin.intrinsics import load_intrinsics
from depth_digital_twin_msgs.msg import Cube3DDetection, Cube3DDetectionArray


# Indices for the 12 edges of an OBB given the corner order in
# `cube_rcnn_predictor.obb_corners`.
_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face (z = -h)
    (4, 5), (5, 6), (6, 7), (7, 4),  # top face    (z = +h)
    (0, 4), (1, 5), (2, 6), (3, 7),  # vertical edges
)


class CubeRcnnNode(Node):
    def __init__(self) -> None:
        super().__init__('cube_rcnn_node')

        self.declare_parameter('intrinsics_path', '')
        self.declare_parameter('config_file', '')
        self.declare_parameter('weights', '')
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('confidence', 0.25)
        self.declare_parameter('target_classes', [''])  # empty -> keep all
        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('detections_topic', '/digital_twin/cube_detections')
        self.declare_parameter('debug_topic', '/digital_twin/cube_debug')

        intr_path = Path(self.get_parameter('intrinsics_path').value)
        if not intr_path.is_file():
            raise FileNotFoundError(f'intrinsics_path not found: {intr_path}')
        self.intr = load_intrinsics(intr_path)
        self.K = self.intr.K
        self.get_logger().info(
            f'Loaded intrinsics {self.intr.width}x{self.intr.height}; '
            f'fx={self.intr.fx:.2f} cx={self.intr.cx:.2f}')

        import os
        cfg_file = self.get_parameter('config_file').value
        weights = self.get_parameter('weights').value
        device = self.get_parameter('device').value
        conf = float(self.get_parameter('confidence').value)
        targets = [s for s in self.get_parameter('target_classes').value if s]

        # ROS-2-Humble safety net: if the launch parameter override didn't
        # reach us (a recurring quirk with -p across multiple --ros-args
        # blocks), fall back to the env vars exported by
        # dependence/activate.bash. Same env vars are inherited by every
        # child process the launcher spawns, so this is always available.
        if not cfg_file:
            cfg_file = os.environ.get('CUBERCNN_CONFIG', '')
            if cfg_file:
                self.get_logger().warn(
                    f'config_file param empty; using $CUBERCNN_CONFIG fallback')
        if not weights:
            weights = os.environ.get('CUBERCNN_WEIGHTS', '')
            if weights:
                self.get_logger().warn(
                    f'weights param empty; using $CUBERCNN_WEIGHTS fallback')

        import os
        problems: list[str] = []
        if not cfg_file:
            problems.append(
                "  config_file is EMPTY. Pass `config_file:=...` to ros2 "
                "launch, or `source dependence/activate.bash` to export "
                "$CUBERCNN_CONFIG (env value: "
                f"{os.environ.get('CUBERCNN_CONFIG', '<unset>')!r})")
        elif not Path(cfg_file).is_file():
            problems.append(f"  config_file does not exist on disk: {cfg_file}")
        if not weights:
            problems.append(
                "  weights is EMPTY. Pass `weights:=...` to ros2 launch, "
                "or `source dependence/activate.bash` to export "
                "$CUBERCNN_WEIGHTS (env value: "
                f"{os.environ.get('CUBERCNN_WEIGHTS', '<unset>')!r})")
        elif not Path(weights).is_file():
            problems.append(f"  weights does not exist on disk: {weights}")
        if problems:
            raise RuntimeError(
                "Cube R-CNN cannot start:\n" + "\n".join(problems))

        self.predictor = CubeRcnnPredictor(
            config_file=cfg_file, weights=weights, device=device,
            score_threshold=conf, categories=targets or None)

        self.bridge = CvBridge()
        self.pub = self.create_publisher(
            Cube3DDetectionArray,
            self.get_parameter('detections_topic').value, 10)
        self.debug_pub = self.create_publisher(
            Image, self.get_parameter('debug_topic').value, 1)
        self.create_subscription(
            Image, self.get_parameter('image_topic').value, self._on_image, 10)
        self.get_logger().info(
            f'Cube R-CNN ready (conf={conf}, targets={targets or "all"})')

    # ------------------------------------------------------------------
    def _on_image(self, msg: Image) -> None:
        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        try:
            dets = self.predictor.predict(bgr, self.K)
        except Exception as e:  # noqa: BLE001 — surface model errors to logs
            self.get_logger().error(f'Cube R-CNN inference failed: {e}',
                                    throttle_duration_sec=2.0)
            return

        out = Cube3DDetectionArray()
        out.header = msg.header
        debug = bgr.copy()

        for d in dets:
            qx, qy, qz, qw = rotation_matrix_to_quaternion(d.R_cam)
            det_msg = Cube3DDetection()
            det_msg.class_name = d.class_name
            det_msg.class_id = int(d.class_id)
            det_msg.score = float(d.score)
            det_msg.x_min, det_msg.y_min, det_msg.x_max, det_msg.y_max = (
                int(v) for v in d.bbox2d)
            det_msg.center = MsgPoint(
                x=float(d.center_cam[0]),
                y=float(d.center_cam[1]),
                z=float(d.center_cam[2]))
            det_msg.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
            det_msg.size = Vector3(
                x=float(d.size[0]), y=float(d.size[1]), z=float(d.size[2]))
            out.detections.append(det_msg)
            _draw_overlay(debug, d, self.K)

        # Always-visible status header.
        cv2.putText(debug, f'cube_rcnn objects={len(out.detections)}',
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if out.detections else (0, 200, 255), 2)

        self.pub.publish(out)
        dbg = self.bridge.cv2_to_imgmsg(debug, encoding='bgr8')
        dbg.header = msg.header
        self.debug_pub.publish(dbg)


def _draw_overlay(img: np.ndarray, d, K: np.ndarray) -> None:
    """Annotate the 2D bbox plus the projected 3D OBB edges."""
    cv2.rectangle(img, (d.bbox2d[0], d.bbox2d[1]),
                  (d.bbox2d[2], d.bbox2d[3]), (0, 255, 0), 2)
    cv2.putText(img, f'{d.class_name} {d.score:.2f}',
                (d.bbox2d[0], max(0, d.bbox2d[1] - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    # Project the 8 OBB corners with K (camera frame -> pixel).
    corners = obb_corners(d.center_cam, d.R_cam, d.size)  # (8, 3)
    in_front = corners[:, 2] > 0.05
    if not np.any(in_front):
        return
    pix = (K @ corners.T).T
    pix = pix[:, :2] / np.clip(pix[:, 2:3], 1e-6, None)
    pts = pix.astype(int)
    for a, b in _EDGES:
        if not (in_front[a] and in_front[b]):
            continue
        cv2.line(img, tuple(pts[a]), tuple(pts[b]), (0, 165, 255), 2)


def main(args: Iterable[str] | None = None) -> None:
    rclpy.init(args=args)
    node = CubeRcnnNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
