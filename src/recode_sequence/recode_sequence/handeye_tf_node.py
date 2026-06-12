"""handeye_tf_node — static TF link_6 (flange) → hand camera optical frame.

Closes the hand camera's extrinsic chain for the fusion pipeline,
driven purely by the recorded JOINT trajectory (no EE / get_current_posx):

    exo_cam ──(world_origin_node, ArUco)──► world
    world ──(robot_state_publisher, m0609 URDF, world_fixed = identity)──► base_link
    base_link ──(URDF FK from recorded /joint_states)──► link_6  (flange)
    link_6 ──(THIS node: hand-eye T_gripper2camera.npy)──► hand_cam

The wrist-mounted hand camera never sees the ArUco marker, so its
world pose is derived from the recorded joint angles via URDF forward
kinematics (robot_state_publisher) plus the FLANGE-referenced hand-eye
calibration. Joints are reliably recorded from /dsr01/joint_states;
the DSR get_current_posx API is no longer used anywhere.

T_gripper2camera.npy is a 4×4 transform whose translation is in
MILLIMETRES; its magnitudes (≈[29,60,11] mm) show it is referenced
to the flange (link_6), NOT a gripper-tip TCP. The TF tree is in
metres, so translation is scaled by `units_scale` (default 0.001).
"""
from __future__ import annotations

import math
import os

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from tf2_ros import StaticTransformBroadcaster

_DEFAULT_NPY = ('/home/eunwoosong/Projects/ros2-recode-sequence/'
                'src/recode_sequence/config/T_gripper2camera.npy')


def _rot_to_quat(R: np.ndarray) -> tuple[float, float, float, float]:
    """3×3 rotation matrix → quaternion (x, y, z, w)."""
    t = float(np.trace(R))
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    return x / n, y / n, z / n, w / n


class HandeyeTfNode(Node):
    def __init__(self) -> None:
        super().__init__('handeye_tf_node')
        self.declare_parameter('handeye_npy', _DEFAULT_NPY)
        self.declare_parameter('parent_frame', 'link_6')
        self.declare_parameter('child_frame', 'hand_color_optical_frame')
        self.declare_parameter('units_scale', 0.001)  # npy mm → TF m

        npy = str(self.get_parameter('handeye_npy').value)
        if not os.path.isfile(npy):
            raise FileNotFoundError(f'hand-eye npy not found: {npy}')
        T = np.load(npy)
        if T.shape != (4, 4):
            raise ValueError(f'expected 4x4 hand-eye matrix, got {T.shape}')
        R = T[:3, :3]
        scale = float(self.get_parameter('units_scale').value)
        t = T[:3, 3] * scale

        det = float(np.linalg.det(R))
        if not (0.95 < det < 1.05):
            self.get_logger().warn(
                f'hand-eye rotation det={det:.4f} (≉1) — npy may be bad')
        qx, qy, qz, qw = _rot_to_quat(R)

        parent = str(self.get_parameter('parent_frame').value)
        child = str(self.get_parameter('child_frame').value)
        self._bc = StaticTransformBroadcaster(self)
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = parent
        tf.child_frame_id = child
        tf.transform.translation.x = float(t[0])
        tf.transform.translation.y = float(t[1])
        tf.transform.translation.z = float(t[2])
        tf.transform.rotation.x = qx
        tf.transform.rotation.y = qy
        tf.transform.rotation.z = qz
        tf.transform.rotation.w = qw
        self._bc.sendTransform(tf)
        self.get_logger().info(
            f'static TF {parent} → {child}  '
            f't(m)=({t[0]:.4f},{t[1]:.4f},{t[2]:.4f})  '
            f'q=({qx:.4f},{qy:.4f},{qz:.4f},{qw:.4f})  [from {npy}]')


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = HandeyeTfNode()
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
