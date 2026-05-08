"""robot_pose_bridge_node — keep the Doosan URDF in a sensible pose at all times.

Background: `robot_state_publisher` consumes `/joint_states` to compute the
forward kinematics that RViz uses to render the robot model. When
`dsr_bringup2` is up, it already publishes `/dsr01/joint_states` from the
controller; we just need to relay (or remap) those into `/joint_states` for
RViz.

When the robot stack is NOT running (e.g. the user only wants a static
visualization of the model alongside the digital twin), no source publishes
joint states — RViz then shows nothing because robot_state_publisher has no
FK input. This node bridges that gap:

  1. Subscribe to `/dsr01/joint_states`.
  2. If a real message arrives, republish on `/joint_states` (rate-limited
     mirror, no transformation).
  3. If no real message arrives within `idle_timeout_s`, publish a zero-joint
     default at `default_rate_hz` so the URDF renders in its home pose.
  4. As soon as a real message resumes, switch back to mirror mode.

This means: connect the robot → live pose; don't connect → still see the
model. Either way the digital_twin pipeline runs unchanged.
"""
from __future__ import annotations

from typing import Iterable

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


# Doosan 6-DOF joint names (m0609 / m1013 share the same convention).
_DEFAULT_JOINT_NAMES = (
    'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6')


class RobotPoseBridge(Node):
    def __init__(self) -> None:
        super().__init__('robot_pose_bridge')
        self.declare_parameter('input_topic', '/dsr01/joint_states')
        self.declare_parameter('output_topic', '/joint_states')
        self.declare_parameter('idle_timeout_s', 1.0)
        self.declare_parameter('default_rate_hz', 10.0)
        self.declare_parameter('joint_names', list(_DEFAULT_JOINT_NAMES))
        self.declare_parameter('default_positions', [0.0] * 6)

        self.idle_timeout_s = float(self.get_parameter('idle_timeout_s').value)
        self.rate_hz = float(self.get_parameter('default_rate_hz').value)
        self.joint_names = list(self.get_parameter('joint_names').value)
        self.default_positions = list(self.get_parameter('default_positions').value)
        if len(self.default_positions) != len(self.joint_names):
            self.default_positions = [0.0] * len(self.joint_names)

        self.pub = self.create_publisher(
            JointState, str(self.get_parameter('output_topic').value), 10)
        self.create_subscription(
            JointState, str(self.get_parameter('input_topic').value),
            self._on_input, 10)

        self._last_real_msg = None  # rclpy.time.Time
        self.create_timer(1.0 / max(self.rate_hz, 0.1), self._on_tick)

        self.get_logger().info(
            f'robot_pose_bridge: relay {self.get_parameter("input_topic").value} '
            f'→ {self.get_parameter("output_topic").value}; '
            f'fallback default after {self.idle_timeout_s:.1f}s.')

    def _on_input(self, msg: JointState) -> None:
        self._last_real_msg = self.get_clock().now()
        # Mirror untouched (header restamped so RViz prefers the latest).
        out = JointState()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = msg.header.frame_id
        out.name = list(msg.name)
        out.position = list(msg.position)
        out.velocity = list(msg.velocity)
        out.effort = list(msg.effort)
        self.pub.publish(out)

    def _on_tick(self) -> None:
        # Only publish a default if no real source has been seen recently.
        if self._last_real_msg is not None:
            elapsed = (self.get_clock().now() - self._last_real_msg).nanoseconds * 1e-9
            if elapsed < self.idle_timeout_s:
                return
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self.joint_names)
        msg.position = list(self.default_positions)
        self.pub.publish(msg)


def main(args: Iterable[str] | None = None) -> None:
    rclpy.init(args=args)
    node = RobotPoseBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
