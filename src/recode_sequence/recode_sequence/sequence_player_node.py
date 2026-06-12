"""sequence_player_node — replay a recorded sequence for RViz debugging.

Publishes, at the recorded rate, for the current step index:
  • exo / hand colour + aligned-depth Image (+ CameraInfo)
  • /joint_states  (drives robot_state_publisher → animated M0609 in RViz)
  • TF base→ee_recorded + an axes Marker for the recorded EE (TCP) pose
  • ~/state  std_msgs/String  "RUNNING 123/2000"  (for the control GUI)

Playback controls (services / topic), matching the spec:
  • ~/stop    std_srvs/Trigger  — pause
  • ~/resume  std_srvs/Trigger  — continue from current step
  • ~/replay  std_srvs/Trigger  — jump to step 0 and run
  • ~/goto_step  std_msgs/Int32 — jump to step N and PAUSE there
                                   (then press Resume to continue)
At the end it stops (does NOT wrap to start) unless loop:=true.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Int32, String
from std_srvs.srv import Trigger
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from recode_sequence.seq_io import SequenceReader


def _zyz_deg_to_quat(a: float, b: float, c: float) -> tuple:
    """Doosan ZYZ Euler (degrees) → quaternion (x, y, z, w)."""
    az, ay, cz = math.radians(a), math.radians(b), math.radians(c)

    def rz(t):
        return np.array([[math.cos(t), -math.sin(t), 0],
                         [math.sin(t), math.cos(t), 0],
                         [0, 0, 1]])

    def ry(t):
        return np.array([[math.cos(t), 0, math.sin(t)],
                         [0, 1, 0],
                         [-math.sin(t), 0, math.cos(t)]])

    R = rz(az) @ ry(ay) @ rz(cz)
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return x, y, z, w


class SequencePlayerNode(Node):
    def __init__(self) -> None:
        super().__init__('sequence_player_node')
        gp = self.declare_parameter
        gp('sequence_dir', '')
        gp('exo_color_topic', '/exo/exo/color/image_raw')
        gp('exo_depth_topic', '/exo/exo/aligned_depth_to_color/image_raw')
        gp('exo_info_topic', '/exo/exo/color/camera_info')
        gp('hand_color_topic', '/hand/hand/color/image_raw')
        gp('hand_depth_topic', '/hand/hand/aligned_depth_to_color/image_raw')
        gp('hand_info_topic', '/hand/hand/color/camera_info')
        gp('joint_states_topic', '/joint_states')
        gp('exo_frame', 'exo_color_optical_frame')
        gp('hand_frame', 'hand_color_optical_frame')
        gp('base_frame', 'base_link')
        gp('ee_frame', 'ee_recorded')
        gp('autostart', True)
        gp('loop', False)

        def P(n: str) -> Any:
            return self.get_parameter(n).value

        seq_dir = str(P('sequence_dir'))
        if not seq_dir:
            raise RuntimeError('sequence_dir parameter is required')
        self.reader = SequenceReader(seq_dir)
        self.bridge = CvBridge()
        self.base_frame = str(P('base_frame'))
        self.ee_frame = str(P('ee_frame'))
        self.exo_frame = str(P('exo_frame'))
        self.hand_frame = str(P('hand_frame'))
        self.loop = bool(P('loop'))

        self.idx = 0
        self.running = bool(P('autostart'))
        n = self.reader.n_steps
        self.get_logger().info(
            f'sequence_player: {seq_dir}  ({n} steps @ '
            f'{self.reader.record_rate_hz:.1f} Hz)')

        self.pub_exo_rgb = self.create_publisher(
            Image, str(P('exo_color_topic')), 5)
        self.pub_exo_depth = self.create_publisher(
            Image, str(P('exo_depth_topic')), 5)
        self.pub_exo_info = self.create_publisher(
            CameraInfo, str(P('exo_info_topic')), 5)
        self.pub_hand_rgb = self.create_publisher(
            Image, str(P('hand_color_topic')), 5)
        self.pub_hand_depth = self.create_publisher(
            Image, str(P('hand_depth_topic')), 5)
        self.pub_hand_info = self.create_publisher(
            CameraInfo, str(P('hand_info_topic')), 5)
        self.pub_joints = self.create_publisher(
            JointState, str(P('joint_states_topic')), 10)
        self.pub_marker = self.create_publisher(
            MarkerArray, '~/ee_marker', 5)
        self.pub_state = self.create_publisher(String, '~/state', 5)
        self.tf_bc = TransformBroadcaster(self)

        self.create_service(Trigger, '~/stop', self._srv_stop)
        self.create_service(Trigger, '~/resume', self._srv_resume)
        self.create_service(Trigger, '~/replay', self._srv_replay)
        self.create_subscription(Int32, '~/goto_step', self._on_goto, 10)

        rate = max(self.reader.record_rate_hz, 1.0)
        self.create_timer(1.0 / rate, self._tick)

    # -- control ------------------------------------------------------------
    def _srv_stop(self, _req, res):
        self.running = False
        res.success = True
        res.message = f'stopped at {self.idx}'
        return res

    def _srv_resume(self, _req, res):
        self.running = True
        res.success = True
        res.message = f'resumed at {self.idx}'
        return res

    def _srv_replay(self, _req, res):
        self.idx = 0
        self.running = True
        res.success = True
        res.message = 'replay from 0'
        return res

    def _on_goto(self, msg: Int32) -> None:
        self.idx = max(0, min(int(msg.data), self.reader.n_steps - 1))
        self.running = False  # spec: jump then stay paused until Resume
        self.get_logger().info(f'goto step {self.idx} (paused)')

    # -- playback -----------------------------------------------------------
    def _tick(self) -> None:
        n = self.reader.n_steps
        if n == 0:
            return
        self.idx = max(0, min(self.idx, n - 1))
        self._publish_step(self.idx)

        state = 'RUNNING' if self.running else 'STOPPED'
        m = String()
        m.data = f'{state} {self.idx}/{n - 1}'
        self.pub_state.publish(m)

        if not self.running:
            return
        if self.idx >= n - 1:
            if self.loop:
                self.idx = 0
            else:
                self.running = False  # spec: stop at end, no wrap
                self.get_logger().info('reached end — stopped.')
            return
        self.idx += 1

    def _publish_step(self, i: int) -> None:
        now = self.get_clock().now().to_msg()
        rec = self.reader.step_record(i)

        for view, frame, pub_rgb, pub_depth, pub_info in (
            ('exo', self.exo_frame, self.pub_exo_rgb,
             self.pub_exo_depth, self.pub_exo_info),
            ('hand', self.hand_frame, self.pub_hand_rgb,
             self.pub_hand_depth, self.pub_hand_info),
        ):
            rgb = self.reader.frame(i, view, 'rgb')
            if rgb is not None:
                msg = self.bridge.cv2_to_imgmsg(rgb, encoding='bgr8')
                msg.header.stamp = now
                msg.header.frame_id = frame
                pub_rgb.publish(msg)
            depth = self.reader.frame(i, view, 'depth')
            if depth is not None:
                dmsg = self.bridge.cv2_to_imgmsg(
                    depth.astype(np.uint16), encoding='16UC1')
                dmsg.header.stamp = now
                dmsg.header.frame_id = frame
                pub_depth.publish(dmsg)
            cam = self.reader.cameras.get(view)
            if cam and cam.K:
                ci = CameraInfo()
                ci.header.stamp = now
                ci.header.frame_id = frame
                ci.width = cam.width
                ci.height = cam.height
                ci.k = [float(v) for v in cam.K]
                ci.d = [float(v) for v in cam.dist]
                ci.distortion_model = 'plumb_bob'
                pub_info.publish(ci)

        # Robot joints → robot_state_publisher (RViz animation).
        if rec.get('joint_pos') and rec.get('joint_names'):
            js = JointState()
            js.header.stamp = now
            js.name = list(rec['joint_names'])
            js.position = [float(v) for v in rec['joint_pos']]
            self.pub_joints.publish(js)

        # Recorded EE (TCP) pose → TF + axes marker.
        ee = rec.get('ee_tcp')
        if ee and len(ee) >= 6:
            x, y, z = ee[0] / 1000.0, ee[1] / 1000.0, ee[2] / 1000.0
            qx, qy, qz, qw = _zyz_deg_to_quat(ee[3], ee[4], ee[5])
            tf = TransformStamped()
            tf.header.stamp = now
            tf.header.frame_id = self.base_frame
            tf.child_frame_id = self.ee_frame
            tf.transform.translation.x = x
            tf.transform.translation.y = y
            tf.transform.translation.z = z
            tf.transform.rotation.x = qx
            tf.transform.rotation.y = qy
            tf.transform.rotation.z = qz
            tf.transform.rotation.w = qw
            self.tf_bc.sendTransform(tf)
            self.pub_marker.publish(self._ee_axes(now, x, y, z,
                                                  qx, qy, qz, qw))

    def _ee_axes(self, stamp, x, y, z, qx, qy, qz, qw) -> MarkerArray:
        arr = MarkerArray()
        m = Marker()
        m.header.frame_id = self.base_frame
        m.header.stamp = stamp
        m.ns = 'ee_recorded'
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = z
        m.pose.orientation.x = qx
        m.pose.orientation.y = qy
        m.pose.orientation.z = qz
        m.pose.orientation.w = qw
        m.scale.x = m.scale.y = m.scale.z = 0.04
        m.color.r = 1.0
        m.color.g = 0.85
        m.color.b = 0.0
        m.color.a = 0.9
        arr.markers.append(m)
        return arr


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = SequencePlayerNode()
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
