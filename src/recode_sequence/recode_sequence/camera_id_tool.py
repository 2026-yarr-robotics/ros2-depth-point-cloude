"""camera_id_tool — figure out which physical D435i is exo vs hand-eye.

Run cameras_only.launch.py first in IDENTIFY mode (no serials yet); it
starts both cameras as /cam_a and /cam_b. This tool then:

  1. shows both colour feeds side by side,
  2. reads each RealSense node's actual ``serial_no`` parameter and overlays
     it on the feed,
  3. lets you assign roles with the keyboard and writes cameras.yaml.

Keys (in the OpenCV window):
  a : LEFT (cam_a) is EXO,  cam_b is HAND
  b : RIGHT (cam_b) is EXO, cam_a is HAND
  s : save assignment → cameras.yaml
  q : quit

After saving, restart cameras_only.launch.py — it now binds /exo and /hand
by serial.
"""
from __future__ import annotations

import threading

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from rcl_interfaces.srv import GetParameters
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image

_DEFAULT_YAML = ('/home/eunwoosong/Projects/ros2-recode-sequence/'
                 'src/recode_sequence/config/cameras.yaml')


class _IdNode(Node):
    def __init__(self) -> None:
        super().__init__('camera_id_tool')
        self.declare_parameter('cam_a_ns', 'cam_a')
        self.declare_parameter('cam_b_ns', 'cam_b')
        self.declare_parameter('cameras_yaml', _DEFAULT_YAML)
        self.bridge = CvBridge()
        self.frames: dict[str, np.ndarray | None] = {'a': None, 'b': None}
        self.serials: dict[str, str] = {'a': '?', 'b': '?'}

        a_ns = str(self.get_parameter('cam_a_ns').value)
        b_ns = str(self.get_parameter('cam_b_ns').value)
        self.create_subscription(
            Image, f'/{a_ns}/{a_ns}/color/image_raw',
            lambda m: self._on_img('a', m), 5)
        self.create_subscription(
            Image, f'/{b_ns}/{b_ns}/color/image_raw',
            lambda m: self._on_img('b', m), 5)
        self._serial_client = {
            'a': self.create_client(
                GetParameters, f'/{a_ns}/{a_ns}/get_parameters'),
            'b': self.create_client(
                GetParameters, f'/{b_ns}/{b_ns}/get_parameters'),
        }
        self.create_timer(2.0, self._poll_serials)

    def _on_img(self, key: str, msg: Image) -> None:
        try:
            self.frames[key] = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='bgr8')
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f'{key}: cv_bridge {e}')

    def _poll_serials(self) -> None:
        for key, cli in self._serial_client.items():
            if self.serials[key] != '?' or not cli.service_is_ready():
                continue
            req = GetParameters.Request()
            req.names = ['serial_no']
            fut = cli.call_async(req)
            fut.add_done_callback(
                lambda f, k=key: self._on_serial(k, f))

    def _on_serial(self, key: str, fut) -> None:
        try:
            res = fut.result()
            if res and res.values:
                v = res.values[0]
                s = v.string_value or str(v.integer_value)
                self.serials[key] = s.strip("'") or '?'
        except Exception:  # noqa: BLE001
            pass

    def save(self, exo_key: str, cameras_yaml: str) -> str:
        hand_key = 'b' if exo_key == 'a' else 'a'
        data = {
            'exo_serial': self.serials[exo_key],
            'hand_serial': self.serials[hand_key],
        }
        with open(cameras_yaml, 'w') as f:
            f.write('# Written by camera_id_tool.\n')
            yaml.safe_dump(data, f, default_flow_style=False)
        return (f'saved exo={data["exo_serial"]} '
                f'hand={data["hand_serial"]} → {cameras_yaml}')


def _label(img: np.ndarray, lines: list[str]) -> np.ndarray:
    out = img.copy()
    y = 28
    for ln in lines:
        cv2.putText(out, ln, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, ln, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2, cv2.LINE_AA)
        y += 32
    return out


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = _IdNode()
    cameras_yaml = str(node.get_parameter('cameras_yaml').value)
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()

    win = 'camera_id_tool  [a]=left exo  [b]=right exo  [s]=save  [q]=quit'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    exo_key = 'a'
    status = 'waiting for both feeds…'
    blank = np.zeros((360, 640, 3), np.uint8)
    try:
        while rclpy.ok():
            fa = node.frames['a']
            fb = node.frames['b']
            la = _label(fa if fa is not None else blank, [
                f'cam_a  serial={node.serials["a"]}',
                'EXO' if exo_key == 'a' else 'HAND'])
            lb = _label(fb if fb is not None else blank, [
                f'cam_b  serial={node.serials["b"]}',
                'EXO' if exo_key == 'b' else 'HAND'])
            h = min(la.shape[0], lb.shape[0])
            canvas = np.hstack([
                cv2.resize(la, (int(la.shape[1] * h / la.shape[0]), h)),
                cv2.resize(lb, (int(lb.shape[1] * h / lb.shape[0]), h))])
            cv2.putText(canvas, status, (12, h - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),
                        2, cv2.LINE_AA)
            cv2.imshow(win, canvas)
            k = cv2.waitKey(30) & 0xFF
            if k == ord('q'):
                break
            if k == ord('a'):
                exo_key = 'a'
                status = 'cam_a = EXO, cam_b = HAND   (press s to save)'
            elif k == ord('b'):
                exo_key = 'b'
                status = 'cam_b = EXO, cam_a = HAND   (press s to save)'
            elif k == ord('s'):
                if '?' in (node.serials['a'], node.serials['b']):
                    status = 'serials not read yet — wait a moment…'
                else:
                    status = node.save(exo_key, cameras_yaml)
                    node.get_logger().info(status)
    finally:
        cv2.destroyAllWindows()
        ex.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
