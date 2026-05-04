"""Capture chessboard images from RealSense /color topic for intrinsic calibration.

Keys:
  s  save current frame as PNG into --output dir
  q  quit
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class CaptureNode(Node):
    def __init__(self, output_dir: Path, board_size: tuple[int, int], topic: str):
        super().__init__('capture_chessboard')
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.board_size = board_size
        self.bridge = CvBridge()
        self.frame: np.ndarray | None = None
        self.create_subscription(Image, topic, self._on_image, 10)
        self.get_logger().info(f'Subscribing to {topic}, saving to {self.output_dir}')

    def _on_image(self, msg: Image) -> None:
        self.frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def next_index(self) -> int:
        existing = sorted(self.output_dir.glob('chess_*.png'))
        if not existing:
            return 0
        last = existing[-1].stem.split('_')[-1]
        try:
            return int(last) + 1
        except ValueError:
            return len(existing)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data'),
                        help='Directory to save PNGs (default: ./data)')
    parser.add_argument('--topic', type=str,
                        default='/camera/camera/color/image_raw',
                        help='RealSense color topic')
    parser.add_argument('--board', type=str, default='10x7',
                        help='Inner corners as CxR (default: 10x7)')
    args, ros_args = parser.parse_known_args(argv)

    cols, rows = (int(x) for x in args.board.lower().split('x'))

    rclpy.init(args=ros_args)
    node = CaptureNode(args.output, (cols, rows), args.topic)

    win = 'capture_chessboard (s=save, q=quit)'
    cv2.namedWindow(win)
    saved = 0
    # Rolling detection history → flicker diagnosis + stability gate
    history: list[bool] = []
    HISTORY_LEN = 30
    STABLE_REQUIRED = 5  # frames in a row before "STABLE"
    streak = 0
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            if node.frame is None:
                continue
            display = node.frame.copy()
            gray = cv2.cvtColor(node.frame, cv2.COLOR_BGR2GRAY)

            # Primary: SB detector — robust to lighting/blur, replaces
            # findChessboardCorners + cornerSubPix.
            found, corners = cv2.findChessboardCornersSB(
                gray, (cols, rows),
                flags=cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_EXHAUSTIVE)
            # Fallback to legacy detector if SB fails (boundary cases).
            if not found:
                f2, c2 = cv2.findChessboardCorners(
                    gray, (cols, rows),
                    flags=cv2.CALIB_CB_ADAPTIVE_THRESH
                    + cv2.CALIB_CB_NORMALIZE_IMAGE)
                if f2:
                    corners = cv2.cornerSubPix(
                        gray, c2, (11, 11), (-1, -1),
                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                         30, 0.001))
                    found = True

            cv2.drawChessboardCorners(display, (cols, rows), corners, found)

            history.append(bool(found))
            if len(history) > HISTORY_LEN:
                history.pop(0)
            streak = streak + 1 if found else 0
            rate = sum(history) / len(history) * 100.0
            stable = streak >= STABLE_REQUIRED

            colour = (0, 255, 0) if stable else (0, 200, 255) if found else (0, 0, 255)
            tag = 'STABLE' if stable else ('detected' if found else 'NOT detected')
            status = (f'{tag} | streak={streak} | rate={rate:5.1f}% '
                      f'(last {len(history)}f) | saved={saved}')
            cv2.putText(display, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, colour, 2)
            cv2.putText(display, "press 's' when STABLE", (10, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.imshow(win, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            if key == ord('s'):
                if not found:
                    node.get_logger().warn('Skip save: corners not detected')
                    continue
                if not stable:
                    node.get_logger().warn(
                        f'Skip save: not stable yet (streak={streak}/{STABLE_REQUIRED})')
                    continue
                idx = node.next_index()
                path = node.output_dir / f'chess_{idx:03d}.png'
                cv2.imwrite(str(path), node.frame)
                saved += 1
                node.get_logger().info(f'Saved {path} (total {saved})')
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main(sys.argv[1:])
