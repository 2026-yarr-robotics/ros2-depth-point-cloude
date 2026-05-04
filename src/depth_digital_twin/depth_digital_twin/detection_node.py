"""detection_node — YOLO segmentation → SegmentedObjectArray.

Loads an Ultralytics segmentation model (default: yolov26n-seg.pt — auto-downloads
on first run), filters detections to a target class set ("scissors" by default),
and publishes per-frame masks aligned to the source image.
"""
from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

from depth_digital_twin_msgs.msg import SegmentedObject, SegmentedObjectArray


class DetectionNode(Node):
    def __init__(self) -> None:
        super().__init__('detection_node')

        self.declare_parameter('model', 'yolov26n-seg.pt')
        self.declare_parameter('target_classes', ['scissors'])
        self.declare_parameter('confidence', 0.35)
        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('detections_topic', '/digital_twin/detections')
        self.declare_parameter('debug_topic', '/digital_twin/detection_debug')
        self.declare_parameter('device', '')  # '', 'cpu', '0'

        # Lazy import so the package can be inspected without ultralytics installed.
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                'ultralytics is not installed. Run: pip install ultralytics') from e

        model_name: str = self.get_parameter('model').value
        device: str = self.get_parameter('device').value
        self.get_logger().info(f'Loading YOLO model: {model_name}')
        try:
            self.model = YOLO(model_name)
        except FileNotFoundError as e:
            raise RuntimeError(
                f"Ultralytics could not download or locate '{model_name}'. "
                f"This usually means the package version does not know that "
                f"weight name. Try one of: yolov8n-seg.pt / yolo11n-seg.pt / "
                f"yolo12n-seg.pt, or upgrade ultralytics:\n"
                f"    pip install --user -U ultralytics") from e
        if device:
            self.model.to(device)

        self.targets: set[str] = {
            s.lower() for s in self.get_parameter('target_classes').value}
        self.conf: float = float(self.get_parameter('confidence').value)
        self.bridge = CvBridge()

        # Resolve target class IDs from model.names.
        self.class_id_to_name: dict[int, str] = {
            int(k): str(v).lower() for k, v in self.model.names.items()}
        self.target_ids: set[int] = {
            cid for cid, name in self.class_id_to_name.items() if name in self.targets}
        if not self.target_ids:
            self.get_logger().warn(
                f'No target class id matched. targets={self.targets}, '
                f'available example: {list(self.class_id_to_name.values())[:10]}')

        self.pub = self.create_publisher(
            SegmentedObjectArray, self.get_parameter('detections_topic').value, 10)
        self.debug_pub = self.create_publisher(
            Image, self.get_parameter('debug_topic').value, 1)
        self.create_subscription(
            Image, self.get_parameter('image_topic').value, self._on_image, 10)
        self.get_logger().info(
            f'Detection ready. targets={sorted(self.targets)} ids={sorted(self.target_ids)}')

    def _on_image(self, msg: Image) -> None:
        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w = bgr.shape[:2]
        results = self.model.predict(
            source=bgr, conf=self.conf, verbose=False, classes=list(self.target_ids) or None)
        if not results:
            return
        r = results[0]

        out = SegmentedObjectArray()
        out.header = msg.header  # share stamp + frame_id with the source image
        debug = bgr.copy()

        # If the model returned nothing, still publish the empty array AND a
        # live debug image (annotated with status) so the user always has a
        # video feed.
        if r.masks is None or r.boxes is None:
            cv2.putText(debug, 'no detections', (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            self.pub.publish(out)
            dbg = self.bridge.cv2_to_imgmsg(debug, encoding='bgr8')
            dbg.header = msg.header
            self.debug_pub.publish(dbg)
            return

        # masks.data: tensor (N, Hm, Wm) in [0,1] in the model's resolution.
        masks_t = r.masks.data
        boxes = r.boxes
        try:
            masks = masks_t.cpu().numpy()
        except AttributeError:  # already numpy
            masks = np.asarray(masks_t)

        for i in range(masks.shape[0]):
            cls_id = int(boxes.cls[i].item()) if hasattr(boxes.cls[i], 'item') else int(boxes.cls[i])
            score = float(boxes.conf[i].item()) if hasattr(boxes.conf[i], 'item') else float(boxes.conf[i])
            name = self.class_id_to_name.get(cls_id, str(cls_id))
            if self.target_ids and cls_id not in self.target_ids:
                continue
            xyxy = boxes.xyxy[i]
            xyxy = xyxy.cpu().numpy() if hasattr(xyxy, 'cpu') else np.asarray(xyxy)
            x1, y1, x2, y2 = (int(round(v)) for v in xyxy.tolist())

            # Resize mask to source frame and binarise.
            mask_src = cv2.resize(masks[i], (w, h), interpolation=cv2.INTER_NEAREST)
            mask_u8 = (mask_src > 0.5).astype(np.uint8) * 255

            obj = SegmentedObject()
            obj.class_name = name
            obj.class_id = cls_id
            obj.score = score
            obj.x_min = max(0, x1)
            obj.y_min = max(0, y1)
            obj.x_max = min(w - 1, x2)
            obj.y_max = min(h - 1, y2)
            obj.mask = self.bridge.cv2_to_imgmsg(mask_u8, encoding='mono8')
            obj.mask.header = msg.header
            out.objects.append(obj)

            cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(debug, f'{name} {score:.2f}', (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            tint = np.zeros_like(debug)
            tint[mask_u8 > 0] = (0, 0, 255)
            debug = cv2.addWeighted(debug, 1.0, tint, 0.4, 0.0)

        # Annotate frame summary so an empty result is still distinguishable
        # from "model not running".
        cv2.putText(debug, f'objects={len(out.objects)}', (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        self.pub.publish(out)
        dbg = self.bridge.cv2_to_imgmsg(debug, encoding='bgr8')
        dbg.header = msg.header
        self.debug_pub.publish(dbg)


def main(args: Iterable[str] | None = None) -> None:
    rclpy.init(args=args)
    node = DetectionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
