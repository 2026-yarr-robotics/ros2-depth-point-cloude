"""cup_dedup_node — collapse duplicate YOLO detections of one physical cup.

Sits BETWEEN detection_node and point_cloud_node on the SegmentedObjectArray
stream. detection_node runs YOLO-seg with ByteTrack and the default NMS
(`iou=0.7`, class-aware), so a single cup at far exo range routinely survives
NMS as TWO heavily-overlapping same-class boxes, each handed its own ByteTrack
`instance_id`. Those two ids then escape point_cloud_node's 5 cm world-centroid
merge (which only fires on a NEW id's first frame) and persist as two tracks →
the cup is counted twice in /vision/cups_on_table.

This node groups detections whose pixel AABBs overlap (IoU ≥ `iou_thresh`) and
forwards only ONE survivor per group — the one with the LOWEST valid
instance_id, i.e. the oldest ByteTrack track ("the one that arrived first").
Everything else about the message is passed through untouched; in particular
`header.stamp` is preserved verbatim so point_cloud_node's
ApproximateTimeSynchronizer still matches the detections to RGB/Depth.

Why IoU is safe here: cups are spaced ~7.8 cm and at the exo working range
(~4.2 m) a cup box is only ~16 px wide, so an ADJACENT cup's box barely
overlaps (IoU ≈ 0) while a true DUPLICATE of one cup is near-coincident
(IoU high). A mid threshold cleanly separates the two.
"""
from __future__ import annotations

from typing import Iterable

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult

from depth_digital_twin_msgs.msg import SegmentedObject, SegmentedObjectArray


def _iou(a: SegmentedObject, b: SegmentedObject) -> float:
    """IoU of two axis-aligned pixel boxes (xyxy, inclusive coords)."""
    ax1, ay1, ax2, ay2 = a.x_min, a.y_min, a.x_max, a.y_max
    bx1, by1, bx2, by2 = b.x_min, b.y_min, b.x_max, b.y_max
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class CupDedupNode(Node):
    def __init__(self) -> None:
        super().__init__('cup_dedup_node')

        # Subscribe to the RAW detector output, republish the deduped array on
        # the topic point_cloud_node already listens to. Default wiring assumes
        # detection_node is remapped to publish on `_raw` (see launch file).
        self.declare_parameter('detections_in', '/digital_twin/detections_raw')
        self.declare_parameter('detections_out', '/digital_twin/detections')
        # IoU above which two boxes are treated as the same physical cup.
        self.declare_parameter('iou_thresh', 0.5)
        # When true, only group boxes that share a class label. The known cause
        # is same-class duplicates, so leaving this true avoids ever collapsing
        # a genuine fallen-cup + upright-cup pair that happens to overlap.
        self.declare_parameter('same_class_only', True)

        self.iou_thresh: float = float(self.get_parameter('iou_thresh').value)
        self.same_class_only: bool = bool(
            self.get_parameter('same_class_only').value)

        self.pub = self.create_publisher(
            SegmentedObjectArray,
            self.get_parameter('detections_out').value, 10)
        self.create_subscription(
            SegmentedObjectArray,
            self.get_parameter('detections_in').value, self._on_detections, 10)
        self.add_on_set_parameters_callback(self._on_set_params)
        self.get_logger().info(
            f"cup_dedup_node ready: "
            f"{self.get_parameter('detections_in').value} → "
            f"{self.get_parameter('detections_out').value} "
            f"(iou_thresh={self.iou_thresh}, "
            f"same_class_only={self.same_class_only})")

    def _on_set_params(self, params):
        for p in params:
            if p.name == 'iou_thresh':
                self.iou_thresh = float(p.value)
            elif p.name == 'same_class_only':
                self.same_class_only = bool(p.value)
        return SetParametersResult(successful=True)

    @staticmethod
    def _id_rank(obj: SegmentedObject) -> tuple:
        """Sort key for picking the survivor of a duplicate group.

        Prefer a confirmed track (instance_id >= 0) over an unconfirmed one
        (-1), then the LOWEST instance_id (oldest ByteTrack track = arrived
        first), then the HIGHEST score as a tie-break.
        """
        confirmed = obj.instance_id >= 0
        # (not confirmed) sorts True>False, so confirmed (False) comes first.
        return (not confirmed,
                obj.instance_id if confirmed else 0,
                -float(obj.score))

    def _on_detections(self, msg: SegmentedObjectArray) -> None:
        out = SegmentedObjectArray()
        out.header = msg.header  # preserve stamp + frame_id for downstream sync

        objs = list(msg.objects)
        n = len(objs)
        if n <= 1:
            out.objects = objs
            self.pub.publish(out)
            return

        # Greedy single-link grouping by box IoU. n is tiny (a handful of cups)
        # so the O(n^2) sweep is negligible.
        group_of = [-1] * n
        groups: list[list[int]] = []
        for i in range(n):
            if group_of[i] != -1:
                continue
            gid = len(groups)
            members = [i]
            group_of[i] = gid
            # Transitively absorb any later box overlapping any current member.
            k = 0
            while k < len(members):
                mi = members[k]
                for j in range(i + 1, n):
                    if group_of[j] != -1:
                        continue
                    if self.same_class_only and \
                            objs[j].class_id != objs[mi].class_id:
                        continue
                    if _iou(objs[mi], objs[j]) >= self.iou_thresh:
                        group_of[j] = gid
                        members.append(j)
                k += 1
            groups.append(members)

        kept: list[SegmentedObject] = []
        dropped = 0
        for members in groups:
            survivor = min((objs[m] for m in members), key=self._id_rank)
            kept.append(survivor)
            dropped += len(members) - 1
        out.objects = kept
        self.pub.publish(out)

        if dropped:
            self.get_logger().info(
                f'deduped {dropped} duplicate detection(s): '
                f'{n} → {len(kept)} objects',
                throttle_duration_sec=2.0)


def main(args: Iterable[str] | None = None) -> None:
    rclpy.init(args=args)
    node = CupDedupNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
