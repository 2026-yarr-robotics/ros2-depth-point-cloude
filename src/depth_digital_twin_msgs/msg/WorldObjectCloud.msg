# One detected object's point cloud, already transformed into the world frame
# by a per-camera point_cloud_node running in `producer` role. Consumed by
# cup_fusion_node, which associates these across cameras (by geometry, NOT by
# instance_id) and fuses overlapping detections into a single cup estimate.

# Source camera label, e.g. "exo" | "hand".
string camera

# ByteTrack instance id (stable per-camera; NOT comparable across cameras).
int32 instance_id

# Class label (e.g. "upright-cup", "fallen-cup").
string class_name

# Detector confidence in [0, 1].
float32 score

# True when this came from a wrist-mounted (hand) camera whose joints were
# moving during the accumulation window — the cloud is low-confidence and the
# fusion node may down-weight or drop it. Always false for a static (exo) view.
bool moving

# World-frame centroid of `points` — precomputed so the fusion node can
# cluster/associate cheaply without unpacking every cloud.
geometry_msgs/Point centroid

# World-frame points for this object (xyz [+ packed rgb]). Window-accumulated
# and lightly filtered (MAD + point cap) by the producer; heavy fitting and
# Kalman filtering happen in the fusion node.
sensor_msgs/PointCloud2 points
