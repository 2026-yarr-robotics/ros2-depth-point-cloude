# One camera's 2D-derived geometric constraint on one upright cup.
#
# Produced by point_cloud_node from the YOLO mask SILHOUETTE + camera
# calibration + world<-camera TF — NOT from per-pixel depth. The producer
# aligns the known truncated-cone model (2 DOF: axis x,y at an assumed base
# elevation) to the observed mask contour via a chamfer fit. Depth only
# contributes the rough z used to pick the level (median/percentile, so it
# tolerates cm-scale noise).
#
# cup_fusion_node consumes these: it snaps z to the level lattice, corrects
# (x,y) along `ray_dir` for the snapped z, and fuses cameras by inverse-
# covariance weighting in the world XY plane.

string camera          # "exo" | "hand"
int32 instance_id      # ByteTrack id (per-camera; NOT comparable across cameras)
string class_name      # e.g. "upright-cup"
float32 score          # YOLO confidence [0, 1]
string color           # producer's mask-pixel HSV color vote ("" = unknown).
                       # Carries the cups_on_table color contract without
                       # needing the point cloud's per-point RGB.

# Silhouette-fit measurement: cup axis position at the ASSUMED base elevation
# z_base0 (world frame). If the consumer decides the base is actually at z',
# correct with: xy' = (x0, y0) + (z' - z_base0) * ray_dir.xy / ray_dir.z
float64 x0
float64 y0
float64 z_base0

# Ray from the camera optical centre through the fitted axis at z_base0
# (world frame, unit vector). Encodes the view direction so the consumer can
# (a) re-slide the estimate along it when snapping z and (b) inflate the
# covariance along its XY projection (oblique views are uncertain range-wise).
geometry_msgs/Point ray_origin
geometry_msgs/Vector3 ray_dir

# Quality
float32 focal_px           # producer camera focal length (px) — converts
                           # sigma_px to metres at the cup's range
float32 sigma_px           # contour-pixel 1-sigma assumed by the producer
                           # (inflated for motion / truncation / low score)
float32 chamfer_rms_px     # model-contour -> observed-contour RMS distance
float32 mask_iou           # rendered silhouette vs YOLO mask IoU [0, 1]
uint32 contour_points      # observed contour samples used by the fit
float32 boundary_offset_px # recovered uniform mask-boundary bias b (px):
                           # >0 = YOLO over-segments, <0 = under-segments.
                           # Per-camera running mean of this IS the detector's
                           # boundary calibration.
float32 edge_coverage      # fraction of silhouette samples snapped to a
                           # strong image-gradient edge (0 = edge-snap off or
                           # failed; the chamfer/mask fit was used as-is)
float32 visible_fraction   # fraction of the model silhouette NOT hidden by
                           # other instances at the fitted pose (occlusion-
                           # aware fit; 1.0 = fully visible). The producer
                           # already inflates sigma_px by 1/max(vf, 0.25).

# Depth's demoted role: rough elevation for LEVEL CLASSIFICATION only.
float32 median_depth_m     # median depth inside the mask (0 = no valid depth)
float32 z_top_rough_m      # robust top-of-cup world z from the masked points
bool moving                # producer's joint-motion gate (wrist camera)
