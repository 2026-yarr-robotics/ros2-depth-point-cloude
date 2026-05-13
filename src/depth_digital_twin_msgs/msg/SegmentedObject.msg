# One detected & segmented object (single instance).

# Class label (e.g. "scissors").
string class_name

# Class id within the model's classes.
uint32 class_id

# Tracker-assigned instance id, stable across frames (Ultralytics ByteTrack).
# -1 when the tracker has not promoted this detection to a confirmed track yet.
int32 instance_id

# Detector confidence in [0, 1].
float32 score

# Axis-aligned bounding box in pixel coordinates (xyxy).
uint32 x_min
uint32 y_min
uint32 x_max
uint32 y_max

# Binary segmentation mask aligned with the source RGB frame.
# encoding: "mono8", values are 0 (background) or 255 (object).
sensor_msgs/Image mask
