# Single 3D detection emitted by Cube R-CNN (Omni3D).
# All 3D fields are expressed in the source camera optical frame.

string class_name
uint32 class_id
float32 score

# 2D bounding box (xyxy, pixel coords) in the source RGB frame.
uint32 x_min
uint32 y_min
uint32 x_max
uint32 y_max

# 3D oriented bounding box.
geometry_msgs/Point      center        # box centre in camera optical frame
geometry_msgs/Quaternion orientation   # box orientation in camera optical frame
geometry_msgs/Vector3    size          # full extents along (x, y, z) of the box's local axes
