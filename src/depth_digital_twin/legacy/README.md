# Legacy — Eye-to-Hand Calibration

These files implemented hand-eye calibration (T_exo2base / T_hand2base via ArUco).
Superseded by the ArUco-origin world frame mode in world_origin_node.

| File | Description |
|---|---|
| `depth_digital_twin/aruco_calibrate.py` | Online capture node (robot + camera live) |
| `depth_digital_twin/aruco_handeye.py` | Offline solver (cv2.calibrateHandEye PARK) |
| `launch/aruco_calibrate.launch.py` | Launch: dsr_bringup2 + capture tool |
| `config/T_exo2base.npy` | Last eye-to-hand calibration result (mm) |
| `config/T_hand2base.npy` | Last eye-in-hand calibration result (mm) |
