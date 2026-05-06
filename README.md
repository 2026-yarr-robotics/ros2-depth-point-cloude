# ros2-depth-point-cloude

RealSense D435i + YOLO segmentation으로 만드는 **단일 카메라 디지털 트윈**.
컬러 PointCloud + 객체별 **3D Position Box**를 RViz2에 실시간으로 표시한다.
(로봇 base가 없는 환경 가정 — 이미지 정중앙 floor를 world 원점으로 사용)

## 파이프라인

```
RealSense (RGB + aligned depth)
        │
        ├─► world_origin_node ─► /tf_static (camera→world, floor-aligned)
        ├─► detection_node    ─► /digital_twin/detections        (SegmentedObjectArray)
        │                       /digital_twin/detection_debug   (Image: mask + 2D bbox)
        └─► point_cloud_node  ─► /digital_twin/points            (PointCloud2 RGB)
                              ├► /digital_twin/boxes            (MarkerArray: CUBE+outline+label)
                              └► /digital_twin/box_debug         (Image: projected 3D box)
```

## 패키지

| 이름 | 역할 |
|---|---|
| `depth_digital_twin_msgs` | 커스텀 메시지 (`SegmentedObject`, `SegmentedObjectArray`) — CMake |
| `depth_digital_twin` | 캘리브레이션 + ROS 노드 + 런치 — Python |

### 노드 목록

| 노드 | 역할 |
|---|---|
| `world_origin_node` | 중앙 패치 평면 적합 → static TF `camera→world` (Z-up) |
| `detection_node` | YOLO seg → `SegmentedObjectArray` (mask + bbox) + segmentation debug image |
| `point_cloud_node` | mask × depth → PointCloud2 + 3D Position Box MarkerArray + 3D pos debug image |
| `capture_chessboard` (CLI) | 체커보드 PNG 캡처 (S/Q 키) |
| `calibrate` (CLI) | `cv2.calibrateCamera` → `intrinsics.yaml` |

### 3D Position Box 추정

대상이 컵(원기둥) 위주라 standing 자세에서는 회전축이 모호하다. 다음 분기로 회전을 안전하게 결정한다.

1. world frame 점들에서 AABB 계산 → `z_ext`, `h_ext = max(x_ext, y_ext)`
2. `z_ext / h_ext > box_standing_ratio` (기본 0.8) → **STANDING** (AABB, yaw=0)
3. 그 외엔 XY 평면 PCA → `sqrt(λ1/λ2) > box_min_elongation` (기본 1.5)
   - 통과 → **FALLEN** (PCA 주축 yaw로 OBB)
   - 실패 → AABB로 fallback (`unknown` 라벨)

`box_force_aabb: true` 로 토글하면 OBB 로직 전체를 끄고 AABB만 사용한다 (디버그용).

## 의존성

- ROS 2 (Humble 이상 권장) + `realsense2_camera`, `cv_bridge`, `tf2_ros`, `message_filters`
- Python: `numpy`, `opencv-python`, `pyyaml`, `ultralytics` (YOLO seg)

```bash
pip install ultralytics
sudo apt install ros-$ROS_DISTRO-realsense2-camera ros-$ROS_DISTRO-cv-bridge \
                 ros-$ROS_DISTRO-tf2-ros
```

## 빌드

```bash
cd ~/Projects/ros2-depth-point-cloude
colcon build --symlink-install
source install/setup.bash
```

## 1단계 — 카메라 캘리브레이션 (intrinsic)

체커보드: **내부 코너 10×7, 한 칸 25 mm** (PDF 자료 링크 참조).
체커보드를 단단한 평판에 부착하고 조명을 균일하게.

### 1-1. RealSense 기동

```bash
ros2 launch realsense2_camera rs_align_depth_launch.py \
  depth_module.depth_profile:=640x480x30 \
  rgb_camera.color_profile:=640x480x30 \
  initial_reset:=true align_depth.enable:=true
```

### 1-2. 체커보드 이미지 수집

```bash
ros2 run depth_digital_twin capture_chessboard --output ./data --board 10x7
```

- 윈도우에 코너 검출 결과가 녹색으로 오버레이된다.
- `s` 키로 저장(코너 미검출 시 무시), `q` 또는 ESC로 종료.
- **20장 이상**, 보드를 다양한 위치/각도/거리로 — 정면/모서리/회전/기울임 골고루.

### 1-3. 캘리브레이션 계산

```bash
ros2 run depth_digital_twin calibrate \
  --images "./data/chess_*.png" \
  --board 10x7 --square 25 \
  --output src/depth_digital_twin/config/intrinsics.yaml \
  --undistort-preview ./data/undistorted_preview.png
```

성공 기준: **RMS reprojection error < 1.0 px** (이상적으로는 < 0.5 px).
값이 크면 이미지 추가 수집 → 재실행.

> 캘리브레이션 결과(`config/intrinsics.yaml`)를 패키지 share에 반영하기 위해
> 다시 `colcon build --symlink-install` 후 source.

## 2단계 — 디지털 트윈 실행

```bash
# 터미널 1: RealSense (위 1-1 그대로)
# 터미널 2:
source install/setup.bash
ros2 launch depth_digital_twin digital_twin.launch.py
```

런치 옵션:

```bash
ros2 launch depth_digital_twin digital_twin.launch.py \
  intrinsics:=$PWD/src/depth_digital_twin/config/intrinsics.yaml \
  rviz:=true
```

RViz2가 같이 뜨고 다음 디스플레이가 자동으로 활성화된다:
- `Grid` (XY plane @ world)
- `TF` (`world` ↔ `camera_color_optical_frame`)
- `PointCloud2` `/digital_twin/points` (RGB8)
- `MarkerArray` `/digital_twin/boxes` — 객체별 CUBE + 외곽선 + 라벨
- `Image` `segmentation debug` (`/digital_twin/detection_debug`) — mask + 2D bbox
- `Image` `3d pos debug` (`/digital_twin/box_debug`) — 투영된 3D 박스

## 토픽 / 프레임

| 토픽 | 타입 | 설명 |
|---|---|---|
| `/digital_twin/detections` | `depth_digital_twin_msgs/SegmentedObjectArray` | 객체별 클래스/스코어/박스/이진 마스크 |
| `/digital_twin/detection_debug` | `sensor_msgs/Image` | YOLO segmentation 시각화 (mask + 2D bbox) |
| `/digital_twin/points` | `sensor_msgs/PointCloud2` | RGB가 들어간 객체 PC (world frame) |
| `/digital_twin/boxes` | `visualization_msgs/MarkerArray` | 객체별 3D Position Box (CUBE + LINE_LIST + TEXT) |
| `/digital_twin/box_debug` | `sensor_msgs/Image` | RGB 위 투영된 3D 박스 + standing/fallen 라벨 |

| Frame | 부모 | 비고 |
|---|---|---|
| `world` | `camera_color_optical_frame` | 이미지 중심 floor로 한 번 latch (Z-up) |

## 파라미터 튜닝

`src/depth_digital_twin/config/params.yaml`

### detection_node
- `target_classes`: 검출 대상 클래스 (기본 `["bottle", "cup"]` — COCO 컵 호환)
- `confidence`: YOLO confidence threshold
- `model`: 기본 `yolov8n-seg.pt` (Ultralytics 자동 다운로드)

### point_cloud_node
- `downsample`: 포인트 다운샘플 stride (값이 클수록 가벼움)
- `z_min` / `z_max`: depth 유효 범위 (m)
- `box_line_width`: RViz LINE_LIST 두께 (m)
- `box_alpha`: CUBE 마커 투명도
- `box_standing_ratio`: `z_ext/h_ext` 가 이 값보다 크면 STANDING → AABB (기본 0.8)
- `box_min_elongation`: PCA `sqrt(λ1/λ2)` 가 이 값보다 작으면 회전 신뢰도 부족 → AABB fallback (기본 1.5)
- `box_force_aabb`: true면 OBB 비활성화 — AABB만 사용 (디버그용)

### world_origin_node
- `window_radius`: 중심 픽셀 주위 median 반경
- `samples_required`: TF 확정에 필요한 프레임 수

## 참고

- `sample/30장_Depth_카메라_캘리브레이션.pptx.pdf` — 체커보드 캘리브레이션 절차
- `sample/Calibration_Tutorial/handeye_calibration.py` — `calibrate_camera_from_chessboard()` 재사용 (handeye 부분은 본 과제에서 사용하지 않음)
