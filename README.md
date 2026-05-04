# ros2-depth-point-cloude

RealSense D435i + YOLO segmentation으로 만드는 **단일 카메라 디지털 트윈**.
컬러 PointCloud와 객체별 3D Convex Hull을 RViz2에 실시간으로 표시한다.
(로봇 base가 없는 환경 가정 — 이미지 정중앙 depth를 world 원점으로 사용)

## 파이프라인

```
RealSense (RGB + aligned depth)
        │
        ├─► world_origin_node ─► /tf_static (camera→world, translation only)
        ├─► detection_node    ─► /digital_twin/detections (SegmentedObjectArray)
        └─► point_cloud_node  ─► /digital_twin/points (PointCloud2, RGB)
                              └► /digital_twin/hulls  (MarkerArray, ConvexHull)
```

## 패키지

| 이름 | 역할 |
|---|---|
| `depth_digital_twin_msgs` | 커스텀 메시지 (`SegmentedObject`, `SegmentedObjectArray`, `Cube3DDetection`, `Cube3DDetectionArray`) — CMake |
| `depth_digital_twin` | 캘리브레이션 + 모든 ROS 노드 + 런치 — Python |

### 노드 목록

| 노드 | 역할 | 사용 파이프라인 |
|---|---|---|
| `world_origin_node` | 중앙 패치 평면 적합 → static TF `camera→world` (Z-up) | 공통 |
| `detection_node` | YOLO seg → `SegmentedObjectArray` (mask + bbox) | YOLO |
| `point_cloud_node` | mask × depth → PointCloud2 + ConvexHull MarkerArray | YOLO |
| `cube_rcnn_node` | RGB → Cube R-CNN → `Cube3DDetectionArray` (6D OBB) + debug image | Cube R-CNN |
| `cube_point_cloud_node` | OBB 내부 depth 필터 → PointCloud2 + CUBE/LINE_LIST MarkerArray | Cube R-CNN |
| `capture_chessboard` (CLI) | 체커보드 PNG 캡처 (S/Q 키) | 캘리브레이션 |
| `calibrate` (CLI) | `cv2.calibrateCamera` → `intrinsics.yaml` | 캘리브레이션 |

## 의존성

- ROS2 (Humble 이상 권장) + `realsense2_camera`, `cv_bridge`, `tf2_ros`, `message_filters`
- Python: `numpy`, `opencv-python`, `pyyaml`, `scipy` (3D ConvexHull), `ultralytics` (YOLO seg)

```bash
pip install ultralytics scipy
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
- `MarkerArray` `/digital_twin/hulls` (객체별 LINE_LIST + 텍스트 라벨)
- `Image` `/digital_twin/detection_debug` (마스크 + 박스 오버레이)

## 토픽 / 프레임

| 토픽 | 타입 | 설명 |
|---|---|---|
| `/digital_twin/detections` | `depth_digital_twin_msgs/SegmentedObjectArray` | 객체별 클래스/스코어/박스/이진 마스크 |
| `/digital_twin/points` | `sensor_msgs/PointCloud2` | RGB가 들어간 전체 장면 PC (world frame) |
| `/digital_twin/hulls` | `visualization_msgs/MarkerArray` | 객체별 3D ConvexHull edges + 라벨 |
| `/digital_twin/detection_debug` | `sensor_msgs/Image` | 검출 시각화 |

| Frame | 부모 | 비고 |
|---|---|---|
| `world` | `camera_color_optical_frame` | 이미지 중심 depth로 한 번 latch (translation only) |

## 파라미터 튜닝

`src/depth_digital_twin/config/params.yaml`

- `detection_node.target_classes`: 검출 대상 클래스 이름 리스트 (`["scissors"]`)
- `detection_node.confidence`: YOLO confidence threshold
- `detection_node.model`: 기본 `yolov26n-seg.pt` (Ultralytics가 자동 다운로드)
- `point_cloud_node.downsample`: 포인트 다운샘플 stride (값이 클수록 가벼움)
- `point_cloud_node.z_min/z_max`: depth 유효 범위(m)
- `world_origin_node.window_radius`: 중심 픽셀 주위 median 반경
- `world_origin_node.samples_required`: TF 확정에 필요한 프레임 수

## 부록 — Cube R-CNN 파이프라인 (RGB 단독 6D 박스)

YOLO 기반(2D mask + ConvexHull)과 **독립**적으로 동작하는 두 번째 파이프라인.
Meta의 **Cube R-CNN (Omni3D, CVPR 2023)** 모델로 RGB 한 장에서 객체의
**6D pose + 크기(width, height, length)** 를 직접 추정합니다.
Depth는 객체 점군을 OBB로 잘라내는 데에만 보조적으로 사용됩니다.

### 새로 추가된 메시지 / 노드 / 런치 / RViz 설정

| 카테고리 | 항목 |
|---|---|
| 메시지 | `depth_digital_twin_msgs/Cube3DDetection`, `Cube3DDetectionArray` |
| 노드 | `cube_rcnn_node`, `cube_point_cloud_node` |
| 모듈 | `depth_digital_twin/cube_rcnn_predictor.py` (Omni3D 추론 래퍼 + OBB 유틸) |
| Launch | `launch/cube_rcnn.launch.py` |
| RViz | `rviz/cube_rcnn.rviz` |
| 파라미터 | `config/params.yaml` 의 `cube_rcnn_node`, `cube_point_cloud_node` 섹션 |

### 설치 — `dependence/setup.bash` 스크립트 (1회 실행)

ROS 2 Humble의 시스템 Python 3.10에 맞춰, 모든 ML 의존성을
**`dependence/` 폴더 안의 venv로 격리** 설치하는 스크립트가 포함되어 있습니다.
시스템 Python을 그대로 쓰므로 conda 없이 ROS와 같은 인터프리터에서
`rclpy` ↔ `cubercnn`이 한 프로세스에 함께 동작합니다.

```bash
cd ~/Projects/ros2-depth-point-cloude
./dependence/setup.bash
```

스크립트가 자동으로 처리하는 것:

| 단계 | 내용 |
|---|---|
| 0 | venv 생성 (`--system-site-packages` → `rclpy`/`cv_bridge` 그대로 사용) |
| 1 | `numpy<2` (Humble `cv_bridge` ABI 호환) |
| 2 | PyTorch 2.1.2 + Torchvision 0.16.2 (CUDA 자동 감지: cu121/cu118/cpu) |
| 3 | iopath / fvcore |
| 4 | **detectron2** 소스 빌드 (5–15 min) |
| 5 | **pytorch3d** 소스 빌드 (10–20 min) |
| 6 | OpenCV, scipy, seaborn, cython, cocoapi, PyYAML, ultralytics |
| 7 | **Omni3D (cubercnn)** clone + editable install (`--no-deps`) |
| 8 | 사전학습 가중치 `cubercnn_DLA34_FPN.pth` 다운로드 |
| 9 | 모든 모듈 import smoke test |

스크립트는 **멱등(idempotent)** 입니다 — 재실행 시 이미 설치된 단계는 probe
import로 자동 skip 됩니다.

선택적 환경변수:

```bash
PYTHON_BIN=python3.10 \
CUDA_TAG=cu118 \      # cu121 | cu118 | cu117 | cpu (미설정 시 자동)
MAX_JOBS=2 \          # pytorch3d 빌드 시 RAM 부족 대비
./dependence/setup.bash
```

설치 후 디렉토리:

```
dependence/
├── setup.bash       # 설치 스크립트 (방금 실행한 것)
├── activate.bash    # 매 터미널에서 source
├── clean.bash       # 전체 초기화
├── README.md
├── venv/            # python venv (생성됨)
├── src/{detectron2,pytorch3d,omni3d}/
├── models/cubercnn_DLA34_FPN.pth
└── logs/            # 단계별 빌드 로그
```

### 활성화 — 매 터미널에서

```bash
source dependence/activate.bash
```

이 한 줄이 다음 4가지를 순서대로 처리합니다:
1. `/opt/ros/humble/setup.bash` (미적용 시)
2. `dependence/venv/bin/activate`
3. `install/setup.bash` (워크스페이스 overlay)
4. `CUBERCNN_CONFIG` / `CUBERCNN_WEIGHTS` 환경변수 export

### ROS에서 실행

```bash
# 터미널 1: RealSense
ros2 launch realsense2_camera rs_align_depth_launch.py \
  depth_module.depth_profile:=640x480x30 \
  rgb_camera.color_profile:=640x480x30 \
  initial_reset:=true align_depth.enable:=true

# 터미널 2:
source dependence/activate.bash
ros2 launch depth_digital_twin cube_rcnn.launch.py \
  config_file:=$CUBERCNN_CONFIG \
  weights:=$CUBERCNN_WEIGHTS \
  device:=cuda
```

### 모델 자체 동작 검증 (선택, ROS 없이)

```bash
source dependence/activate.bash
cd dependence/src/omni3d
sh demo/download_demo_COCO_images.sh
python demo/demo.py \
  --config-file cubercnn://omni3d/cubercnn_DLA34_FPN.yaml \
  --input-folder "datasets/coco_examples" \
  --threshold 0.25 --display \
  MODEL.WEIGHTS cubercnn://omni3d/cubercnn_DLA34_FPN.pth \
  OUTPUT_DIR output/demo
```

`output/demo`에 3D 박스가 그려진 결과 이미지가 나오면 모델 자체는 정상 — 그 후
ROS 파이프라인 실행으로 넘어가세요.

### 의존성 초기화

```bash
./dependence/clean.bash            # 확인 후 venv/src/models/logs 삭제
./dependence/clean.bash --force    # 묻지 않고 삭제
```

### Omni3D 공식 권장 환경과의 차이

[Omni3D README](https://github.com/facebookresearch/omni3d#installation)는
`conda + Python 3.8 + PyTorch 1.8 + CUDA 10.1`을 권장하지만, 우리 ROS 2
Humble은 시스템 Python 3.10을 씁니다. conda env로 격리하면 **`rclpy`가
import 안 되므로** 같은 프로세스에서 ROS 노드를 띄울 수 없습니다.

따라서 본 프로젝트는 공식 README의 *"slight variations in versions are also
compatible"* 단서를 활용해 다음 조합으로 동작 검증되었습니다:

- Python 3.10 (Ubuntu 22.04 / Humble 기본)
- PyTorch 2.1.2 + Torchvision 0.16.2 + CUDA 12.1 휠
- detectron2 / pytorch3d / cubercnn 모두 소스에서 직접 빌드

이 조합이 깨지면 옵션 2(추론 마이크로서비스 분리)로 전환해야 합니다 — 본
README 하단의 트러블슈팅 표 참고.

### 산출 토픽

| 토픽 | 타입 | 설명 |
|---|---|---|
| `/digital_twin/cube_detections` | `depth_digital_twin_msgs/Cube3DDetectionArray` | 객체별 클래스/스코어/2D bbox/3D OBB(center+orientation+size) — 카메라 광학 프레임 |
| `/digital_twin/cube_points` | `sensor_msgs/PointCloud2` | OBB 내부에 들어온 픽셀만 deproject한 컬러 PC (world frame) |
| `/digital_twin/cube_boxes` | `visualization_msgs/MarkerArray` | 객체별 `Marker.CUBE`(반투명) + LINE_LIST 외곽선 + TEXT 라벨 (world frame) |
| `/digital_twin/cube_debug` | `sensor_msgs/Image` | 2D bbox + 투영된 3D OBB 모서리가 그려진 라이브 RGB |

### RViz2에 자동 활성화되는 디스플레이

`rviz/cube_rcnn.rviz`:
- `Grid` (XY plane @ world)
- `TF` (world ↔ camera optical)
- `PointCloud2` `/digital_twin/cube_points` (RGB8)
- `MarkerArray` `/digital_twin/cube_boxes`
- `Image` `/digital_twin/cube_debug`

### 파이프라인

```
RGB ──► cube_rcnn_node (Cube R-CNN, GPU)
         │
         └─► /digital_twin/cube_detections   (3D OBB in camera frame)
                          │
RGB + Depth + TF ──► cube_point_cloud_node
                          ├─► /digital_twin/cube_points  (OBB 내부 픽셀만, world frame)
                          └─► /digital_twin/cube_boxes   (CUBE + 외곽선 + 라벨, world frame)
```

YOLO 파이프라인(`digital_twin.launch.py`)과 **독립**입니다. 한 번에 하나만
실행하세요 (둘 다 띄우면 RealSense 토픽을 두 번 처리하느라 자원 낭비).

### 파라미터

`config/params.yaml`:

```yaml
cube_rcnn_node:
  ros__parameters:
    config_file: ""        # launch 인자로 주입
    weights: ""            # launch 인자로 주입
    device: cuda           # 'cuda' | 'cpu'
    confidence: 0.25
    target_classes: [""]   # 빈 리스트 = Omni3D 50+ 클래스 모두 통과
    image_topic: /camera/camera/color/image_raw
    detections_topic: /digital_twin/cube_detections
    debug_topic: /digital_twin/cube_debug

cube_point_cloud_node:
  ros__parameters:
    rgb_topic: /camera/camera/color/image_raw
    depth_topic: /camera/camera/aligned_depth_to_color/image_raw
    detections_topic: /digital_twin/cube_detections
    points_topic: /digital_twin/cube_points
    boxes_topic: /digital_twin/cube_boxes
    z_min: 0.1
    z_max: 4.0
    obb_inflate: 0.02      # m, OBB 내부 판정 padding
    approx_sync_slop: 0.05
    box_alpha: 0.25        # CUBE 반투명도
```

### 트러블슈팅

| 증상 | 원인 / 조치 |
|---|---|
| `RuntimeError: cubercnn (Omni3D) is not installed` | conda env 활성화 안 됨 또는 `pip install -e .` 누락 |
| `RuntimeError: Both 'config_file' and 'weights' must be set` | launch에 `config_file:= weights:=` 인자 누락 |
| `Cube R-CNN inference failed: CUDA out of memory` | `device:=cpu`로 시도, 또는 더 가벼운 모델 가중치 |
| RViz에 박스만 뜨고 PC가 비어 있음 | OBB와 depth가 정렬되지 않음 → 1) RealSense `align_depth.enable:=true` 확인, 2) `obb_inflate`를 0.05로 늘려보기 |
| 박스가 카메라 정면을 향해 너무 길게 그려짐 | Cube R-CNN의 깊이 추정 오차. depth 사용으로 보정하려면 OBB의 z-axis 길이를 PC 분포로 다시 적합 (다음 작업) |
| `/cube_debug`에 `objects=0` 가 계속 표시 | 클래스 못 잡음 → `confidence`를 0.15까지 낮춰보거나 `target_classes`를 빈 리스트로 두고 모든 클래스 모드로 확인 |

## 참고

- `sample/30장_Depth_카메라_캘리브레이션.pptx.pdf` — 체커보드 캘리브레이션 절차
- `sample/Calibration_Tutorial/handeye_calibration.py` — `calibrate_camera_from_chessboard()` 재사용 (handeye 부분은 본 과제에서 사용하지 않음)
- Brazil et al., "Omni3D: A Large Benchmark and Model for 3D Object Detection in the Wild", CVPR 2023 — Cube R-CNN 논문/코드 https://github.com/facebookresearch/omni3d
