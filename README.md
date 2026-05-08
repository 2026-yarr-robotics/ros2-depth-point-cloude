# ros2-depth-point-cloude

RealSense D435i + YOLO segmentation으로 만드는 **단일 카메라 디지털 트윈**.
컬러 PointCloud + 객체별 **3D Position Box / 컵 frustum**을 RViz2에 표시한다.

`world` frame은 두 가지 모드로 정의 가능:
- **floor-plane fit** (기본): 이미지 중심 patch에 평면 적합 → Z-up world.
- **calibrated** (eye-to-hand / eye-in-hand): 미리 계산해 둔 hand-eye 행렬을
  로드하여 `world := robot base`. Doosan 로봇 모델을 RViz에 함께 띄울 수 있음.

## 파이프라인

### 런타임 (실시간)

```
RealSense (RGB + aligned depth)
        │
        ├─► world_origin_node ─► /tf_static (camera→world)
        │                       모드 A: floor-plane fit (기본)
        │                       모드 B: T_exo2base.npy / T_hand2base.npy 로드
        │
        ├─► detection_node    ─► /digital_twin/detections        (SegmentedObjectArray)
        │                       /digital_twin/detection_debug   (Image: mask + 2D bbox)
        │
        └─► point_cloud_node  ─► /digital_twin/points            (PointCloud2 RGB; window 단위)
                              ├► /digital_twin/boxes            (MarkerArray: CUBE+outline+frustum+label)
                              ├► /digital_twin/box_debug         (Image: 투영된 3D 박스/frustum, 매 프레임)
                              └► /digital_twin/depth_debug       (Image: JET-colormap, patch box overlay)

(옵션) digital_twin_with_robot.launch.py 사용 시 추가:
        robot_state_publisher (m0609 / m1013 etc) ─► /robot_description, /tf
        robot_pose_bridge_node ─► /joint_states  (← /dsr01/joint_states or zero default)
        static TF: world ↔ base_0  (calibrated 모드에서 일치)
```

### 캘리브레이션 (한 번만)

```
[1] 카메라 intrinsic
    capture_chessboard ─► ./data/chess_*.png
    calibrate           ─► config/intrinsics.yaml

[2] Hand-eye (선택, robot 좌표계로 매핑할 때)
    aruco_calibrate (online, with bringup) ─► <output_dir>/aruco_NNN.png
                                            └► <output_dir>/calibrate_data.json
    aruco_handeye   (offline) ─► T_exo2base.npy  (--mode exo)
                              └► T_hand2base.npy (--mode hand)
```

## 패키지

| 이름 | 역할 |
|---|---|
| `depth_digital_twin_msgs` | 커스텀 메시지 (`SegmentedObject`, `SegmentedObjectArray`) — CMake |
| `depth_digital_twin` | 캘리브레이션 + ROS 노드 + 런치 — Python |

### 노드 목록

| 노드 | 역할 |
|---|---|
| `world_origin_node` | floor-plane fit *or* hand-eye 행렬 로드 → static TF `camera→world` |
| `detection_node` | YOLO seg → `SegmentedObjectArray` (mask + bbox) + segmentation debug image |
| `point_cloud_node` | mask × depth → window 누적 PointCloud2 + 3D box / cup frustum + 디버그 영상 (depth/box) |
| `robot_pose_bridge_node` | `/dsr01/joint_states` ↔ `/joint_states` 미러; idle 시 zero default 발행 |
| `capture_chessboard` (CLI) | 체커보드 PNG 캡처 (S/Q 키) |
| `calibrate` (CLI) | `cv2.calibrateCamera` → `intrinsics.yaml` |
| `aruco_calibrate` (CLI) | Online capture: `s` 누를 때마다 RGB + DSR posx를 디스크에 저장 |
| `aruco_handeye` (CLI) | Offline solve: capture 디렉토리 → `T_exo2base.npy` / `T_hand2base.npy` |

### Cup 추정 (Speed Stack 컵 가정)

대상이 ‘위·아래 지름이 다른 절두원뿔’ 형태의 컵임을 사전 정보로 활용한다.

1. world frame 점들에 **Kasa-style LS** 로 cup 축 (cx, cy) + base 높이 z_base 추정
   - r(z) = r_bot + (r_top − r_bot)·(z−z_base)/H 가정
   - z_base는 cluster의 robust **5th percentile** Z (테이블/선반 위에 있어도 자동)
   - 한쪽 면(은면)만 보여도 곡률로 축이 풀림 → 박스가 컵 전체를 감쌈
2. residual 통과(`cup_fit_residual_max` 기본 0.02 m) → **cup mode**:
   - box size = `(max(top_d, bot_d), max(top_d, bot_d), height)` (풀 지름 보장)
   - frustum wireframe (`LINE_STRIP` 두 원 + generatrix `LINE_LIST`) 함께 발행
3. residual 초과 → 기존 OBB/AABB fallback (`fallen` / `unknown` 라벨)

치수는 `params.yaml`의 `cup_top_diameter_m`, `cup_bottom_diameter_m`, `cup_height_m` 로 고정 (기본 5.4 / 7.8 / 9.5 cm).

### 0.5초 윈도우 누적 파이프라인

depth 노이즈를 한 프레임 단위로 잡으면 박스가 진동한다. 그래서:

- **매 프레임**: TF 조회 + 마스크별 world 점 deproject → track 버퍼에 누적. 디버그 이미지(`detection_debug`, `box_debug`, `depth_debug`)는 곧바로 발행 (15 Hz).
- **`window_period_s` 마다 (기본 0.5 s)**: 트랙별 누적 클러스터에 MAD 필터 적용 → cup fit (또는 OBB fallback) → EMA smoothing → `/digital_twin/points` 와 `/digital_twin/boxes`를 **단 한 번 갱신**.
- 결과: PointCloud / Marker는 2 Hz로 반응하지만 진동이 거의 없음. 디버그 이미지는 그대로 빠르게 흐름.

## 의존성

- ROS 2 (Humble 이상 권장) + `realsense2_camera`, `cv_bridge`, `tf2_ros`, `message_filters`, `robot_state_publisher`
- Python: `numpy`, `opencv-python` (≥4.7 권장 — `cv2.aruco.ArucoDetector`), `pyyaml`, `scipy`, `ultralytics` (YOLO seg)
- (옵션) Doosan 로봇 모델/상태 동기화: `~/ros2_ws` 의 `dsr_description2`, `dsr_msgs2` 등

```bash
pip install ultralytics scipy
sudo apt install ros-$ROS_DISTRO-realsense2-camera ros-$ROS_DISTRO-cv-bridge \
                 ros-$ROS_DISTRO-tf2-ros ros-$ROS_DISTRO-robot-state-publisher
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

### 2-A. 기본 모드 (floor-plane fit)

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
  calibration:= \                # 빈 문자열 ⇒ floor-plane fit (기본)
  rviz:=true
```

### 2-B. Calibrated 모드 + 로봇 모델 (eye-to-hand)

먼저 hand-eye 행렬을 만들어둔 뒤(아래 3단계 참조):

```bash
# Doosan 워크스페이스 함께 source — dsr_description2 share를 꺼내오기 위함
source ~/ros2_ws/install/setup.bash
source ~/Projects/ros2-depth-point-cloude/install/setup.bash

# (옵션) 실제 로봇/에뮬레이터 연결: 별도 터미널에서
ros2 launch dsr_bringup2 dsr_bringup2_rviz.launch.py    # 또는 본인 환경의 bringup

ros2 launch depth_digital_twin digital_twin_with_robot.launch.py \
  model:=m0609 \
  calibration:=/home/eunwoosong/Projects/ros2-depth-point-cloude/src/depth_digital_twin/config/T_exo2base.npy \
  with_pose_bridge:=true \
  rviz:=true
```

동작 정리:
- `calibration:=...`: `world_origin_node`가 floor-plane fit 대신 행렬을 로드. `world := robot base`.
- `with_pose_bridge:=true`: `/dsr01/joint_states`가 들어오면 그대로 `/joint_states`로 미러, 안 들어오면 1 Hz로 zero default를 발행 → URDF는 항상 home pose 이상으로 보임.
- `world ↔ base_0` identity static TF가 자동 발행되어 URDF가 디지털 트윈 TF tree에 합류.

RViz2 디스플레이 (`config/digital_twin.rviz`):
- `Grid` / `TF`
- `PointCloud2` `/digital_twin/points` (RGB8)
- `MarkerArray` `/digital_twin/boxes` — CUBE + outline + cup frustum + label
- `Image` segmentation debug (`/digital_twin/detection_debug`)
- `Image` 3d pos debug (`/digital_twin/box_debug`)
- `Image` depth debug (`/digital_twin/depth_debug`) — JET colormap + floor patch box
- `RobotModel` (`/robot_description`) — `digital_twin_with_robot.launch.py` 사용 시

## 3단계 — Hand-Eye 캘리브레이션 (ArUco)

EE에 ArUco 마커를 부착해 카메라와 로봇 사이 변환을 추정한다. `capture_chessboard` + `calibrate` 와 같은 **두 단계 워크플로우**:

1. **capture** (`aruco_calibrate`, 온라인): `s` 누를 때마다 RGB 프레임 + 로봇 posx를 디스크에 저장.
2. **solve** (`aruco_handeye`, 오프라인): 저장된 데이터로 ArUco 검출 + cv2.calibrateHandEye → `T_*2base.npy`.

| 모드 | 카메라 위치 | 마커 위치 | 결과 파일 | 런타임 사용 |
|---|---|---|---|---|
| `exo` | 외부 고정 | EE | `T_exo2base.npy` | 정적 TF (eye-to-hand) |
| `hand` | EE 부착 | 외부 고정 | `T_hand2base.npy` | 동적 TF (eye-in-hand, 라이브 robot pose 필요) |

> 모드 선택은 **solve 단계**에서 `--mode exo|hand` 로 지정한다. capture는 모드 무관 — 같은 데이터로 두 결과를 모두 뽑을 수도 있다.

### 3-1. Capture (이미지 + posx 저장)

`aruco_calibrate.launch.py`가 `dsr_bringup2_rviz`를 함께 띄운다. RealSense는 별도 터미널에서 실행 (1-1 명령 그대로).
```bash
# 터미널 1: RealSense (1-1 그대로)

# 터미널 2: bringup + capture
source ~/ros2_ws/install/setup.bash               # dsr_bringup2 / dsr_msgs2
source ~/Projects/ros2-depth-point-cloude/install/setup.bash

# 가상 로봇(에뮬레이터) — 기본
ros2 launch depth_digital_twin aruco_calibrate.launch.py

# 실제 로봇
ros2 launch depth_digital_twin aruco_calibrate.launch.py \
    bringup_mode:=real bringup_host:=192.168.1.100

# 실제 로봇(with_bringup:=false )
ros2 launch depth_digital_twin aruco_calibrate.launch.py \
    bringup_mode:=real bringup_host:=192.168.1.100  with_bringup:=false 

# 마커 / 출력 디렉토리 변경
ros2 launch depth_digital_twin aruco_calibrate.launch.py \
    marker_length:=0.04 marker_id:=0 dict:=4X4_50 \
    output_dir:=$PWD/data/aruco
```

> **인자 네이밍 주의**: capture launch에는 `aruco_mode`가 **없다** (capture는 모드 무관). 모드는 solve 단계의 `--mode` 로만 결정된다. `bringup_mode`(virtual/real)는 dsr_bringup2 통과용이라 그대로 유지.
>
> capture launch의 주요 인자:
> - `output_dir` — 캡처 저장 디렉토리 (기본 `<pkg_share>/config/aruco_capture`)
> - `marker_length` (m), `dict`, `marker_id` — ArUco 설정 (라이브 overlay용; solve도 동일값 사용 권장)
> - `with_bringup` — true면 `dsr_bringup2_rviz` 자동 실행 (기본 true)
> - `bringup_mode` — `virtual` (에뮬레이터, 기본) | `real`
> - `bringup_host`, `bringup_port`, `rt_host` — 실제 로봇 연결 정보
> - `startup_delay_s` — bringup 서비스 안정화 대기 (기본 6초; bringup 외부 실행 시 0으로)

bringup이 이미 떠 있는 경우엔 launch에서 toggle:

```bash
ros2 launch depth_digital_twin aruco_calibrate.launch.py \
    with_bringup:=false startup_delay_s:=0.0
```

CLI만 따로 쓰고 싶다면 (스크립트 / 디버그):

```bash
ros2 run depth_digital_twin aruco_calibrate \
    --intrinsics src/depth_digital_twin/config/intrinsics.yaml \
    --output_dir:=$PWD/record/exo \
    --marker-length 0.05 --dict 4X4_50 --marker-id 0
```

핫키 (cv2 윈도우):
- `s` — **이미지 + 현재 robot posx**를 `<output_dir>/aruco_NNN.png` + `calibrate_data.json` 에 저장 (ArUco STABLE 일 때만 허용)
- `u` — 마지막 저장 undo (PNG 삭제 + JSON 항목 제거)
- `q` / ESC — 종료. 저장은 `s` 누를 때마다 즉시 디스크에 반영되므로 별도 flush 불필요.

저장 결과 (`sample/Calibration_Tutorial/data_recording.py` 와 동일 layout):

```
<output_dir>/
  calibrate_data.json     # {"poses":[[x,y,z,rx,ry,rz],...], "file_name":["aruco_000.png",...]}
  aruco_000.png
  aruco_001.png
  ...
```

권장 절차: **15~25개** 샘플, EE를 **여러 회전 + 위치**로 분산. 평면 회전만 모이면 해가 불안해진다. 같은 디렉토리로 다시 launch 하면 기존 JSON을 읽어 이어서 누적한다.

### 3-2. Solve (오프라인 계산)

캡처 끝나고 cv2 윈도우 닫은 뒤 (또는 다른 터미널에서):

```bash
# eye-to-hand (camera 외부 고정)
ros2 run depth_digital_twin aruco_handeye \
    --data-dir $PWD/record/exo \
    --intrinsics src/depth_digital_twin/config/intrinsics.yaml \
    --output     src/depth_digital_twin/config/T_exo2base.npy \
    --mode       exo \
    --marker-length 0.05 --dict 4X4_50 --marker-id 0

# eye-in-hand (camera가 EE에 부착)
ros2 run depth_digital_twin aruco_handeye \
    --data-dir   ./data/aruco \
    --intrinsics src/depth_digital_twin/config/intrinsics.yaml \
    --output     src/depth_digital_twin/config/T_hand2base.npy \
    --mode       hand \
    --marker-length 0.05 --dict 4X4_50 --marker-id 0
```

solve는 디스크의 모든 이미지를 일괄 처리해 각 이미지에서 ArUco 검출 + `solvePnP` 를 수행한다. 실패 항목은 사유와 함께 skip 로그(`no marker detected` / `marker id=N missing` / `solvePnP failed`)로 출력되며, 최소 `--min-samples` (기본 10) 이상이면 `T_*2base.npy` 를 저장한다. 같은 capture 디렉토리에서 `--mode exo` / `--mode hand` 를 따로 두 번 실행해 두 결과를 모두 만들어둘 수도 있다.

### 3-3. 단위 / 명명 규약

- 입력 `posx`: Doosan 표준 — translation **mm**, ZYZ Euler **deg**
- 출력 `T_*2base.npy`: translation **mm** (`world_origin_node`가 자동 mm→m 변환)
- 파일 prefix가 모드를 결정 (필수):
  - `T_exo2base*.npy` → eye-to-hand 모드로 인식 (정적 TF, world := robot base)
  - `T_hand2base*.npy` → eye-in-hand 모드로 인식 (라이브 robot pose 필요; 현재 floor-fit fallback)
- `world_origin_node` 관련 파라미터:
  - `calibration_matrix_path`: 위 .npy 절대경로 (빈 문자열 ⇒ floor-fit)
  - `calibration_translation_unit`: `"mm"` (기본) 또는 `"m"`
  - `calibration_fallback_to_floor_fit`: 행렬 검증 실패 / hand-mode 미지원 시 floor-fit으로 폴백할지 여부

### 3-4. 라이브 robot pose 없이 사용

DSR_ROBOT2가 없는 환경 (오프라인 데이터셋, 모의 캘리) 에서는 capture에 `--no-robot` 플래그를 주면 `s` 누를 때마다 stdin으로 `posx` 6값을 직접 입력받는다. solve는 동일한 디렉토리/JSON에서 그대로 동작한다.

## 토픽 / 프레임

| 토픽 | 타입 | 설명 |
|---|---|---|
| `/digital_twin/detections` | `depth_digital_twin_msgs/SegmentedObjectArray` | 객체별 클래스/스코어/박스/이진 마스크 |
| `/digital_twin/detection_debug` | `sensor_msgs/Image` | YOLO segmentation 시각화 (mask + 2D bbox) |
| `/digital_twin/points` | `sensor_msgs/PointCloud2` | window 누적 후 발행되는 컬러 PC (world frame) |
| `/digital_twin/boxes` | `visualization_msgs/MarkerArray` | 객체별 box (CUBE + LINE_LIST) + cup frustum + 라벨 |
| `/digital_twin/box_debug` | `sensor_msgs/Image` | RGB 위 투영된 3D 박스/frustum + 라벨 (매 프레임) |
| `/digital_twin/depth_debug` | `sensor_msgs/Image` | JET colormap + mask edges + floor patch box (매 프레임) |
| `/joint_states` | `sensor_msgs/JointState` | (옵션) `robot_pose_bridge_node` 가 미러/디폴트 발행 |
| `/robot_description` | `std_msgs/String` | (옵션) `robot_state_publisher` URDF |

| Frame | 부모 | 비고 |
|---|---|---|
| `world` | `camera_color_optical_frame` | floor-fit 또는 calibrated 모드 결과 |
| `base_0` | `world` | `digital_twin_with_robot` 사용 시 identity static TF |

## 파라미터 튜닝

`src/depth_digital_twin/config/params.yaml`

### `/**:` 글로벌
- `window_radius`, `window_center_x_px`, `window_center_y_px` — floor patch 크기/위치 (`-1` ⇒ 이미지 중앙). `point_cloud_node`도 같은 값을 읽어 depth_debug에 patch box를 그림.
- `depth_unit` — RealSense aligned-depth 단위 (기본 `0.001`, mm→m).

### detection_node
- `target_classes`: 검출 대상 클래스 (기본 `["cup"]`)
- `confidence`: YOLO confidence threshold
- `model`: weight 경로 (Ultralytics 자동 다운로드 또는 절대경로)

### point_cloud_node
- `downsample`: 포인트 다운샘플 stride
- `z_min` / `z_max`: depth 유효 범위 (m)
- `box_line_width`: RViz LINE_LIST 두께 (m)
- `box_alpha`: CUBE 마커 투명도
- `box_standing_ratio` / `box_min_elongation` / `box_force_aabb`: OBB fallback 동작
- `mask_erode_px`: YOLO mask erosion (mixed-pixel 제거)
- `box_outlier_mad_k`: window 누적 클러스터 MAD 필터 k (3.0 ≈ 3σ)
- **Cup 모델**:
  - `cup_top_diameter_m` / `cup_bottom_diameter_m` / `cup_height_m`
  - `cup_polygon_segments`: frustum 원 세분
  - `cup_smoothing_alpha`: EMA on (cx, cy, z_base)
  - `cup_track_max_dist_m`, `cup_track_keepalive_frames`: 트래커 매칭/유지
  - `cup_fit_residual_max`: cup fit 잔차 임계 (초과 시 OBB fallback)
- **윈도우**: `window_period_s` (기본 0.5 s)

### world_origin_node
- floor-fit: `min_patch_points`, `max_plane_residual`, `samples_required`
- calibrated: `calibration_matrix_path`, `calibration_translation_unit`, `calibration_fallback_to_floor_fit`

### robot_pose_bridge_node (옵션 launch)
- `input_topic` (기본 `/dsr01/joint_states`), `output_topic` (기본 `/joint_states`)
- `idle_timeout_s`: 미수신 시 디폴트 발행으로 전환할 시간
- `default_rate_hz`, `joint_names`, `default_positions`

## 참고

- `sample/30장_Depth_카메라_캘리브레이션.pptx.pdf` — 체커보드 캘리브레이션 절차
- `sample/Calibration_Tutorial/data_recording.py` — `aruco_calibrate` capture가 따르는 JSON 레이아웃 (`{"poses":[...], "file_name":[...]}`)
- `sample/Calibration_Tutorial/handeye_calibration.py` (eye-in-hand) / `eye2hand_calibration.py` (eye-to-hand) — `aruco_handeye`가 따르는 단위/회전 규약 (mm, ZYZ deg) + cv2.calibrateHandEye 사용 예
- `sample/doosan_robot.pdf` — Doosan RViz 시각화 가이드 (`dsr_bringup2_rviz`)
- 본 패키지는 sample의 두 캘리 코드를 두 단계로 분리해 통합:
  - capture는 `aruco_calibrate.py` (이미지 + posx만 저장; sample/data_recording.py 패턴)
  - solve는 `aruco_handeye.py` (`--mode exo|hand` 분기로 두 캘리 모두 지원)
