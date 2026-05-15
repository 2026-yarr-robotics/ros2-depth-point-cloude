# ROS 2 Depth Digital Twin

RealSense D435i + YOLO segmentation 기반 **단일 카메라 디지털 트윈**.
YOLO로 컵을 검출하고, 깊이 카메라로 3D 위치를 추정해 RViz2에 Doosan 로봇 모델과 함께 표시한다.
작업 공간에 배치된 ArUco 마커를 인식해 **world 프레임 = 로봇 base 프레임**으로 자동 정렬한다.

---

## 1. 개요

```
RealSense (RGB + aligned depth)
    │
    ├─► world_origin_node ─── ArUco 마커 인식 → static TF camera→world(=robot base)
    │                          (15 s 검출 실패 시 depth plane-fit으로 자동 폴백)
    │
    ├─► detection_node ─────── YOLO seg → SegmentedObjectArray
    │
    └─► point_cloud_node ───── 누적 윈도우 포인트 클라우드 + 3D box + cup frustum
```

패키지 구성:

| 패키지 | 설명 |
|---|---|
| `depth_digital_twin` | ROS 노드 + 런치 파일 (Python) |
| `depth_digital_twin_msgs` | 커스텀 메시지 `SegmentedObject`, `SegmentedObjectArray` (CMake) |

---

## 2. 기능

- **ArUco-origin world frame**: 작업 공간에 놓인 ArUco 마커(ID 0, 4×4)를 인식해 카메라→로봇 base 정적 TF를 자동 발행. 마커가 없으면 depth plane-fit으로 폴백.
- **YOLO segmentation**: cup 클래스 검출 + 인스턴스 마스크.
- **Speed Stack 컵 frustum 모델**: 위·아래 지름/높이를 사전 정보로 활용해 자기 가림이 있어도 컵 전체 크기의 box를 생성. frustum wireframe 동시 발행.
- **누적 윈도우 파이프라인**: 0.1 s 윈도우 동안 포인트를 누적 후 MAD 필터 → 박스 fit. depth 단발 노이즈로 인한 box 진동 억제.
- **Doosan URDF 통합**: robot_state_publisher + `/dsr01/joint_states` 미러를 통해 로봇 모델과 컵 검출이 같은 좌표계에 표시됨.
- **카메라 내부 파라미터 캘리브레이션**: 체커보드 캡처 → `calibrate` → `intrinsics.yaml`.

---

## 3. 실행 방법

### 3.1 사전 준비

```bash
# ROS 2 Humble + 의존성
sudo apt install ros-humble-realsense2-camera ros-humble-cv-bridge \
                 ros-humble-tf2-ros ros-humble-robot-state-publisher

pip install ultralytics opencv-python

# 빌드
cd ~/Projects/ros2-depth-point-cloude
colcon build --symlink-install
source install/setup.bash
```

체커보드 캘리브레이션을 아직 안 했다면 먼저 실행 (한 번만):

```bash
# 캡처
ros2 run depth_digital_twin capture_chessboard --output ./data --board 10x7

# 계산
ros2 run depth_digital_twin calibrate \
    --images "./data/chess_*.png" --board 10x7 --square 25 \
    --output src/depth_digital_twin/config/intrinsics.yaml
```

### 3.2 카메라 실행 (별도 터미널)

```bash
ros2 launch realsense2_camera rs_align_depth_launch.py \
    depth_module.depth_profile:=1280x720x30 \
    rgb_camera.color_profile:=1280x720x30 \
    initial_reset:=true align_depth.enable:=true
```

> 2대 카메라 사용 시 (exo + hand — 아래 3.4 참고): RealSense 시리얼 번호로 구분.
> ```bash
> # exo 카메라
> ros2 launch realsense2_camera rs_align_depth_launch.py \
>     camera_name:=camera_exo serial_no:=<SERIAL_EXO> ...
> # hand 카메라
> ros2 launch realsense2_camera rs_align_depth_launch.py \
>     camera_name:=camera_hand serial_no:=<SERIAL_HAND> ...
> ```

### 3.3 Exo view 실행 (현재 구현)

1. **ArUco 마커(ID 0, DICT_4X4_50)를 카메라가 볼 수 있는 위치에 배치.**
   - 마커가 테이블 위 평면에 놓인 경우: 마커 normal(Z축) = world Z-up.
   - `params.yaml`의 `world_marker_offset_*` 에 마커 위치(robot base 기준, 단위 m)를 설정.
   - `world_marker_rot_z_deg`를 RViz 확인 후 조정해 world +Y 축이 robot base +Y와 정렬되도록.

2. **pipeline만 실행 (RViz 포함):**

```bash
source install/setup.bash
ros2 launch depth_digital_twin digital_twin.launch.py
```

3. **Doosan 로봇 URDF 함께 표시:**

```bash
source ~/ros2_ws/install/setup.bash   # dsr_description2 필요
source install/setup.bash

ros2 launch depth_digital_twin digital_twin_with_robot.launch.py \
    model:=m0609
```

> `with_pose_bridge:=true` (기본): `/dsr01/joint_states`가 들어오면 URDF가 실제 로봇 자세로 움직임. 없으면 home pose 고정.

**RViz 확인 포인트:**
- world 프레임 axes (X=빨강, Y=초록, Z=파랑)가 로봇 base axes와 일치하는지 확인.
- 불일치 시 → `params.yaml` 의 `world_marker_rot_z_deg` 조정 후 재실행.

**startup 로그 예시:**

```
[world_origin_node] ArUco mode: ID=0 dict=DICT_4X4_50 length=5.0cm target=30 samples.
[world_origin_node] Marker [1/30] reproj=0.82px dist=94.3cm
...
[world_origin_node] Marker average pose:
  position (cm) = (23.4, -41.2, 89.1)
  euler_xyz (deg) = (88.3, 1.2, -2.1)
  position std (mm) = (0.9, 1.1, 1.4)
  → Check RViz: world +X (red), +Y (green), +Z (blue) should match robot base.
[world_origin_node] [aruco-origin] Static TF published: camera_color_optical_frame → world
```

### 3.4 Hand view (예정)

카메라를 Doosan 그리퍼에 장착해 gripper-mounted camera로 segmentation 및 3D 위치 추정.

- 현재 미구현.
- **2대 카메라 구성 시 namespace 분리 방안**:
  - RealSense 시리얼 번호로 두 카메라를 구분 (각각 `camera_exo`, `camera_hand` 네임스페이스).
  - `detection_node`, `point_cloud_node`를 각 카메라 네임스페이스에 맞게 인스턴스 2개 실행.
  - `world_origin_node`는 exo 카메라 기준으로 world 프레임 설정; hand 카메라는 TF로 연결.

---

## 4. Launch 파일 상세

### `digital_twin.launch.py`

exo 카메라 기반 파이프라인. RViz2 포함.

```bash
ros2 launch depth_digital_twin digital_twin.launch.py [args]
```

| arg | default | 설명 |
|---|---|---|
| `intrinsics` | `config/intrinsics.yaml` | 카메라 내부 파라미터 경로 |
| `params` | `config/params.yaml` | 노드 파라미터 YAML 경로 |
| `rviz` | `true` | RViz2 실행 여부 |
| `rviz_config` | `rviz/digital_twin.rviz` | RViz2 설정 파일 경로 |

실행하는 노드: `world_origin_node`, `detection_node`, `point_cloud_node`, `rviz2`

### `digital_twin_with_robot.launch.py`

`digital_twin.launch.py` + Doosan URDF + joint_states 브릿지 + world↔base_0 identity TF.

```bash
ros2 launch depth_digital_twin digital_twin_with_robot.launch.py [args]
```

| arg | default | 설명 |
|---|---|---|
| `model` | `m0609` | Doosan 모델명 (`m0609`, `m1013`, …) |
| `color` | `white` | URDF 색상 |
| `name` | `dsr01` | 로봇 네임스페이스 (dsr_bringup2 기본값과 일치해야 함) |
| `with_pose_bridge` | `true` | `/dsr01/joint_states`→`/joint_states` 브릿지 활성화 |
| `rviz` | `true` | RViz2 실행 여부 |
| `rviz_config` | `rviz/digital_twin.rviz` | RViz2 설정 파일 |
| `intrinsics` | `config/intrinsics.yaml` | 카메라 내부 파라미터 경로 |
| `params` | `config/params.yaml` | 노드 파라미터 YAML 경로 |

추가 실행 노드: `robot_state_publisher`, `robot_pose_bridge_node`, `static_transform_publisher`(world↔base_0)

### `digital_twin_sequence.launch.py` (Phase 2a — 녹화 시퀀스 재생)

라이브 카메라 대신 `ros2-recode-sequence`로 **녹화한 시퀀스**(exo RGB-D)를
기존 파이프라인에 그대로 투입한다. 코어 노드(world_origin/detection/
point_cloud) **수정 없음**. exo 카메라 intrinsics는 시퀀스 `meta.json`에서
자동 추출(`<sequence>/exo_intrinsics.yaml`)되어 파이프라인에 전달된다.

```bash
# 두 워크스페이스 모두 source
source ~/Projects/ros2-recode-sequence/install/setup.bash
source ~/Projects/ros2-depth-point-cloude/install/setup.bash

# exo (기본)
ros2 launch depth_digital_twin digital_twin_sequence.launch.py \
    sequence:=/home/eunwoosong/Projects/record_sequence/0005
# hand 카메라로 보기  (ROS2는 --hand 가 아니라 view:=hand)
ros2 launch depth_digital_twin digital_twin_sequence.launch.py \
    sequence:=/home/eunwoosong/Projects/record_sequence/0005 view:=hand
```

| arg | default | 설명 |
|---|---|---|
| `sequence` | *(필수)* | 녹화 시퀀스 폴더 절대경로 (`record_sequence/NNNN`) |
| `view` | `exo` | `exo`\|`hand` — 파이프라인에 투입할 녹화 카메라. 해당 카메라 intrinsics를 meta.json에서 추출해 사용 |
| `yolo_model` | `''` | 명시 지정 시 우선 적용. 비우면 **params.yaml의 `detection_node.model_<view>`** 사용 (`model_exo`/`model_hand`) |
| `loop` | `false` | 끝에서 정지(미순환) / `true` 시 반복 |
| `autostart` | `true` | 즉시 재생 |
| `params` | `config/params.yaml` | 파이프라인 파라미터 |
| `rviz` | `true` | RViz2 실행 |

hand 특화 YOLO는 `params.yaml`의 `detection_node.model_hand` 경로를
교체하면 `view:=hand` 시 자동 적용된다(`model_exo`는 exo용, `model`은
라이브/기본). 임시 오버레이 파일을 만들지 않는다.

> ⚠️ `view:=hand`: 손목 장착 hand 카메라는 보통 ArUco 마커를 보지 못하므로
> `world_origin_node`가 15초 후 **floor-plane fallback**으로 전환된다(정상).
> EE + hand-eye 기반의 정확한 hand→world 정렬은 **Phase 2b**.

데이터 흐름: `sequence_player_node`(exo→`/camera/camera/color|aligned_depth`,
frame=`camera_color_optical_frame`) → `world_origin_node`가 재생 프레임의
ArUco로 world 보정 → `detection_node`+`point_cloud_node`가 컵 검출. 재생
제어는 `playback_control` 패널(Stop/Resume/Replay/Step+Apply). hand 카메라
융합(2b)은 후속.

---

## 5. params.yaml 기능 정리

파일 위치: `src/depth_digital_twin/config/params.yaml`

### `/**:` 글로벌 (전 노드 공유)

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `camera_frame` | `camera_color_optical_frame` | 카메라 optical frame 이름 |
| `world_frame` | `world` | world frame 이름 |
| `depth_unit` | `0.001` | depth 이미지 단위 (mm→m 변환계수) |
| `window_radius` | `30` | floor patch 반경 (px) |
| `window_center_x_px` | `640` | floor patch 중심 x (-1 = 이미지 중앙) |
| `window_center_y_px` | `600` | floor patch 중심 y |

### `world_origin_node`

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `world_origin_mode` | `aruco` | `aruco` (마커 인식) \| `floor` (depth 평면 fit) |
| `color_topic` | `/camera/camera/color/image_raw` | 컬러 이미지 토픽 (aruco 모드) |
| `world_marker_id` | `0` | 검출할 ArUco 마커 ID |
| `world_marker_dict` | `DICT_4X4_50` | ArUco 딕셔너리 |
| `world_marker_length_m` | `0.05` | 마커 한 변 길이 (m) — **실측 후 반드시 설정** |
| `world_marker_samples_required` | `30` | TF 발행 전 평균낼 프레임 수 |
| `world_marker_reproj_err_max_px` | `2.0` | solvePnP 재투영 오차 허용치 (px); 초과 시 샘플 기각 |
| `world_marker_timeout_s` | `15.0` | 마커 미검출 허용 시간 (초); 초과 시 floor 폴백 |
| `aruco_timeout_then_floor` | `true` | 타임아웃 후 floor 모드로 폴백 여부 |
| `world_marker_offset_x_m` | `0.367` | 마커 위치 — robot base 기준 X (m) |
| `world_marker_offset_y_m` | `0.003` | 마커 위치 — robot base 기준 Y (m) |
| `world_marker_offset_z_m` | `0.0` | 마커 위치 — robot base 기준 Z (m) |
| `world_marker_rot_x_deg` | `0.0` | 마커 frame → base frame 회전 Euler X (deg) |
| `world_marker_rot_y_deg` | `0.0` | 마커 frame → base frame 회전 Euler Y (deg) |
| `world_marker_rot_z_deg` | `0.0` | 마커 frame → base frame 회전 Euler Z (deg) |
| `depth_topic` | `…aligned_depth…` | depth 이미지 토픽 (floor 모드) |
| `min_patch_points` | `100` | floor fit: patch 내 최소 유효 픽셀 수 |
| `max_plane_residual` | `0.1` | floor fit: 평면 잔차 허용치 (m) |
| `samples_required` | `10` | floor fit: TF 발행 전 평균 프레임 수 |

> **마커 크기 설정 중요**: `world_marker_length_m`은 실제 인쇄된 마커의 한 변 길이를 자로 측정해 정확히 입력. 이 값이 틀리면 위치 추정이 비례해서 틀림.

> **첫 실행 후 RViz에서 world 축 방향 확인**: 로봇 base +X(빨강)/+Y(초록)/+Z(파랑)와 불일치 시 `world_marker_rot_z_deg` 값을 90° 단위로 바꾸며 테스트.

### `detection_node`

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `model` | `…yolo26m-seg-best.pt` | YOLO checkpoint 경로 (절대경로 or Ultralytics 모델명) |
| `target_classes` | `["cup"]` | 검출 대상 클래스 |
| `confidence` | `0.35` | 검출 confidence threshold |
| `image_topic` | `/camera/camera/color/image_raw` | 입력 컬러 이미지 |
| `detections_topic` | `/digital_twin/detections` | 출력 검출 결과 |
| `debug_topic` | `/digital_twin/detection_debug` | 시각화 이미지 (mask overlay) |
| `device` | `""` | 추론 디바이스 (빈 문자열 = auto) |
| `imgsz` | `1280` | YOLO 추론 해상도 (px). 학습 해상도(640)와 달라도 무방 |

### `point_cloud_node`

**기본 설정**

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `downsample` | `2` | 포인트 다운샘플 stride |
| `z_min` / `z_max` | `0.1` / `4.0` | 유효 depth 범위 (m) |
| `mask_erode_px` | `13` | YOLO 마스크 침식 px — 경계 혼합 픽셀 제거 |
| `depth_gradient_max_m` | `0.015` | depth Laplacian 임계 (m) — 경계 mixed-pixel 추가 제거 |
| `box_outlier_mad_k` | `3.0` | MAD 필터 k값 (낮을수록 공격적; 0 = 비활성) |
| `approx_sync_slop` | `0.05` | RGB/depth 시간 동기화 허용 오차 (s) |
| `window_period_s` | `0.1` | 누적 윈도우 주기 (s) — 작을수록 box 업데이트 빠름 |

**Cup frustum 모델**

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `cup_top_diameter_m` | `0.054` | 컵 윗면 지름 (m) |
| `cup_bottom_diameter_m` | `0.078` | 컵 아랫면 지름 (m) |
| `cup_height_m` | `0.095` | 컵 높이 (m) |
| `cup_polygon_segments` | `24` | frustum 원 분할 수 (wireframe) |
| `cup_smoothing_alpha` | `0.3` | 축 위치 EMA 계수 (1.0 = 비활성) |
| `cup_track_keepalive_frames` | `10` | 검출 소실 후 마커 유지 프레임 수 |
| `cup_fit_residual_max` | `0.02` | frustum fit 잔차 임계 (m); 초과 시 OBB fallback |

**Box 판정**

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `box_standing_ratio` | `0.8` | Z extent / XY max 비율 임계 — 초과 시 standing(AABB) |
| `box_min_elongation` | `1.5` | PCA elongation 최솟값 — 미달 시 AABB |
| `box_force_aabb` | `false` | true = OBB 비활성, 항상 AABB |
| `box_line_width` | `0.0015` | RViz LINE_LIST 두께 (m) |
| `box_alpha` | `0.25` | CUBE 마커 투명도 |

---

## 참고

- `sample/` : Doosan 로봇 가이드 PDF, 체커보드 캘리브레이션 예시, eye-to-hand/eye-in-hand 참조 코드
- `legacy/` : 구 eye-to-hand 캘리브레이션 파일 (aruco_calibrate, aruco_handeye) — 현재 미사용
