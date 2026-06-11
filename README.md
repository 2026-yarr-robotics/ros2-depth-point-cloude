# ROS 2 Depth Digital Twin

RealSense D435i + YOLO segmentation 기반 **컵 디지털 트윈** (단일/듀얼 카메라).
작업 공간에 배치된 ArUco 마커를 인식해 **world 프레임 = 로봇 base 프레임**으로 자동 정렬한다.

직립 컵의 위치 측정은 **실루엣(rim) 방식**이 기본이다: 컵은 제원이 알려진
truncated cone(자유도 = 축 x,y)이므로, YOLO 마스크 윤곽 + 카메라 캘리브레이션이
노이즈 많은 depth 점군보다 훨씬 정밀하게 축을 구속한다 (1 m에서 마스크 1 px ≈ 1.5 mm
vs 컵 벽면 depth 바이어스 mm~cm). **depth는 단 수(레벨) 분류로 강등**되어 z를 격자
(포개기 `nesting_offset_m` / 피라미드 `rim_layer_height_m` = 서버
`PYRAMID_LAYER_HEIGHT`)에 스냅하는 데만 쓰인다. 융합 모드에서 직립 컵 점군은 더
이상 생성하지 않으며(`upright_clouds: false`), 점군 경로는 누운 컵 OBB 전용으로
남아 있다. 롤백: `cup_fusion_node fit_source: cloud`.

---

## 1. 개요

```
RealSense (RGB + aligned depth)                ← 카메라당 1세트 (exo / hand)
    │
    ├─► world_origin_node ─── ArUco 마커 인식 → static TF camera→world(=robot base)
    │                          (hand: handeye_aruco — FK 체인 + 핸드아이 자동 산출)
    │
    ├─► detection_node ─────── YOLO seg + ByteTrack → SegmentedObjectArray
    │
    └─► point_cloud_node ──┬── CupObservationArray  (직립 컵: 실루엣 fit + 시선 광선)
                           └── WorldObjectCloudArray (누운 컵 전용 world 점군)
                                    │
                       cup_fusion_node (fit_source=rim)
                           · 관측 3D 클러스터 → z 격자 스냅 → 광선 슬라이드
                           · 역공분산 융합 + extrinsic 자가 보정 + 컵당 KF
                           → /digital_twin/boxes, /vision/cups_on_table,
                             /digital_twin/fusion_health (잔차·바이어스 JSON)
```

패키지 구성:

| 패키지 | 설명 |
|---|---|
| `depth_digital_twin` | ROS 노드 + 런치 파일 + `cup_geometry.py` 측정 수학 (Python) |
| `depth_digital_twin_msgs` | `SegmentedObject(Array)`, `WorldObjectCloud(Array)`, `CupObservation(Array)` (CMake) |

---

## 2. 기능

- **ArUco-origin world frame**: 작업 공간에 놓인 ArUco 마커(ID 0, 4×4)를 인식해 카메라→로봇 base 정적 TF를 자동 발행. 마커가 없으면 depth plane-fit으로 폴백.
- **YOLO segmentation**: cup 클래스 검출 + 인스턴스 마스크 (+ ByteTrack id).
- **실루엣(rim) 측정** (`cup_geometry.py`): 마스크 윤곽 거리변환에 cone 실루엣을
  chamfer 정렬(soft_l1, 2-DOF). 세 번째 파라미터 b가 YOLO 경계의 균일 과소/과대
  세그먼트 바이어스를 흡수하고, image-gradient **edge-snap**이 최종 경계를 마스크가
  아닌 실제 영상 에지에 재정렬한다. 정지 컵 per-frame σ 1.6–2.9 mm (seq 0010,
  imgsz 1280). 디버그 오버레이: `/digital_twin/rim_debug_{exo,hand}`.
- **듀얼 카메라 역공분산 융합** (`cup_fusion_node fit_source: rim`): 카메라별 관측
  (`CupObservation` = fit 축 + 시선 광선 + 품질 + 색)을 3D 타원체 클러스터 →
  z 이중 격자 스냅(포개기/피라미드 중 rough z에 맞는 쪽) → 광선 슬라이드 →
  역공분산 융합. 시선이 비스듬할수록 시선 방향 분산을 키워 exo(원거리·경사)와
  hand(근거리·수직)가 올바른 가중치로 섞이고, **한쪽 카메라에만 보이는 컵도
  단독 트랙**으로 유지된다. 모션 중 hand 관측은 폐기(`rim_drop_moving`).
- **extrinsic 자가 보정**: 두 카메라가 같은 컵을 보면 그 (x,y) 차이가 곧 extrinsic
  불일치다. EMA로 비기준 카메라(exo)의 world 바이어스를 추정해 보정
  (`rim_bias_apply`). seq 0010: exo ArUco 바이어스 (+15.8, −21.4) mm 수렴, 공유 컵
  잔차 30 mm → 6–11 mm. 진단 JSON: `/digital_twin/fusion_health`.
- **Speed Stack 컵 frustum 모델**: 위·아래 지름/높이를 사전 정보로 활용해 자기 가림이 있어도 컵 전체 크기의 box를 생성. frustum wireframe 동시 발행.
- **누적 윈도우 파이프라인** (점군 경로 — 누운 컵 OBB 전용): 윈도우 누적 + MAD 필터 → 박스 fit.
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

**⭐ 권장: RealSense 필터 활성화 + Laser Power 조절**

```bash
# 포인트 클라우드 노이즈 극적 감소 (경사 탑다운 뷰 필수)
ros2 launch realsense2_camera rs_align_depth_launch.py \
    depth_module.depth_profile:=1280x720x30 \
    rgb_camera.color_profile:=1280x720x30 \
    initial_reset:=true \
    align_depth.enable:=true \
    decimation_filter.enable:=true \
    spatial_filter.enable:=true \
    temporal_filter.enable:=true \
    hole_filling_filter.enable:=true \
    visual_preset:=High\ Accuracy \
    laser_power:=60
```

**필터 설명:**
- `decimation_filter` (데시메이션): 해상도 낮춤/전체 노이즈 감소
- `spatial_filter` (공간): 빈틈채우기 + 부드러운 표면 보정
- `temporal_filter` (시간): 프레임 간 누적으로 깜빡이는 노이즈 제거
- `hole_filling_filter` (홀필링): 남은 빈 공간 메우기
- `laser_power:=60`: 플라스틱 컵 반사 감소 (기본값 80)

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

### 3.4 Hand view (구현됨 — UC-4 참고)

손목(link_6) 장착 카메라는 `world_origin_node`의 `handeye_aruco` 모드가 핸드아이를
런타임에 자동 산출하고, FK 체인(`base_link→link_6→hand_color_optical_frame`)으로
world 변환을 공급한다. exo + hand 동시 구동·융합은 `digital_twin_fusion.launch.py`
하나로 끝난다 — 토폴로지/실행법은 아래 **UC-4** 참고.

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
    sequence:=/home/eunwoo/Projects/cup_stack/seq_record/0010
# hand 카메라로 보기  (ROS2는 --hand 가 아니라 view:=hand)
ros2 launch depth_digital_twin digital_twin_sequence.launch.py \
    sequence:=/home/eunwoo/Projects/cup_stack/seq_record/0010 view:=hand
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
| `cup_track_keepalive_frames` | `10` | 검출 소실 후 마커 유지 프레임 수 |
| `cup_fit_residual_max` | `0.02` | frustum fit 잔차 임계 (m); 초과 시 OBB fallback |

**Kalman 위치 필터** (기존 EMA + scan-and-lock 대체)

트랙별 constant-position 칼만 필터로 컵 중심을 추정한다. 추정값을 **freeze하지 않으므로**, 한 번 잘못 추정돼도 이후 윈도우들이 계속 보정해 스스로 수렴한다 (기존 LOCKED 박스는 컵이 물리적으로 3 cm 이상 움직이기 전까지 고정됨). 일시적 depth spike는 Mahalanobis gate로 기각하고, gate-out이 연속되면 실제 이동(로봇 pick/place)으로 보고 해당 위치로 재획득(re-acquire)한다.

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `kf_process_std_xy_m` | `0.002` | 윈도우당 허용 drift (XY) — 클수록 보정 빠름/지터 증가 |
| `kf_process_std_z_m` | `0.004` | 윈도우당 허용 drift (Z, depth가 더 noisy) |
| `kf_meas_std_xy_m` | `0.005` | 윈도우 fit 측정 오차 가정 (XY) |
| `kf_meas_std_z_m` | `0.010` | 윈도우 fit 측정 오차 가정 (Z) |
| `kf_init_std_m` | `0.05` | 초기/재획득 공분산 std (첫 측정 신뢰) |
| `kf_gate_mahalanobis` | `9.0` | χ²(3)≈97% gate; 초과 측정은 spike로 기각, 0=비활성 |
| `kf_reacquire_windows` | `3` | 연속 gate-out N회 → 이동으로 판정해 재획득 |
| `kf_settled_std_m` | `0.006` | 위치 1σ가 이 값 이하면 "settled" (마커 `[L]` 표기) |
| `kf_resid_infl` | `1.0` | residual/cup_fit_residual_max 비례로 측정노이즈 inflate |

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

---

# USE CASE

## UC-1. 라이브 단독 실행 (exo 카메라 + 로봇)

로봇과 exo 카메라가 연결된 상태에서 실시간 컵 검출 + pick UI.

```bash
# 터미널 1: 로봇 bringup
source ~/ros2_ws/install/setup.bash
ros2 launch dsr_bringup2 dsr_bringup2_rviz.launch.py \
    mode:=real model:=m0609 host:=192.168.1.100
ros2 service call /dsr01/system/set_robot_mode \
    dsr_msgs2/srv/SetRobotMode "robot_mode: 0"

# 터미널 2: exo 카메라 (serial 고정, hand 카메라 충돌 방지)
source ~/Projects/ros2-recode-sequence/install/setup.bash
ros2 launch recode_sequence cameras_only.launch.py view:=exo

# 터미널 3: detection 파이프라인 (exo 토픽 구독)
source ~/Projects/ros2-depth-point-cloude/install/setup.bash
ros2 launch depth_digital_twin digital_twin.launch.py camera_ns:=exo

# 터미널 4: pick UI
ros2 run depth_digital_twin pick_ui_node
```

> `camera_ns:=exo` 를 주면 `/exo/exo/color/image_raw` 등 exo 토픽을 자동으로 구독한다.
> 기본값(`camera_ns:=camera`)은 `rs_align_depth_launch.py` 단독 실행 시 토픽과 호환된다.

---

## UC-2. 녹화 시퀀스 재생 (오프라인 검증)

라이브 카메라·로봇 없이 녹화된 시퀀스로 파이프라인 검증.

```bash
# 두 워크스페이스 source
source ~/Projects/ros2-recode-sequence/install/setup.bash
source ~/Projects/ros2-depth-point-cloude/install/setup.bash

# exo view (기본)
ros2 launch depth_digital_twin digital_twin_sequence.launch.py \
    sequence:=/home/eunwoo/Projects/cup_stack/seq_record/0010

# pick UI (별도 터미널)
ros2 run depth_digital_twin pick_ui_node
```

재생 제어: `playback_control` 패널(Stop / Resume / Replay / Step+Apply).

---

## UC-3. 녹화 시퀀스 재생 — hand view (FK 기반 world 변환)

hand 카메라 FK 체인 검증 (ArUco 불필요, joint FK + handeye 사용).

```bash
# 세 워크스페이스 모두 source
source ~/ros2_ws/install/setup.bash   # dsr_description2 (m0609 URDF)
source ~/Projects/ros2-recode-sequence/install/setup.bash
source ~/Projects/ros2-depth-point-cloude/install/setup.bash

ros2 launch depth_digital_twin digital_twin_sequence.launch.py \
    sequence:=/home/eunwoo/Projects/cup_stack/seq_record/0010 view:=hand
```

> ⚠ 라이브 `dsr_bringup2`가 켜져 있으면 TF 충돌. 종료 후 실행하거나 `ROS_DOMAIN_ID` 분리.

---

## UC-4. Hand/Exo 통합 재생 및 실행 (rim 융합 — 기본 경로)

녹화 시퀀스에서 **exo + hand 두 카메라를 동시에** 파이프라인에 투입하고, 두 뷰의
실루엣 관측을 **하나의 물리 컵으로 융합**해 통합 추정한다. 이미지는 RViz가 아니라
**통합 Tk 패널**(`digital_twin_panel`)에 뜬다.

```bash
# 세 워크스페이스 모두 source
source ~/ros2_ws/install/setup.bash   # dsr_description2 (m0609 URDF, joint FK)
source ~/Projects/ros2-recode-sequence/install/setup.bash
source ~/Projects/yarr_projects/install/setup.bash

ros2 launch depth_digital_twin digital_twin_fusion.launch.py \
    sequence:=/home/eunwoo/Projects/cup_stack/seq_record/0010 \
    loop:=true
```

### 토폴로지 (Producer → Fusion, fit_source=rim)

```
sequence_player ─► /camera_exo/*, /camera_hand/*, /joint_states
  ├─ world_origin_node_exo  (ArUco)        → world ← exo_color_optical_frame
  ├─ world_origin_node_hand (handeye_aruco)→ link_6 → hand_color_optical_frame
  ├─ detection_node_exo|hand (YOLO-seg + ByteTrack) → /digital_twin/detections_*
  ├─ point_cloud_node_exo|hand (role=producer)
  │      → /digital_twin/cup_obs_{exo,hand}   (직립: 실루엣 fit 관측 — 주 측정)
  │      → /digital_twin/cups_{exo,hand}      (누운 컵 전용 점군; upright_clouds=false)
  │      → /digital_twin/rim_debug_{exo,hand} (fit 오버레이: 윤곽·실루엣·실패 사유)
  └─ cup_fusion_node (fit_source=rim)
         · 관측 3D 타원체 클러스터(피라미드 층 분리) → z 이중 격자 스냅
         · 광선 슬라이드 → extrinsic 바이어스 보정 → 역공분산 융합 → 컵당 KF
         → /digital_twin/boxes, /vision/cups_on_table,
           /digital_twin/fusion_health (컵별·카메라별 잔차 + 바이어스 JSON)
```

핵심: per-camera 노드는 **관측 생산자**일 뿐, 융합·Kalman·박스 발행은 모두
`cup_fusion_node`가 단독으로 한다. 한쪽 카메라에만 보이는 컵(exo 사각지대 등)도
그 카메라 단독 트랙으로 유지된다. 레거시 점군 경로로 롤백:
`ros2 param set /cup_fusion_node fit_source cloud`.

### 융합 상태 진단

```bash
ros2 topic echo /digital_twin/fusion_health   # 컵별 카메라 잔차(mm), 레벨,
                                              # extrinsic_bias_mm (exo 자가 보정)
```

정지 장면에서 공유 컵의 카메라 간 잔차가 크게(>15 mm) 유지되면 extrinsic 문제다 —
자가 보정(`rim_bias_apply`)이 EMA로 수렴할 때까지 수십 초 기다리거나, ArUco
재검출 버튼으로 캘리브레이션 자체를 갱신한다.

### 통합 Tk 패널 (`digital_twin_panel`)

- 상단: **ArUco 재검출 버튼 3개 / 2행** — `ArUco Re-detect All (hand, exo)`,
  그 아래 `ArUco Exo` · `ArUco Hand` (고정 크기, 가운데 정렬). 각 버튼은 해당
  카메라의 `world_origin_node_{exo,hand}/redetect` 서비스를 호출한다.
- 본문: **2행 × 3열 이미지 그리드** — (exo/hand) × (RGB, Depth, 3D). **3D 열은
  rim fit 오버레이**(`/digital_twin/rim_debug_*`): 관측 윤곽(녹), fit 실루엣(시안),
  depth 초기값(빨강), 컵별 `iou/rms/b/cov` 텍스트(실패 시 사유), ArUco/base 축
  투영. 원시 YOLO 박스는 `/digital_twin/detection_debug_*`에 그대로 남아 있다.
- **Debug plot 행**: `1 H-cloud(주황, 기본 on) / 2 H-box / 3 E-cloud(파랑, 기본
  on) / 4 E-box / 5 F(final, 기본 on) | Use hand(live, 기본 off)`.
  H/E 채널은 카메라별 단색 러프 표시(`/digital_twin/points_{exo,hand}`,
  `/digital_twin/dbg_boxes_{exo,hand}`, 박스에는 `Hand1`/`Exo2` 텍스트)이고,
  정밀 추정은 **F = `/digital_twin/boxes`** 하나뿐이다. 융합 `/digital_twin/points`
  토픽은 제거되었다.
- **Use hand(live)**: 해제(기본) 시 라이브 fit은 exo 단독 — hand 검출은 패널에는
  보이지만 RViz/측정에는 들어가지 않는다. 스캔으로 동결된 hand 관측([S])은 이
  설정과 무관하게 항상 사용된다.
- **라벨 v2**: `[F] [S] #N <color> cup(x, y, z)` — `[F]`=exo+hand 융합,
  `[S]`=scan 지지(둘 다면 `[F] [S]`), 좌표는 KF 중심. 구 `[L]`(settled) 태그는
  폐지. 다운스트림 파서(skill-manager/plan_executor/pick_node/
  boxes_to_detections)는 구·신 포맷을 모두 수용한다.
- **Scan**: skill-manager가 `scan_lock_active`를 켜면 관절이 `scan_waypoints_deg`
  범위로 들어올 때 1 s 대기 후 1 s간 hand 관측을 동결한다(점군 lock 아님 — exo는
  항상 라이브 재피팅). exo가 못 보는 컵은 `[S]`로 영구 추적되고, exo와 한 번이라도
  융합된 `[S]` 컵은 exo가 놓치면 함께 사라진다. 패널 **Clear Scan**(=`~/clear_scan`)
  으로 동결 해제; replay/sim 검증용 `~/capture_scan_now` 서비스도 있다.
- 기존 `world_origin_control` 팝업은 이 패널로 대체된다. (구 Scan&Lock /
  Lock exo too / Clear Lock 컨트롤은 제거되었다.)

> ⚠ 라이브 `dsr_bringup2`가 켜져 있으면 `world→base_link→…→link_6` TF가 충돌한다.
> 종료 후 실행하거나 `ROS_DOMAIN_ID`/`ns:=`로 분리한다.
>
> ⚠ 듀얼 YOLO + 듀얼 point_cloud는 무겁다. 전송 폭주로 `/tf`가 밀리면 컵이 어긋나므로
> 필요시 `playback_rate:=`(저속 재생)로 대역폭을 낮춘다.

---

## UC-5. pick UI 재스캔 트리거

pick 전 최신 포즈로 갱신하고 싶을 때:

```bash
ros2 service call /point_cloud_node/trigger_scan std_srvs/srv/Trigger
```

또는 pick UI 창의 **⟳ Re-scan** 버튼 클릭.

---

## UC-6. 오프라인 실루엣 fit 검증 (ROS 불필요)

녹화 시퀀스에 대해 실루엣 측정 수학(`cup_geometry.py`)을 ROS 없이 단독 검증한다.
회귀 테스트·파라미터 튜닝·fit 품질 정량화에 사용.

**단일 프레임** — 컵별 fit 표 + 오버레이 PNG:

```bash
cd ~/Projects/cup_stack/cup-stack-integration/vision/ros2-depth-point-cloude
python3 src/depth_digital_twin/test/fit_check_frame.py \
    --seq /home/eunwoo/Projects/cup_stack/seq_record/0010 --frame 1666
# → /tmp/fit_check_0010_001666_exo.png (윤곽 녹 / fit 실루엣 시안 / depth 초기값 빨강)
```

**시퀀스 전체** — ByteTrack id 기반 트랙별 통계 + CSV + 오버레이 영상:

```bash
python3 src/depth_digital_twin/test/fit_check_sequence.py \
    --seq /home/eunwoo/Projects/cup_stack/seq_record/0010 --stride 3 \
    --csv /tmp/fit_0010.csv --video /tmp/fit_0010.avi
# 종료 시 트랙별 요약: 정지 구간 σ_xy (fit vs depth-init A/B), IoU/rms, 실패율
```

- 영상은 `.avi` 권장 (**MJPEG** — 코덱 팩 없이 어디서나 재생). `.mp4`는 mp4v로
  쓴 뒤 ffmpeg가 있으면 H.264 재인코딩.
- ablation 플래그: `--no-boundary-offset`, `--no-edge-snap`, `--try-flip`
  (mouth-up 프로파일), `--imgsz`(기본 1280 = 라이브와 동일).
- 합성 단위 테스트(렌더링된 cone 마스크로 수학 자체를 검증, 13개):
  `python3 src/depth_digital_twin/test/test_cup_geometry.py`

측정 기준치 (seq 0010, imgsz 1280): fit 실패율 0 %, 정지 컵 per-frame σ_xy
1.6–2.9 mm, exo 마스크 경계 바이어스 b ≈ −0.95 px.
