"""Phase 2b — dual-camera (exo + hand) fusion, LIVE or RECORD.

Two parallel pipelines (exo + hand) feed `cup_fusion_node` in a common `world`
frame. The SAME perception nodes run in both modes — only the TRAJECTORY +
camera source differ. The `ns` arg picks the mode (the user's rule:
"no ns + bringup running ⇒ live; ns set ⇒ use the recorded trajectory"):

  ns EMPTY + NO sequence  → LIVE   (default): defer the trajectory to the real
      robot — `robot_pose_bridge_node` mirrors `/dsr01/joint_states`→`/joint_states`,
      our robot_state_publisher does the m0609 FK, perception consumes the LIVE
      `/camera_exo|hand/*` feeds. NO sequence_player. Intrinsics come from the
      `intr_exo:=`/`intr_hand:=` static YAMLs (no recording → no meta.json).
      Run dsr_bringup2 + the live cameras first.

  ns EMPTY + sequence:=…  → REPLAY at root (legacy standalone playback). Do NOT
      run dsr_bringup2 at the same time (its /joint_states + /tf would fight the
      replay's).

  ns:=record sequence:=…  → REPLAY isolated under /record (joint_states, TF,
      camera, /digital_twin/*, the fusion services). Coexists with a live stack
      at root without conflict (see _namespace_group).

Frame / TF tree (identical in both modes — only the /joint_states SOURCE moves)
  exo_color_optical_frame ──(world_origin_node_exo, ArUco)──► world
  world ──(robot_state_publisher m0609, world_fixed=identity)──► base_link
  base_link ──(URDF FK from /joint_states)──► … ──► link_6
  link_6 ──(world_origin_node_hand, handeye ArUco)──► hand_color_optical_frame

Prerequisites (THREE workspaces sourced — dsr_description2 provides the m0609
URDF used for the joint FK):
  source ~/ros2_ws/install/setup.bash
  source ~/Projects/ros2-recode-sequence/install/setup.bash
  source ~/Projects/ros2-depth-point-cloude/install/setup.bash

Usage:
  # LIVE (verify scan-lock motion against the real robot): bringup + cameras up,
  ros2 launch depth_digital_twin digital_twin_fusion.launch.py \\
      intr_exo:=/abs/exo_intrinsics.yaml intr_hand:=/abs/hand_intrinsics.yaml
  # RECORD alongside live:
  ros2 launch depth_digital_twin digital_twin_fusion.launch.py \\
      ns:=record sequence:=/abs/record_sequence/0010

Args:
  ns          : '' (default) ⇒ LIVE (or root replay if sequence given);
                set (e.g. record) ⇒ namespaced replay alongside a live stack.
  sequence    : recorded sequence dir — REQUIRED for replay, ignored for live.
  intr_exo    : LIVE exo intrinsics YAML (default: package config/intrinsics.yaml)
  intr_hand   : LIVE hand intrinsics YAML (default: package config/intrinsics.yaml)
  dsr_joint_topic : LIVE source joints bridged to /joint_states (default
                    /dsr01/joint_states)
  with_pose_bridge: LIVE — run robot_pose_bridge_node (default true)
  loop        : replay loop (default false)        autostart : play now (default true)
  params      : pipeline params.yaml (default: package config)
  rviz        : launch RViz2 overlay (default true)
"""
import glob
import json
import os
import re

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, GroupAction, OpaqueFunction)
from launch.conditions import IfCondition
from launch.substitutions import (Command, FindExecutable,
                                  LaunchConfiguration, PathJoinSubstitution)
from launch_ros.actions import Node, PushRosNamespace, SetRemap
from launch_ros.descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

_HANDEYE_DEFAULT = ('/home/eunwoosong/Projects/ros2-recode-sequence/'
                    'src/recode_sequence/config/T_gripper2camera.npy')

# Absolute names the replay stack pub/subs that ALSO exist in a live bringup.
# When `ns:=<name>` is set, each is redirected to /<ns>/… so a recorded replay
# runs next to the real robot without fighting over /joint_states, the TF tree,
# the camera feeds, the /digital_twin/* pipeline, or the fusion services.
# (PushRosNamespace moves node names + relative names; these absolute ones —
# hardcoded here and in params.yaml — ignore the namespace, so they need
# explicit remaps. SetRemap also remaps the service NAMES the panel/skill_manager
# build from params, so the namespaced servers are reached.)
_NS_REMAP_NAMES = (
    '/camera_exo/color/image_raw',
    '/camera_exo/aligned_depth_to_color/image_raw',
    '/camera_exo/color/camera_info',
    '/camera_hand/color/image_raw',
    '/camera_hand/aligned_depth_to_color/image_raw',
    '/camera_hand/color/camera_info',
    '/joint_states', '/tf', '/tf_static',
    '/digital_twin/detections_exo', '/digital_twin/detection_debug_exo',
    '/digital_twin/detections_hand', '/digital_twin/detection_debug_hand',
    '/digital_twin/cups_exo', '/digital_twin/cups_hand',
    '/digital_twin/points', '/digital_twin/boxes',
    '/world_origin_node_exo/redetect', '/world_origin_node_hand/redetect',
    '/cup_fusion_node/clear_scan', '/cup_fusion_node/set_parameters',
)


def _namespace_group(ns: str, actions: list):
    """Wrap every replay action under `/<ns>`: push the namespace (node names +
    relative names) and remap the absolute colliding names above into it."""
    remaps = [SetRemap(src=n, dst=f'/{ns}{n}') for n in _NS_REMAP_NAMES]
    return GroupAction([PushRosNamespace(ns), *remaps, *actions])


_TUNING_RE = re.compile(r'^params_\d{6}_\d{6}\.yaml$')


def _latest_tuning(cfg_dir: str):
    """Newest panel-saved tuning snapshot (params_YYMMDD_HHMMSS.yaml) in
    cfg_dir, or None. Matches ONLY the timestamped name so params.yaml and
    params_back.yaml (which sorts AFTER digits!) are never picked. Snapshots
    are PARTIAL — the caller must OVERLAY them on params.yaml, not replace it."""
    snaps = [f for f in glob.glob(os.path.join(cfg_dir, 'params_*.yaml'))
             if _TUNING_RE.match(os.path.basename(f))]
    return max(snaps, key=os.path.basename) if snaps else None


def _export_intrinsics(seq_dir: str, view: str) -> str:
    with open(os.path.join(seq_dir, 'meta.json')) as f:
        meta = json.load(f)
    cam = (meta.get('cameras') or {}).get(view) or {}
    K = cam.get('K') or []
    if len(K) != 9 or not cam.get('width'):
        raise RuntimeError(
            f'sequence {seq_dir}: no {view} intrinsics in meta.json')
    dist = cam.get('dist') or [0.0] * 5
    out = {
        'image_width': int(cam['width']),
        'image_height': int(cam['height']),
        'camera_matrix': {'rows': 3, 'cols': 3,
                          'data': [float(v) for v in K]},
        'distortion_coefficients': {'rows': 1, 'cols': len(dist),
                                    'data': [float(v) for v in dist]},
        'reprojection_error': 0.0,
    }
    p = os.path.join(seq_dir, f'{view}_intrinsics.yaml')
    with open(p, 'w') as f:
        yaml.safe_dump(out, f, sort_keys=False)
    return p


def _setup(context, *_, **__):
    pkg = get_package_share_directory('depth_digital_twin')
    ns = LaunchConfiguration('ns').perform(context).strip().strip('/')
    seq = LaunchConfiguration('sequence').perform(context)
    seq_valid = bool(seq and os.path.isdir(seq))
    # LIVE = no namespace AND no sequence → defer the trajectory to the real
    # robot (dsr_bringup2). A namespace, or a sequence, means REPLAY.
    live = (not ns) and (not seq_valid)

    # The m0609 URDF FK (joints → link_6) is needed in BOTH modes for the
    # robot_state_publisher below.
    try:
        get_package_share_directory('dsr_description2')
    except Exception:
        raise RuntimeError(
            "dsr_description2 not found — the m0609 URDF (joint FK for the "
            "hand camera) needs the doosan workspace:\n"
            "    source ~/ros2_ws/install/setup.bash")

    if live:
        # Intrinsics: no recording → take them from static YAML args.
        intr_exo = os.path.abspath(
            LaunchConfiguration('intr_exo').perform(context))
        intr_hand = os.path.abspath(
            LaunchConfiguration('intr_hand').perform(context))
        for v, p in (('exo', intr_exo), ('hand', intr_hand)):
            if not os.path.isfile(p):
                raise RuntimeError(
                    f'LIVE mode {v} intrinsics not found: {p}\n'
                    f'    pass intr_{v}:=/abs/path/to/{v}_intrinsics.yaml')
        print(f'[fusion] MODE=LIVE  joints←{LaunchConfiguration("dsr_joint_topic").perform(context)}'
              f'  intr_exo={intr_exo}  intr_hand={intr_hand}')
    else:
        if not seq_valid:
            raise RuntimeError(
                'REPLAY needs sequence:=/abs/path/to/record_sequence/NNNN '
                f'(got {seq!r}). For LIVE, omit both ns and sequence and run '
                'dsr_bringup2 + the live cameras.')
        seq = os.path.abspath(seq)
        intr_exo = _export_intrinsics(seq, 'exo')
        intr_hand = _export_intrinsics(seq, 'hand')
        print(f'[fusion] MODE=REPLAY  sequence={seq}'
              f'{("  namespace=/" + ns) if ns else "  (root)"}')

    params = LaunchConfiguration('params').perform(context)
    # Optionally OVERLAY the newest panel-saved tuning snapshot on top of
    # params.yaml (snapshots are partial → later files win, base stays intact).
    base_params = [params]
    if LaunchConfiguration('load_latest_tuning').perform(context) == 'true':
        latest = _latest_tuning(os.path.dirname(params))
        if latest:
            base_params.append(latest)
            print(f'[fusion] tuning overlay → {latest}')
        else:
            print('[fusion] load_latest_tuning: no params_*.yaml snapshot found')
    with open(params) as f:
        _y = yaml.safe_load(f) or {}
    dn = (_y.get('detection_node') or {}).get('ros__parameters') or {}
    # ArUco / hand-eye params live under `world_origin_node:`, which would NOT
    # apply to the renamed _exo/_hand nodes — re-pass them as an inline dict.
    wo = (_y.get('world_origin_node') or {}).get('ros__parameters') or {}
    # Cup model lives under point_cloud_node: — pass the geometry to
    # cup_fusion_node so the fused fit/frustum use the SAME Speed Stack cup.
    pn = (_y.get('point_cloud_node') or {}).get('ros__parameters') or {}
    cup_geom = {k: pn[k] for k in (
        'cup_top_diameter_m', 'cup_bottom_diameter_m', 'cup_height_m',
        'cup_fit_residual_max', 'cup_polygon_segments', 'cup_class_names',
        'box_standing_ratio', 'box_min_elongation') if k in pn}
    model_exo = dn.get('model_exo') or dn.get('model') or ''
    model_hand = dn.get('model_hand') or dn.get('model') or ''
    imgsz = int(LaunchConfiguration('imgsz').perform(context))

    EXO_F = 'exo_color_optical_frame'
    HAND_F = 'hand_color_optical_frame'

    # ---- trajectory source: REPLAY player vs LIVE joint bridge ----------
    if live:
        traj = [Node(
            package='depth_digital_twin', executable='robot_pose_bridge_node',
            name='robot_pose_bridge', output='screen',
            condition=IfCondition(LaunchConfiguration('with_pose_bridge')),
            parameters=[{
                'input_topic':
                    LaunchConfiguration('dsr_joint_topic').perform(context),
                'output_topic': '/joint_states',
            }])]
    else:
        traj = [Node(
            package='recode_sequence', executable='sequence_player_node',
            name='sequence_player_node', output='screen',
            parameters=[{
                'sequence_dir': seq,
                'exo_color_topic': '/camera_exo/color/image_raw',
                'exo_depth_topic':
                    '/camera_exo/aligned_depth_to_color/image_raw',
                'exo_info_topic': '/camera_exo/color/camera_info',
                'hand_color_topic': '/camera_hand/color/image_raw',
                'hand_depth_topic':
                    '/camera_hand/aligned_depth_to_color/image_raw',
                'hand_info_topic': '/camera_hand/color/camera_info',
                'exo_frame': EXO_F, 'hand_frame': HAND_F,
                'joint_states_topic': '/joint_states',
                'autostart':
                    LaunchConfiguration('autostart').perform(context) == 'true',
                'loop': LaunchConfiguration('loop').perform(context) == 'true',
            }]),
            Node(package='recode_sequence', executable='playback_control',
                 name='playback_control', output='screen')]

    # ---- m0609 URDF FK: world → base_link → … → link_6 (BOTH modes) -----
    robot_description = Command([
        PathJoinSubstitution([FindExecutable(name='xacro')]), ' ',
        PathJoinSubstitution([FindPackageShare('dsr_description2'),
                              'xacro', LaunchConfiguration('model')]),
        '.urdf.xacro',
        ' color:=', LaunchConfiguration('color'),
        ' name:=', LaunchConfiguration('name'),
        ' host:=127.0.0.1 port:=12345 mode:=virtual',
        ' rt_host:=127.0.0.1 update_rate:=100',
        ' model:=', LaunchConfiguration('model'),
    ])
    rsp = Node(
        package='robot_state_publisher', executable='robot_state_publisher',
        name='robot_state_publisher', output='screen',
        parameters=[{'robot_description': ParameterValue(
            robot_description, value_type=str)}])

    # ---- exo: ArUco world origin + detection + point cloud (producer) ----
    common_exo = [*base_params, {'intrinsics_path': intr_exo}]
    world_origin_exo = Node(
        package='depth_digital_twin', executable='world_origin_node',
        name='world_origin_node_exo', output='screen',
        parameters=common_exo + [wo, {
            'world_origin_mode': 'aruco',
            'color_topic': '/camera_exo/color/image_raw',
            'depth_topic': '/camera_exo/aligned_depth_to_color/image_raw',
            'camera_frame': EXO_F,
            'world_frame': 'world',
        }])
    common_hand = [*base_params, {'intrinsics_path': intr_hand}]
    world_origin_hand = Node(
        package='depth_digital_twin', executable='world_origin_node',
        name='world_origin_node_hand', output='screen',
        parameters=common_hand + [wo, {
            'world_origin_mode': 'handeye_aruco',
            'color_topic': '/camera_hand/color/image_raw',
            'camera_frame': HAND_F,
            'world_frame': 'world',
            'base_frame': 'base_link',
            'ee_frame': 'link_6',
            'aruco_timeout_then_floor': False,
        }])
    det_exo = Node(
        package='depth_digital_twin', executable='detection_node',
        name='detection_node_exo', output='screen',
        parameters=common_exo + [{
            'image_topic': '/camera_exo/color/image_raw',
            'detections_topic': '/digital_twin/detections_exo',
            'debug_topic': '/digital_twin/detection_debug_exo',
            'model': model_exo,
            'imgsz': imgsz,
        }])
    pc_exo = Node(
        package='depth_digital_twin', executable='point_cloud_node',
        name='point_cloud_node_exo', output='screen',
        parameters=common_exo + [{
            'rgb_topic': '/camera_exo/color/image_raw',
            'depth_topic': '/camera_exo/aligned_depth_to_color/image_raw',
            'detections_topic': '/digital_twin/detections_exo',
            'camera_frame': EXO_F,
            'world_frame': 'world',
            'role': 'producer',
            'camera_name': 'exo',
            'world_clouds_topic': '/digital_twin/cups_exo',
        }])
    det_hand = Node(
        package='depth_digital_twin', executable='detection_node',
        name='detection_node_hand', output='screen',
        parameters=common_hand + [{
            'image_topic': '/camera_hand/color/image_raw',
            'detections_topic': '/digital_twin/detections_hand',
            'debug_topic': '/digital_twin/detection_debug_hand',
            'model': model_hand,
            'imgsz': imgsz,
        }])
    pc_hand = Node(
        package='depth_digital_twin', executable='point_cloud_node',
        name='point_cloud_node_hand', output='screen',
        parameters=common_hand + [{
            'rgb_topic': '/camera_hand/color/image_raw',
            'depth_topic': '/camera_hand/aligned_depth_to_color/image_raw',
            'detections_topic': '/digital_twin/detections_hand',
            'camera_frame': HAND_F,
            'world_frame': 'world',
            'role': 'producer',
            'camera_name': 'hand',
            'world_clouds_topic': '/digital_twin/cups_hand',
            'hand_motion_gating': True,
            'joint_states_topic': '/joint_states',
            'aruco_overlay': False,
        }])

    # ---- fusion: associate + merge + fit + KF (+ scan&lock) → /boxes,/points
    fusion = Node(
        package='depth_digital_twin', executable='cup_fusion_node',
        name='cup_fusion_node', output='screen',
        parameters=[*base_params, dict({
            'exo_clouds_topic': '/digital_twin/cups_exo',
            'hand_clouds_topic': '/digital_twin/cups_hand',
            'boxes_topic': '/digital_twin/boxes',
            'points_topic': '/digital_twin/points',
            'world_frame': 'world',
            # scan-lock reads /joint_states (remapped under /<ns> in replay).
            'scan_joint_states_topic': '/joint_states',
        }, **cup_geom)])

    panel = Node(
        package='depth_digital_twin', executable='digital_twin_panel',
        name='digital_twin_panel', output='screen',
        parameters=[{
            'exo_redetect_srv': '/world_origin_node_exo/redetect',
            'hand_redetect_srv': '/world_origin_node_hand/redetect',
        }])

    rviz = Node(
        package='rviz2', executable='rviz2', name='rviz2', output='log',
        arguments=['-d', os.path.join(pkg, 'rviz', 'fusion.rviz')],
        condition=IfCondition(LaunchConfiguration('rviz')))

    entities = traj + [
        rsp, world_origin_exo, world_origin_hand,
        det_exo, pc_exo, det_hand, pc_hand,
        fusion, panel, rviz]

    # ns set ⇒ isolate the whole replay stack under /<ns> (coexist with live).
    return [_namespace_group(ns, entities)] if ns else entities


def generate_launch_description() -> LaunchDescription:
    pkg = get_package_share_directory('depth_digital_twin')
    return LaunchDescription([
        DeclareLaunchArgument(
            'ns', default_value='',
            description='Empty (default) = LIVE (or root replay if sequence '
                        'given). Set e.g. ns:=record to run a namespaced '
                        'replay alongside a live bringup.'),
        DeclareLaunchArgument(
            'sequence', default_value='',
            description='Recorded sequence dir — REQUIRED for replay, ignored '
                        'for live.'),
        DeclareLaunchArgument(
            'intr_exo',
            default_value=os.path.join(pkg, 'config', 'intrinsics.yaml'),
            description='LIVE exo intrinsics YAML (no meta.json in live mode).'),
        DeclareLaunchArgument(
            'intr_hand',
            default_value=os.path.join(pkg, 'config', 'intrinsics.yaml'),
            description='LIVE hand intrinsics YAML.'),
        DeclareLaunchArgument(
            'dsr_joint_topic', default_value='/dsr01/joint_states',
            description='LIVE source joints bridged → /joint_states.'),
        DeclareLaunchArgument('with_pose_bridge', default_value='true'),
        DeclareLaunchArgument('handeye_npy', default_value=_HANDEYE_DEFAULT),
        DeclareLaunchArgument('model', default_value='m0609'),
        DeclareLaunchArgument('color', default_value='white'),
        DeclareLaunchArgument('name', default_value='dsr01'),
        DeclareLaunchArgument('loop', default_value='false'),
        DeclareLaunchArgument('autostart', default_value='true'),
        DeclareLaunchArgument(
            'imgsz', default_value='640',
            description='YOLO inference size for BOTH detectors (dual = heavy; '
                        '640 default, set 1280 if GPU allows)'),
        DeclareLaunchArgument(
            'params',
            default_value=os.path.join(pkg, 'config', 'params.yaml')),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument(
            'load_latest_tuning', default_value='true',
            description='Overlay the newest panel-saved params_<ts>.yaml '
                        'snapshot on top of params.yaml at startup '
                        '(false = base params.yaml only).'),
        OpaqueFunction(function=_setup),
    ])
