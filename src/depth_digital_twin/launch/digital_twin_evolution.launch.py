"""digital_twin_evolution — replay the *evolution* of the depth digital twin as
five switchable detection methods, for recording GIFs in RViz.

Plays a recorded sequence (default seq 0010) and feeds exo + hand RGB-D + YOLO
into `evolution_demo_node`, which renders ONE method at a time. A Tk window
(`evolution_control`) cycles the active method with "Next method ▶"; record each
method as a separate GIF, 1 → 5, to tell the story:

  1 Raw point-cloud OBB · 2 cloud cone fit · 3 exo+hand fusion (hand FK) ·
  4 RGB rim fit · 5 erode + shape-prior AABB + target select.

Playback first LOOPS the early warm-up frames so `world_origin_node_exo` can
lock the ArUco world (= robot base) TF; the demo node then triggers the player
to jump to [range_start, range_end] (default 1333..1666) and loop there.

Frame / TF tree (same as digital_twin_fusion):
  exo_color_optical_frame ──(world_origin_node_exo, ArUco)──► world
  world ──(robot_state_publisher m0609)──► base_link ──(URDF FK)──► … ──► link_6
  link_6 ──(world_origin_node_hand, fixed hand-eye)──► hand_color_optical_frame

Prerequisites (three workspaces — dsr_description2 provides the m0609 URDF):
  source ~/ros2_ws/install/setup.bash
  source ~/Projects/ros2-recode-sequence/install/setup.bash
  source ~/Projects/ros2-depth-point-cloude/install/setup.bash

Usage:
  ros2 launch depth_digital_twin digital_twin_evolution.launch.py \\
      sequence:=/home/eunwoo/Projects/cup_stack/seq_record/0010
"""
import json
import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, OpaqueFunction,
                            SetEnvironmentVariable)
from launch.conditions import IfCondition
from launch.substitutions import (Command, FindExecutable, LaunchConfiguration,
                                  PathJoinSubstitution)
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

EXO_NS = '/camera_exo'
HAND_NS = '/camera_hand'
EXO_F = 'exo_color_optical_frame'
HAND_F = 'hand_color_optical_frame'


def _export_intrinsics(seq_dir: str, view: str) -> str:
    with open(os.path.join(seq_dir, 'meta.json')) as f:
        meta = json.load(f)
    cam = (meta.get('cameras') or {}).get(view) or {}
    K = cam.get('K') or []
    if len(K) != 9 or not cam.get('width'):
        raise RuntimeError(f'sequence {seq_dir}: no {view} intrinsics in meta.json')
    dist = cam.get('dist') or [0.0] * 5
    out = {
        'image_width': int(cam['width']), 'image_height': int(cam['height']),
        'camera_matrix': {'rows': 3, 'cols': 3, 'data': [float(v) for v in K]},
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
    seq = LaunchConfiguration('sequence').perform(context)
    if not seq or not os.path.isdir(seq):
        raise RuntimeError(
            'sequence:=/abs/path/to/record_sequence/NNNN is required '
            f'(got {seq!r})')
    seq = os.path.abspath(seq)

    try:
        get_package_share_directory('dsr_description2')
    except Exception:
        raise RuntimeError(
            "dsr_description2 not found — the m0609 URDF (joint FK for the "
            "hand camera) needs the doosan workspace:\n"
            "    source ~/ros2_ws/install/setup.bash")

    intr_exo = _export_intrinsics(seq, 'exo')
    intr_hand = _export_intrinsics(seq, 'hand')
    params = LaunchConfiguration('params').perform(context)
    with open(params) as f:
        y = yaml.safe_load(f) or {}
    dn = (y.get('detection_node') or {}).get('ros__parameters') or {}
    wo = (y.get('world_origin_node') or {}).get('ros__parameters') or {}
    pn = (y.get('point_cloud_node') or {}).get('ros__parameters') or {}
    # ArUco must succeed (cups → robot base frame); never fall back to floor.
    wo = dict(wo, aruco_timeout_then_floor=False,
              world_marker_samples_required=int(
                  LaunchConfiguration('aruco_samples').perform(context)))

    model = (LaunchConfiguration('model_yolo').perform(context).strip()
             or dn.get('model') or '')
    det_common = {k: dn[k] for k in ('target_classes', 'confidence', 'device')
                  if k in dn}
    imgsz = int(LaunchConfiguration('imgsz').perform(context))
    range_start = int(LaunchConfiguration('range_start').perform(context))
    range_end = int(LaunchConfiguration('range_end').perform(context))
    warmup_end = int(LaunchConfiguration('warmup_end').perform(context))
    autostart = LaunchConfiguration('autostart').perform(context) == 'true'

    base = [{'release_mode': False}]
    common_exo = base + [{'intrinsics_path': intr_exo}]
    common_hand = base + [{'intrinsics_path': intr_hand}]

    player = Node(
        package='recode_sequence', executable='sequence_player_node',
        name='sequence_player_node', output='screen',
        parameters=[{
            'sequence_dir': seq,
            'exo_color_topic': f'{EXO_NS}/color/image_raw',
            'exo_depth_topic': f'{EXO_NS}/aligned_depth_to_color/image_raw',
            'exo_info_topic': f'{EXO_NS}/color/camera_info',
            'hand_color_topic': f'{HAND_NS}/color/image_raw',
            'hand_depth_topic': f'{HAND_NS}/aligned_depth_to_color/image_raw',
            'hand_info_topic': f'{HAND_NS}/color/camera_info',
            'exo_frame': EXO_F, 'hand_frame': HAND_F,
            'joint_states_topic': '/joint_states',
            'autostart': autostart, 'loop': False,
            'range_start': range_start, 'range_end': range_end,
            'warmup_end': warmup_end,
        }])
    playback_ctrl = Node(
        package='recode_sequence', executable='playback_control',
        name='playback_control', output='screen')

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
        condition=IfCondition(LaunchConfiguration('with_rsp')),
        parameters=[{'robot_description': ParameterValue(
            robot_description, value_type=str)}])

    world_origin_exo = Node(
        package='depth_digital_twin', executable='world_origin_node',
        name='world_origin_node_exo', output='screen',
        parameters=common_exo + [wo, {
            'world_origin_mode': 'aruco',
            'color_topic': f'{EXO_NS}/color/image_raw',
            'depth_topic': f'{EXO_NS}/aligned_depth_to_color/image_raw',
            'camera_frame': EXO_F, 'world_frame': 'world',
        }])
    # handeye_use_param_offset (params.yaml) → publishes link_6→hand_camera
    # immediately; world←base_link comes from the URDF + joint FK.
    world_origin_hand = Node(
        package='depth_digital_twin', executable='world_origin_node',
        name='world_origin_node_hand', output='screen',
        parameters=common_hand + [wo, {
            'world_origin_mode': 'handeye_aruco',
            'color_topic': f'{HAND_NS}/color/image_raw',
            'camera_frame': HAND_F, 'world_frame': 'world',
            'base_frame': 'base_link', 'ee_frame': 'link_6',
            'aruco_timeout_then_floor': False,
        }])

    det_exo = Node(
        package='depth_digital_twin', executable='detection_node',
        name='detection_node_exo', output='screen',
        parameters=base + [det_common, {
            'image_topic': f'{EXO_NS}/color/image_raw',
            'detections_topic': '/digital_twin/detections_exo',
            'debug_topic': '/digital_twin/detection_debug_exo',
            'model': model, 'imgsz': imgsz,
        }])
    det_hand = Node(
        package='depth_digital_twin', executable='detection_node',
        name='detection_node_hand', output='screen',
        parameters=base + [det_common, {
            'image_topic': f'{HAND_NS}/color/image_raw',
            'detections_topic': '/digital_twin/detections_hand',
            'debug_topic': '/digital_twin/detection_debug_hand',
            'model': model, 'imgsz': imgsz,
        }])

    geom = {k: pn[k] for k in (
        'cup_top_diameter_m', 'cup_bottom_diameter_m', 'cup_height_m',
        'cup_polygon_segments', 'mask_erode_px', 'depth_gradient_max_m',
        'box_outlier_mad_k', 'z_min', 'z_max') if k in pn}
    demo = Node(
        package='depth_digital_twin', executable='evolution_demo_node',
        name='evolution_demo_node', output='screen',
        parameters=[dict({
            'intrinsics_exo_path': intr_exo,
            'intrinsics_hand_path': intr_hand,
            'world_frame': 'world', 'exo_frame': EXO_F, 'hand_frame': HAND_F,
            'exo_rgb_topic': f'{EXO_NS}/color/image_raw',
            'exo_depth_topic': f'{EXO_NS}/aligned_depth_to_color/image_raw',
            'exo_detections_topic': '/digital_twin/detections_exo',
            'hand_rgb_topic': f'{HAND_NS}/color/image_raw',
            'hand_depth_topic': f'{HAND_NS}/aligned_depth_to_color/image_raw',
            'hand_detections_topic': '/digital_twin/detections_hand',
            'player_begin_range_srv': '/sequence_player_node/begin_range',
            'start_method': int(
                LaunchConfiguration('start_method').perform(context)),
            'render_period_s': float(
                LaunchConfiguration('render_period_s').perform(context)),
        }, **geom)])
    control = Node(
        package='depth_digital_twin', executable='evolution_control',
        name='evolution_control', output='screen',
        parameters=[{
            'demo_node': '/evolution_demo_node',
            'exo_redetect_srv': '/world_origin_node_exo/redetect',
        }])

    rviz = Node(
        package='rviz2', executable='rviz2', name='rviz2', output='log',
        arguments=['-d', os.path.join(pkg, 'rviz', 'evolution.rviz')],
        condition=IfCondition(LaunchConfiguration('rviz')))

    return [player, playback_ctrl, rsp, world_origin_exo, world_origin_hand,
            det_exo, det_hand, demo, control, rviz]


def generate_launch_description() -> LaunchDescription:
    pkg = get_package_share_directory('depth_digital_twin')
    return LaunchDescription([
        # Declare BEFORE SetEnvironmentVariable: launch actions run top-to-bottom
        # and the substitution resolves when SetEnvironmentVariable executes.
        DeclareLaunchArgument('domain_id', default_value='1',
                              description='ROS_DOMAIN_ID (collision avoidance).'),
        SetEnvironmentVariable('ROS_DOMAIN_ID',
                               LaunchConfiguration('domain_id')),
        DeclareLaunchArgument('sequence', default_value='',
                              description='Absolute path to a sequence dir.'),
        DeclareLaunchArgument('range_start', default_value='1333'),
        DeclareLaunchArgument('range_end', default_value='1666'),
        DeclareLaunchArgument(
            'warmup_end', default_value='200',
            description='Loop [0, warmup_end] until the exo ArUco world TF '
                        'locks, then jump to [range_start, range_end].'),
        DeclareLaunchArgument('aruco_samples', default_value='15',
                              description='ArUco frames to average before lock '
                                          '(lower = faster warm-up lock).'),
        DeclareLaunchArgument('start_method', default_value='1'),
        DeclareLaunchArgument(
            'model_yolo', default_value='',
            description='YOLO weights (empty = params.yaml detection_node.model '
                        '= 0610-2.pt). Used for BOTH exo and hand.'),
        DeclareLaunchArgument(
            'imgsz', default_value='1280',
            description='YOLO inference size for both detectors (dual = heavy; '
                        'lower to 640 if the GPU lags).'),
        DeclareLaunchArgument(
            'render_period_s', default_value='0.0',
            description='Min seconds between heavy renders (0 = every frame).'),
        DeclareLaunchArgument('model', default_value='m0609'),
        DeclareLaunchArgument('color', default_value='white'),
        DeclareLaunchArgument('name', default_value='dsr01'),
        DeclareLaunchArgument('with_rsp', default_value='true'),
        DeclareLaunchArgument('autostart', default_value='true'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument(
            'params', default_value=os.path.join(pkg, 'config', 'params.yaml')),
        OpaqueFunction(function=_setup),
    ])
