"""Phase 2a — run the cup-detection pipeline from a RECORDED sequence.

Choose which recorded camera feeds the pipeline with `view:=exo|hand`.

  view:=exo  (default)
    ArUco-based world calibration — same as live-camera mode.
    Frame: camera_color_optical_frame → world (via world_origin_node).

  view:=hand
    FK-based world transform — no ArUco needed (hand camera is wrist-mounted
    and does not see the marker).  Requires dsr_description2:
      source ~/ros2_ws/install/setup.bash
    TF chain:
      world ──(RSP m0609, world_fixed=identity)──► base_link
      base_link ──(URDF FK from /joint_states)──► link_6
      link_6 ──(handeye_tf_node, T_gripper2camera.npy)──► hand_color_optical_frame

Prerequisites:
  source ~/Projects/ros2-recode-sequence/install/setup.bash
  source ~/Projects/ros2-depth-point-cloude/install/setup.bash
  source ~/ros2_ws/install/setup.bash   # view:=hand only (dsr_description2)

Usage:
  ros2 launch depth_digital_twin digital_twin_sequence.launch.py \\
      sequence:=/home/eunwoosong/Projects/record_sequence/0010
  ros2 launch depth_digital_twin digital_twin_sequence.launch.py \\
      sequence:=/home/eunwoosong/Projects/record_sequence/0010 view:=hand

Args:
  sequence    : absolute path to a recorded sequence dir (REQUIRED)
  view        : exo | hand — camera into pipeline (default exo)
  yolo_model  : override detection_node.model (empty = params.yaml selection)
  handeye_npy : T_gripper2camera.npy (view:=hand only)
  model       : Doosan URDF model name (default m0609, view:=hand only)
  color       : URDF color (default white, view:=hand only)
  name        : robot instance name (default dsr01, view:=hand only)
  loop        : replay loop (default false)
  autostart   : start playing immediately (default true)
  params      : pipeline params.yaml (default: package config)
  rviz        : launch RViz2 (default true)
"""
import json
import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            OpaqueFunction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (Command, FindExecutable, LaunchConfiguration,
                                  PathJoinSubstitution)
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

_HANDEYE_DEFAULT = ('/home/eunwoosong/Projects/ros2-recode-sequence/'
                    'src/recode_sequence/config/T_gripper2camera.npy')

_PIPE_COLOR = '/camera/camera/color/image_raw'
_PIPE_DEPTH = '/camera/camera/aligned_depth_to_color/image_raw'
_PIPE_INFO = '/camera/camera/color/camera_info'
_IDLE = {
    'exo': {'c': '/exo/exo/color/image_raw',
            'd': '/exo/exo/aligned_depth_to_color/image_raw',
            'i': '/exo/exo/color/camera_info', 'f': 'exo_color_optical_frame'},
    'hand': {'c': '/hand/hand/color/image_raw',
             'd': '/hand/hand/aligned_depth_to_color/image_raw',
             'i': '/hand/hand/color/camera_info',
             'f': 'hand_color_optical_frame'},
}


def _export_intrinsics(seq_dir: str, view: str) -> str:
    with open(os.path.join(seq_dir, 'meta.json')) as f:
        meta = json.load(f)
    cam = (meta.get('cameras') or {}).get(view) or {}
    K = cam.get('K') or []
    if len(K) != 9 or not cam.get('width'):
        raise RuntimeError(
            f'sequence {seq_dir} has no {view} camera intrinsics in '
            'meta.json (was that camera_info topic recorded?)')
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
    path = os.path.join(seq_dir, f'{view}_intrinsics.yaml')
    with open(path, 'w') as f:
        yaml.safe_dump(out, f, sort_keys=False)
    return path


def _setup(context, *_, **__):
    seq = LaunchConfiguration('sequence').perform(context)
    view = LaunchConfiguration('view').perform(context).strip().lower()
    if view not in ('exo', 'hand'):
        raise RuntimeError(f"view must be 'exo' or 'hand' (got {view!r})")
    if not seq or not os.path.isdir(seq):
        raise RuntimeError(
            'sequence:=/abs/path/to/record_sequence/NNNN is required '
            f'(got {seq!r})')
    seq = os.path.abspath(seq)
    intr = _export_intrinsics(seq, view)
    print(f'[seq] view={view}  intrinsics → {intr}')

    pkg = get_package_share_directory('depth_digital_twin')
    params = LaunchConfiguration('params').perform(context)
    with open(params) as f:
        dn = ((yaml.safe_load(f) or {}).get('detection_node') or {}).get(
            'ros__parameters') or {}
    explicit = LaunchConfiguration('yolo_model').perform(context).strip()
    model = explicit or dn.get(f'model_{view}') or dn.get('model') or ''
    print(f'[seq] view={view}  yolo model → {model or "(params.yaml default)"}')

    autostart = LaunchConfiguration('autostart').perform(context) == 'true'
    loop_flag = LaunchConfiguration('loop').perform(context) == 'true'
    common_params = [params, {'intrinsics_path': intr}]

    if view == 'exo':
        return _setup_exo(context, seq, intr, pkg, params, model,
                          autostart, loop_flag)
    else:
        return _setup_hand(context, seq, intr, pkg, params, model,
                           common_params, autostart, loop_flag)


def _setup_exo(context, seq, intr, pkg, params, model, autostart, loop_flag):
    """Exo view: ArUco world calibration via world_origin_node (existing approach)."""
    player = Node(
        package='recode_sequence', executable='sequence_player_node',
        name='sequence_player_node', output='screen',
        parameters=[{
            'sequence_dir': seq,
            'exo_color_topic': _PIPE_COLOR,
            'exo_depth_topic': _PIPE_DEPTH,
            'exo_info_topic': _PIPE_INFO,
            'exo_frame': 'camera_color_optical_frame',
            'hand_color_topic': _IDLE['hand']['c'],
            'hand_depth_topic': _IDLE['hand']['d'],
            'hand_info_topic': _IDLE['hand']['i'],
            'hand_frame': _IDLE['hand']['f'],
            'autostart': autostart,
            'loop': loop_flag,
        }])

    pipeline = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg, 'launch', 'digital_twin.launch.py')),
        launch_arguments={
            'intrinsics': intr,
            'params': params,
            'yolo_model': model,
            'rviz': LaunchConfiguration('rviz'),
            'control_panel': 'true',
        }.items())

    playback_ctrl = Node(
        package='recode_sequence', executable='playback_control',
        name='playback_control', output='screen')

    return [player, pipeline, playback_ctrl]


def _setup_hand(context, seq, intr, pkg, params, model, common_params,
                autostart, loop_flag):
    """Hand view: FK-based world transform (joint_states → RSP → handeye_tf)."""
    try:
        get_package_share_directory('dsr_description2')
    except Exception:
        raise RuntimeError(
            "dsr_description2 not found — the m0609 URDF (joint FK for "
            "the hand camera) needs the doosan workspace:\n"
            "    source ~/ros2_ws/install/setup.bash")

    player = Node(
        package='recode_sequence', executable='sequence_player_node',
        name='sequence_player_node', output='screen',
        parameters=[{
            'sequence_dir': seq,
            'hand_color_topic': _PIPE_COLOR,
            'hand_depth_topic': _PIPE_DEPTH,
            'hand_info_topic': _PIPE_INFO,
            'hand_frame': 'hand_color_optical_frame',
            'exo_color_topic': _IDLE['exo']['c'],
            'exo_depth_topic': _IDLE['exo']['d'],
            'exo_info_topic': _IDLE['exo']['i'],
            'exo_frame': _IDLE['exo']['f'],
            'joint_states_topic': '/joint_states',
            'autostart': autostart,
            'loop': loop_flag,
        }])

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

    handeye = Node(
        package='recode_sequence', executable='handeye_tf_node',
        name='handeye_tf_node', output='screen',
        parameters=[{
            'handeye_npy': LaunchConfiguration('handeye_npy').perform(context),
            'parent_frame': 'link_6',
            'child_frame': 'hand_color_optical_frame',
            'units_scale': 0.001,
        }])

    det_params = common_params + ([{'model': model}] if model else [])
    detection = Node(
        package='depth_digital_twin', executable='detection_node',
        name='detection_node', output='screen',
        parameters=det_params)

    point_cloud = Node(
        package='depth_digital_twin', executable='point_cloud_node',
        name='point_cloud_node', output='screen',
        parameters=common_params + [{
            'rgb_topic': _PIPE_COLOR,
            'depth_topic': _PIPE_DEPTH,
            'camera_frame': 'hand_color_optical_frame',
            'world_frame': 'world',
            'aruco_overlay': False,
        }])

    rviz = Node(
        package='rviz2', executable='rviz2', name='rviz2', output='log',
        arguments=['-d', os.path.join(pkg, 'rviz', 'digital_twin.rviz')],
        condition=IfCondition(LaunchConfiguration('rviz')))

    playback_ctrl = Node(
        package='recode_sequence', executable='playback_control',
        name='playback_control', output='screen')

    return [player, rsp, handeye, detection, point_cloud, rviz, playback_ctrl]


def generate_launch_description() -> LaunchDescription:
    pkg = get_package_share_directory('depth_digital_twin')
    return LaunchDescription([
        DeclareLaunchArgument('sequence', default_value='',
                              description='Absolute path to a sequence dir'),
        DeclareLaunchArgument('view', default_value='exo',
                              description='exo | hand — camera into pipeline'),
        DeclareLaunchArgument('yolo_model', default_value='',
                              description='Override detection_node.model'),
        DeclareLaunchArgument('handeye_npy', default_value=_HANDEYE_DEFAULT,
                              description='T_gripper2camera.npy (view:=hand)'),
        DeclareLaunchArgument('model', default_value='m0609',
                              description='Doosan URDF model (view:=hand)'),
        DeclareLaunchArgument('color', default_value='white',
                              description='URDF color (view:=hand)'),
        DeclareLaunchArgument('name', default_value='dsr01',
                              description='Robot instance name (view:=hand)'),
        DeclareLaunchArgument('loop', default_value='false'),
        DeclareLaunchArgument('autostart', default_value='true'),
        DeclareLaunchArgument(
            'params',
            default_value=os.path.join(pkg, 'config', 'params.yaml')),
        DeclareLaunchArgument('rviz', default_value='true'),
        OpaqueFunction(function=_setup),
    ])
