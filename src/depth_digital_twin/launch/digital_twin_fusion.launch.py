"""Phase 2b — dual-camera (exo + hand) replay, world overlay + (later) fusion.

Brings up TWO parallel pipelines from one recorded sequence so the exo
and hand cup point clouds can be compared / fused in a common `world`
frame. Core nodes are UNCHANGED — only launch-level param overrides
(Approach A topology).

Frame / TF tree  (NO EE / get_current_posx — joints only)
---------------
  exo_color_optical_frame ──(world_origin_node, ArUco)──► world
  world ──(robot_state_publisher m0609, world_fixed=identity)──► base_link
  base_link ──(URDF FK from recorded /joint_states)──► … ──► link_6
  link_6 ──(handeye_tf_node, T_gripper2camera.npy, flange-ref)──► hand_color_optical_frame

  → point_cloud_node_exo  : world ← exo_color_optical_frame   (ArUco)
  → point_cloud_node_hand : world ← hand_color_optical_frame  (joint FK + hand-eye)

This launch is the RISK-FIRST verification step: RViz shows
/digital_twin/points_exo and /digital_twin/points_hand together in
`world`. If the same physical cup overlaps from both clouds, the
hand→world chain (npy + world_marker_offset) is sound and the fusion
node (cup_fusion_node, added next) can be tuned. If they are offset,
the calibration must be fixed before fusion is meaningful.

Prerequisites (THREE workspaces sourced — dsr_description2 provides
the m0609 URDF used for the joint FK):
  source ~/ros2_ws/install/setup.bash
  source ~/Projects/ros2-recode-sequence/install/setup.bash
  source ~/Projects/ros2-depth-point-cloude/install/setup.bash

⚠ Do NOT run the LIVE robot stack (dsr_bringup2) at the same time:
  it publishes the same world→base_link→…→link_6 TF from the LIVE
  joints, which would fight this launch's robot_state_publisher fed
  by the RECORDED /joint_states. Shut dsr_bringup2 down first.

Usage:
  ros2 launch depth_digital_twin digital_twin_fusion.launch.py \\
      sequence:=/home/eunwoosong/Projects/record_sequence/0010

Args:
  sequence    : absolute path to a recorded sequence dir (REQUIRED)
  handeye_npy : hand-eye T_gripper2camera.npy (default: recode_sequence cfg)
  loop        : replay loop (default false)
  autostart   : start playing immediately (default true)
  params      : pipeline params.yaml (default: package config)
  rviz        : launch RViz2 overlay (default true)
"""
import json
import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import (Command, FindExecutable,
                                  LaunchConfiguration, PathJoinSubstitution)
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

_HANDEYE_DEFAULT = ('/home/eunwoosong/Projects/ros2-recode-sequence/'
                    'src/recode_sequence/config/T_gripper2camera.npy')


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
    seq = LaunchConfiguration('sequence').perform(context)
    if not seq or not os.path.isdir(seq):
        raise RuntimeError(
            'sequence:=/abs/path/to/record_sequence/NNNN is required')
    try:
        get_package_share_directory('dsr_description2')
    except Exception:
        raise RuntimeError(
            "dsr_description2 not found — the m0609 URDF (joint FK for "
            "the hand camera) needs the doosan workspace:\n"
            "    source ~/ros2_ws/install/setup.bash")
    seq = os.path.abspath(seq)
    intr_exo = _export_intrinsics(seq, 'exo')
    intr_hand = _export_intrinsics(seq, 'hand')

    pkg = get_package_share_directory('depth_digital_twin')
    params = LaunchConfiguration('params').perform(context)
    with open(params) as f:
        dn = ((yaml.safe_load(f) or {}).get('detection_node') or {}).get(
            'ros__parameters') or {}
    model_exo = dn.get('model_exo') or dn.get('model') or ''
    model_hand = dn.get('model_hand') or dn.get('model') or ''
    # Two YOLO-seg + two point_cloud nodes are heavy; the dual case
    # defaults to a smaller inference size (override with imgsz:=).
    imgsz = int(LaunchConfiguration('imgsz').perform(context))

    EXO_F = 'exo_color_optical_frame'
    HAND_F = 'hand_color_optical_frame'

    # ---- sequence player: both cameras + recorded /joint_states --------
    # No EE / get_current_posx anywhere: the hand camera pose comes from
    # recorded joints → URDF FK (robot_state_publisher below).
    player = Node(
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
            'exo_frame': EXO_F,
            'hand_frame': HAND_F,
            'joint_states_topic': '/joint_states',
            'autostart':
                LaunchConfiguration('autostart').perform(context) == 'true',
            'loop': LaunchConfiguration('loop').perform(context) == 'true',
        }])

    # ---- m0609 URDF FK: world → base_link → … → link_6 -----------------
    # world_fixed (world→base_link) is identity, so the URDF `world`
    # coincides with world_origin_node's ArUco `world` (= robot base).
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

    # ---- hand-eye static TF: link_6 (flange) → hand_color_optical ------
    handeye = Node(
        package='recode_sequence', executable='handeye_tf_node',
        name='handeye_tf_node', output='screen',
        parameters=[{
            'handeye_npy': LaunchConfiguration('handeye_npy').perform(
                context),
            'parent_frame': 'link_6',
            'child_frame': HAND_F,
            'units_scale': 0.001,
        }])

    # ---- exo: ArUco world origin + detection + point cloud ------------
    common_exo = [params, {'intrinsics_path': intr_exo}]
    world_origin_exo = Node(
        package='depth_digital_twin', executable='world_origin_node',
        name='world_origin_node', output='screen',
        parameters=common_exo + [{
            'color_topic': '/camera_exo/color/image_raw',
            'depth_topic': '/camera_exo/aligned_depth_to_color/image_raw',
            'camera_frame': EXO_F,
            'world_frame': 'world',
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
            'points_topic': '/digital_twin/points_exo',
            'boxes_topic': '/digital_twin/boxes_exo',
            'box_debug_topic': '/digital_twin/box_debug_exo',
            'depth_debug_topic': '/digital_twin/depth_debug_exo',
        }])

    # ---- hand: NO ArUco (world via EE+hand-eye TF) + det + cloud ------
    common_hand = [params, {'intrinsics_path': intr_hand}]
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
            'points_topic': '/digital_twin/points_hand',
            'boxes_topic': '/digital_twin/boxes_hand',
            'box_debug_topic': '/digital_twin/box_debug_hand',
            'depth_debug_topic': '/digital_twin/depth_debug_hand',
            'aruco_overlay': False,
        }])

    rviz = Node(
        package='rviz2', executable='rviz2', name='rviz2', output='log',
        arguments=['-d', os.path.join(pkg, 'rviz', 'fusion.rviz')],
        condition=IfCondition(LaunchConfiguration('rviz')))
    playback_ctrl = Node(
        package='recode_sequence', executable='playback_control',
        name='playback_control', output='screen')

    return [player, rsp, handeye,
            world_origin_exo, det_exo, pc_exo,
            det_hand, pc_hand,
            rviz, playback_ctrl]


def generate_launch_description() -> LaunchDescription:
    pkg = get_package_share_directory('depth_digital_twin')
    return LaunchDescription([
        DeclareLaunchArgument('sequence', default_value=''),
        DeclareLaunchArgument('handeye_npy',
                              default_value=_HANDEYE_DEFAULT),
        DeclareLaunchArgument('model', default_value='m0609'),
        DeclareLaunchArgument('color', default_value='white'),
        DeclareLaunchArgument('name', default_value='dsr01'),
        DeclareLaunchArgument('loop', default_value='false'),
        DeclareLaunchArgument('autostart', default_value='true'),
        DeclareLaunchArgument(
            'imgsz', default_value='640',
            description='YOLO inference size for BOTH detectors (dual = '
                        'heavy; 640 default, set 1280 if GPU allows)'),
        DeclareLaunchArgument(
            'params',
            default_value=os.path.join(pkg, 'config', 'params.yaml')),
        DeclareLaunchArgument('rviz', default_value='true'),
        OpaqueFunction(function=_setup),
    ])
