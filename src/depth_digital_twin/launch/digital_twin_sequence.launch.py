"""Phase 2a — run the cup-detection pipeline from a RECORDED sequence.

Replays one camera of a recode_sequence recording into the existing
single-camera digital-twin pipeline with ZERO changes to the core nodes.
Choose which recorded camera feeds the pipeline with `view:=exo|hand`.

  recode_sequence/sequence_player_node
      ├─ <view> colour → /camera/camera/color/image_raw
      ├─ <view> depth  → /camera/camera/aligned_depth_to_color/image_raw
      └─ <view> info   → /camera/camera/color/camera_info
  (frame_id = camera_color_optical_frame, matching the pipeline default)

  world_origin_node  detects the ArUco marker in the replayed frames
                     → static TF camera→world, exactly as live.
  detection_node + point_cloud_node  run unchanged → cup boxes / cloud.

The chosen camera's intrinsics come from the sequence meta.json and are
written to <sequence>/<view>_intrinsics.yaml (load_intrinsics format),
then passed as the pipeline intrinsics.

NOTE: ROS 2 launch uses `key:=value`, not `--hand`. Use `view:=hand`.
NOTE: the hand camera is wrist-mounted; ArUco-based world calibration
      only succeeds if the marker is visible in the recorded hand frames
      (true exo↔hand world fusion via EE + hand-eye is Phase 2b).
The YOLO weight is chosen per view FROM params.yaml
(detection_node.model_exo / model_hand) — set a hand-specialised weight
in params.yaml's `model_hand`. An explicit `yolo_model:=/path.pt`
argument, if given, overrides that selection.

Prerequisites (both workspaces sourced):
  source ~/Projects/ros2-recode-sequence/install/setup.bash
  source ~/Projects/ros2-depth-point-cloude/install/setup.bash

Usage:
  ros2 launch depth_digital_twin digital_twin_sequence.launch.py \\
      sequence:=/home/eunwoosong/Projects/record_sequence/0005          # exo
  ros2 launch depth_digital_twin digital_twin_sequence.launch.py \\
      sequence:=/home/eunwoosong/Projects/record_sequence/0005 view:=hand

Args:
  sequence   : absolute path to a recorded sequence dir (REQUIRED)
  view       : exo | hand — which recorded camera feeds the pipeline
               (default exo)
  yolo_model : optional explicit weight; overrides params.yaml
               detection_node.model_<view> selection
  loop       : replay loop (default false — stop at end)
  autostart  : start playing immediately (default true)
  params     : pipeline params.yaml (default: package config)
  rviz       : launch RViz2 (default true)
"""
import json
import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            OpaqueFunction)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _export_intrinsics(seq_dir: str, view: str) -> str:
    """meta.json <view> camera → <seq>/<view>_intrinsics.yaml."""
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


# Pipeline-side topics the chosen camera must feed.
_PIPE_COLOR = '/camera/camera/color/image_raw'
_PIPE_DEPTH = '/camera/camera/aligned_depth_to_color/image_raw'
_PIPE_INFO = '/camera/camera/color/camera_info'
# Player defaults for the camera that is NOT selected (left idle).
_IDLE = {
    'exo': {'c': '/exo/exo/color/image_raw',
            'd': '/exo/exo/aligned_depth_to_color/image_raw',
            'i': '/exo/exo/color/camera_info', 'f': 'exo_color_optical_frame'},
    'hand': {'c': '/hand/hand/color/image_raw',
             'd': '/hand/hand/aligned_depth_to_color/image_raw',
             'i': '/hand/hand/color/camera_info',
             'f': 'hand_color_optical_frame'},
}


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

    # Route the SELECTED view to the pipeline topics + the pipeline's
    # default camera frame; leave the other view on its idle defaults.
    other = 'hand' if view == 'exo' else 'exo'
    player_params = {
        'sequence_dir': seq,
        f'{view}_color_topic': _PIPE_COLOR,
        f'{view}_depth_topic': _PIPE_DEPTH,
        f'{view}_info_topic': _PIPE_INFO,
        f'{view}_frame': 'camera_color_optical_frame',
        f'{other}_color_topic': _IDLE[other]['c'],
        f'{other}_depth_topic': _IDLE[other]['d'],
        f'{other}_info_topic': _IDLE[other]['i'],
        f'{other}_frame': _IDLE[other]['f'],
        'base_frame': 'world',
        'autostart':
            LaunchConfiguration('autostart').perform(context) == 'true',
        'loop': LaunchConfiguration('loop').perform(context) == 'true',
    }
    player = Node(
        package='recode_sequence', executable='sequence_player_node',
        name='sequence_player_node', output='screen',
        parameters=[player_params])

    # Pick the per-view YOLO weight FROM params.yaml
    # (detection_node.model_<view>); no temp overlay file. An explicit
    # yolo_model:= arg, if given, takes precedence.
    with open(params) as f:
        dn = ((yaml.safe_load(f) or {}).get('detection_node') or {}).get(
            'ros__parameters') or {}
    explicit = LaunchConfiguration('yolo_model').perform(context).strip()
    model = explicit or dn.get(f'model_{view}') or dn.get('model') or ''
    print(f'[seq] view={view}  yolo model → {model or "(params.yaml default)"}')

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


def generate_launch_description() -> LaunchDescription:
    pkg = get_package_share_directory('depth_digital_twin')
    return LaunchDescription([
        DeclareLaunchArgument('sequence', default_value='',
                              description='Absolute path to a sequence dir'),
        DeclareLaunchArgument('view', default_value='exo',
                              description='exo | hand — camera into pipeline'),
        DeclareLaunchArgument('yolo_model', default_value='',
                              description='Override detection_node.model'),
        DeclareLaunchArgument('loop', default_value='false'),
        DeclareLaunchArgument('autostart', default_value='true'),
        DeclareLaunchArgument(
            'params',
            default_value=os.path.join(pkg, 'config', 'params.yaml')),
        DeclareLaunchArgument('rviz', default_value='true'),
        OpaqueFunction(function=_setup),
    ])
