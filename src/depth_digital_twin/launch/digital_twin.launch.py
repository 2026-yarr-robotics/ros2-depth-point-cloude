"""Bring up the depth digital twin pipeline (exo camera) + RViz2.

Steps:
  1. Start RealSense camera (separate terminal):
       ros2 launch realsense2_camera rs_align_depth_launch.py \
           depth_module.depth_profile:=640x480x30 \
           rgb_camera.color_profile:=640x480x30 \
           initial_reset:=true align_depth.enable:=true

  2. Place the ArUco marker (ID 0, DICT_4X4_50) so it is visible to the camera.

  3. Launch:
       ros2 launch depth_digital_twin digital_twin.launch.py

  world_origin_node detects the marker, averages 30 samples, publishes a static
  TF camera→world (world = robot base). Falls back to depth plane-fit if the
  marker is not detected within 15 s.

  Tune world_marker_rot_z_deg in params.yaml until world +X/+Y/+Z in RViz
  match the robot base axes.

Args:
  intrinsics      : path to intrinsics.yaml (default: package config/)
  params          : path to params.yaml     (default: package config/)
  rviz            : true|false              (default: true)
  rviz_config     : path to .rviz file      (default: package rviz/)
  control_panel   : true|false — show Reset/Redetect popup window (default: true)
"""
import os
import tempfile

import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _write_node_param_override(node_name: str, params: dict) -> str:
    """Temp YAML pinning per-node parameters; needed because a wildcard inline
    dict in `parameters=` cannot override a node-specific entry from the user
    yaml (rcl precedence). A second --params-file with matching node
    namespace IS applied (in order) and wins."""
    fd, path = tempfile.mkstemp(
        suffix=f'_{node_name}_override.yaml', text=True)
    with os.fdopen(fd, 'w') as f:
        yaml.safe_dump({node_name: {'ros__parameters': params}}, f)
    return path


def _make_nodes(context, *args, **kwargs):
    intrinsics = LaunchConfiguration('intrinsics').perform(context)
    params = LaunchConfiguration('params').perform(context)
    rviz_cfg = LaunchConfiguration('rviz_config').perform(context)
    camera_ns = LaunchConfiguration('camera_ns').perform(context).strip()

    common_params = [params, {'intrinsics_path': intrinsics}]
    # When a non-default camera namespace is used (cameras_only.launch.py
    # view:=exo publishes to /exo/exo/...) we need two things:
    #
    #  1. camera_frame parameter override — this IS a wildcard in params.yaml
    #     so the inline dict correctly overrides it.
    #
    #  2. Topic redirection — params.yaml stores topic names in node-specific
    #     sections (world_origin_node, detection_node, point_cloud_node), which
    #     take precedence over any wildcard inline dict.  Use ROS 2 topic
    #     remapping instead: it operates below the parameter layer and reliably
    #     redirects whichever topic the node actually subscribes to.
    cam_remaps: list[tuple[str, str]] = []
    if camera_ns and camera_ns != 'camera':
        pfx = f'/{camera_ns}/{camera_ns}'
        common_params.append({'camera_frame': f'{camera_ns}_color_optical_frame'})
        cam_remaps = [
            ('/camera/camera/color/image_raw',
             f'{pfx}/color/image_raw'),
            ('/camera/camera/aligned_depth_to_color/image_raw',
             f'{pfx}/aligned_depth_to_color/image_raw'),
        ]

    # Optional YOLO weight override (view-specialised model selected from
    # params.yaml by digital_twin_sequence.launch.py).  See
    # _write_node_param_override above — node-specific yaml beats wildcard
    # inline dicts, so the override is its own --params-file appended AFTER
    # params.yaml.
    yolo_model = LaunchConfiguration('yolo_model').perform(context)
    detection_params = list(common_params)
    if yolo_model:
        detection_params.append(_write_node_param_override(
            'detection_node', {'model': yolo_model}))

    world_origin = Node(
        package='depth_digital_twin', executable='world_origin_node',
        name='world_origin_node', output='screen', parameters=common_params,
        remappings=cam_remaps)
    detection = Node(
        package='depth_digital_twin', executable='detection_node',
        name='detection_node', output='screen', parameters=detection_params,
        remappings=cam_remaps)
    point_cloud = Node(
        package='depth_digital_twin', executable='point_cloud_node',
        name='point_cloud_node', output='screen', parameters=common_params,
        remappings=cam_remaps)
    rviz = Node(
        package='rviz2', executable='rviz2', name='rviz2',
        arguments=['-d', rviz_cfg],
        condition=IfCondition(LaunchConfiguration('rviz')),
        output='screen')
    control_panel = Node(
        package='depth_digital_twin', executable='world_origin_control',
        name='world_origin_control', output='screen',
        condition=IfCondition(LaunchConfiguration('control_panel')))

    return [world_origin, detection, point_cloud, rviz, control_panel]


def generate_launch_description() -> LaunchDescription:
    pkg_share = FindPackageShare('depth_digital_twin')

    return LaunchDescription([
        DeclareLaunchArgument(
            'intrinsics',
            default_value=PathJoinSubstitution([pkg_share, 'config', 'intrinsics.yaml']),
            description='Absolute path to intrinsics.yaml'),
        DeclareLaunchArgument(
            'params',
            default_value=PathJoinSubstitution([pkg_share, 'config', 'params.yaml']),
            description='Absolute path to params.yaml'),
        DeclareLaunchArgument(
            'rviz', default_value='true',
            description='Launch RViz2'),
        DeclareLaunchArgument(
            'rviz_config',
            default_value=PathJoinSubstitution([pkg_share, 'rviz', 'digital_twin.rviz']),
            description='RViz2 config file'),
        DeclareLaunchArgument(
            'control_panel', default_value='true',
            description='Show Reset/Redetect control panel popup'),
        DeclareLaunchArgument(
            'yolo_model', default_value='',
            description='Override detection_node.model (empty = params.yaml)'),
        DeclareLaunchArgument(
            'camera_ns', default_value='camera',
            description='RealSense camera namespace. '
                        '"camera" = rs_align_depth_launch.py default (/camera/camera/...). '
                        '"exo" = cameras_only.launch.py view:=exo (/exo/exo/...).'),
        OpaqueFunction(function=_make_nodes),
    ])
