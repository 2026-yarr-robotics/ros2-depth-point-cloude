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
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _make_nodes(context, *args, **kwargs):
    intrinsics = LaunchConfiguration('intrinsics').perform(context)
    params = LaunchConfiguration('params').perform(context)
    rviz_cfg = LaunchConfiguration('rviz_config').perform(context)

    common_params = [params, {'intrinsics_path': intrinsics}]

    # Optional YOLO weight override (e.g. a view-specialised model selected
    # from params.yaml by digital_twin_sequence.launch.py). Empty → use
    # detection_node.model from params.yaml.
    yolo_model = LaunchConfiguration('yolo_model').perform(context)
    detection_params = list(common_params)
    if yolo_model:
        detection_params.append({'model': yolo_model})

    world_origin = Node(
        package='depth_digital_twin', executable='world_origin_node',
        name='world_origin_node', output='screen', parameters=common_params)
    detection = Node(
        package='depth_digital_twin', executable='detection_node',
        name='detection_node', output='screen', parameters=detection_params)
    point_cloud = Node(
        package='depth_digital_twin', executable='point_cloud_node',
        name='point_cloud_node', output='screen', parameters=common_params)
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
        OpaqueFunction(function=_make_nodes),
    ])
