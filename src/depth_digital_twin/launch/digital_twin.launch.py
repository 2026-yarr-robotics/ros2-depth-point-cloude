"""Bring up the depth digital twin pipeline + RViz2.

Usage:
  # Launch RealSense camera (separate terminal — use the command from the PDF):
  ros2 launch realsense2_camera rs_align_depth_launch.py \
      depth_module.depth_profile:=640x480x30 \
      rgb_camera.color_profile:=640x480x30 \
      initial_reset:=true align_depth.enable:=true

  # Then:
  ros2 launch depth_digital_twin digital_twin.launch.py

Optional args:
  intrinsics:=/abs/path/to/intrinsics.yaml
  params:=/abs/path/to/params.yaml
  rviz:=true|false
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    pkg_share = FindPackageShare('depth_digital_twin')

    default_intrinsics = PathJoinSubstitution([pkg_share, 'config', 'intrinsics.yaml'])
    default_params = PathJoinSubstitution([pkg_share, 'config', 'params.yaml'])
    default_rviz = PathJoinSubstitution([pkg_share, 'rviz', 'digital_twin.rviz'])

    intrinsics_arg = DeclareLaunchArgument(
        'intrinsics', default_value=default_intrinsics,
        description='Absolute path to intrinsics.yaml')
    params_arg = DeclareLaunchArgument(
        'params', default_value=default_params,
        description='Absolute path to ROS params YAML')
    rviz_arg = DeclareLaunchArgument(
        'rviz', default_value='true', description='Launch RViz2 alongside the pipeline')
    rviz_config_arg = DeclareLaunchArgument(
        'rviz_config', default_value=default_rviz,
        description='RViz2 config file')

    intrinsics = LaunchConfiguration('intrinsics')
    params = LaunchConfiguration('params')

    common_params = [params, {
        'intrinsics_path': ParameterValue(intrinsics, value_type=str),
    }]

    world_origin = Node(
        package='depth_digital_twin', executable='world_origin_node',
        name='world_origin_node', output='screen', parameters=common_params)
    detection = Node(
        package='depth_digital_twin', executable='detection_node',
        name='detection_node', output='screen', parameters=common_params)
    point_cloud = Node(
        package='depth_digital_twin', executable='point_cloud_node',
        name='point_cloud_node', output='screen', parameters=common_params)

    rviz = Node(
        package='rviz2', executable='rviz2', name='rviz2',
        arguments=['-d', LaunchConfiguration('rviz_config')],
        condition=IfCondition(LaunchConfiguration('rviz')),
        output='screen')

    return LaunchDescription([
        intrinsics_arg, params_arg, rviz_arg, rviz_config_arg,
        world_origin, detection, point_cloud, rviz,
    ])
