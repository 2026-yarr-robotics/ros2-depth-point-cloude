"""Bring up the depth digital twin alongside a Doosan URDF in RViz.

world_origin_node detects the ArUco marker and sets world = robot base.
The identity static TF world↔base_0 then grafts the URDF onto the digital
twin TF tree so detections and the robot model share one coordinate frame.

Prerequisites:
  source ~/ros2_ws/install/setup.bash          # dsr_description2
  source ~/ros2-depth-point-cloude/install/setup.bash

Usage:
  ros2 launch depth_digital_twin digital_twin_with_robot.launch.py \\
      model:=m0609

Args:
  model            : Doosan model (default: m0609)
  color            : robot color  (default: white)
  name             : robot namespace, must match dsr_bringup2 (default: dsr01)
  with_pose_bridge : bridge /dsr01/joint_states→/joint_states (default: true)
  rviz             : launch RViz (default: true)
  rviz_config      : path to .rviz file
  intrinsics        : path to intrinsics.yaml
  params            : path to params.yaml
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution,
    PythonExpression)
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    pkg_dt = FindPackageShare('depth_digital_twin')
    pkg_dsr = FindPackageShare('dsr_description2')

    # ----- args -----
    model_arg = DeclareLaunchArgument(
        'model', default_value='m0609',
        description='Doosan model (m0609, m1013, ...)')
    color_arg = DeclareLaunchArgument(
        'color', default_value='white',
        description='Robot model color')
    name_arg = DeclareLaunchArgument(
        'name', default_value='dsr01',
        description='Robot namespace (matches dsr_bringup2 default)')
    bridge_arg = DeclareLaunchArgument(
        'with_pose_bridge', default_value='true',
        description='Bridge /dsr01/joint_states → /joint_states with zero-joint fallback')
    rviz_arg = DeclareLaunchArgument(
        'rviz', default_value='true', description='Launch RViz')
    rviz_config_arg = DeclareLaunchArgument(
        'rviz_config',
        default_value=PathJoinSubstitution([pkg_dt, 'rviz', 'digital_twin.rviz']),
        description='RViz config file')
    intrinsics_arg = DeclareLaunchArgument(
        'intrinsics',
        default_value=PathJoinSubstitution([pkg_dt, 'config', 'intrinsics.yaml']),
        description='Absolute path to intrinsics.yaml')
    params_arg = DeclareLaunchArgument(
        'params',
        default_value=PathJoinSubstitution([pkg_dt, 'config', 'params.yaml']),
        description='Absolute path to params.yaml')

    # ----- robot URDF -----
    robot_description_content = Command([
        PathJoinSubstitution([FindExecutable(name='xacro')]),
        ' ',
        PathJoinSubstitution([pkg_dsr, 'xacro', LaunchConfiguration('model')]),
        '.urdf.xacro',
        ' color:=', LaunchConfiguration('color'),
        ' name:=', LaunchConfiguration('name'),
        ' host:=127.0.0.1 port:=12345 mode:=virtual',
        ' rt_host:=127.0.0.1 update_rate:=100',
        ' model:=', LaunchConfiguration('model'),
    ])

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{'robot_description': ParameterValue(
            robot_description_content, value_type=str)}],
        output='screen')

    pose_bridge = Node(
        package='depth_digital_twin',
        executable='robot_pose_bridge_node',
        name='robot_pose_bridge',
        parameters=[{
            'input_topic': PythonExpression([
                "'/'+'", LaunchConfiguration('name'), "'+'/joint_states'"]),
            'output_topic': '/joint_states',
        }],
        condition=IfCondition(LaunchConfiguration('with_pose_bridge')),
        output='screen')

    # world ↔ base_0 identity TF: world_origin_node places `world` on the robot
    # base, so the URDF (rooted at base_0) can be grafted with an identity TF.
    world_to_base_tf = Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='world_to_base_tf',
        arguments=[
            '--x', '0', '--y', '0', '--z', '0',
            '--qx', '0', '--qy', '0', '--qz', '0', '--qw', '1',
            '--frame-id', 'world', '--child-frame-id', 'base_0',
        ],
        output='screen')

    digital_twin = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([pkg_dt, 'launch', 'digital_twin.launch.py']),
        ]),
        launch_arguments={
            'rviz': LaunchConfiguration('rviz'),
            'rviz_config': LaunchConfiguration('rviz_config'),
            'intrinsics': LaunchConfiguration('intrinsics'),
            'params': LaunchConfiguration('params'),
        }.items())

    return LaunchDescription([
        model_arg, color_arg, name_arg, bridge_arg,
        rviz_arg, rviz_config_arg, intrinsics_arg, params_arg,
        robot_state_publisher, pose_bridge, world_to_base_tf, digital_twin,
    ])
