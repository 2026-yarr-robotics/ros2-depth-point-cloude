"""Bring up the depth digital twin alongside a Doosan URDF in RViz.

The digital twin's `world` frame is published by `world_origin_node`. When a
hand-eye calibration matrix (T_exo2base.npy / T_hand2base.npy) is loaded,
that frame *is* the robot base, so the URDF can be attached to `world` via
an identity static TF and the cup detections automatically render in the
robot's coordinate system.

Prerequisites:
  source ~/ros2_ws/install/setup.bash      # provides dsr_description2 share
  source ~/ros2-depth-point-cloude/install/setup.bash

Behaviour w.r.t. the live robot (per the spec):
  * `with_pose_bridge:=true`  (default): a small bridge mirrors
    `/dsr01/joint_states` → `/joint_states`. If `dsr_bringup2` is also up,
    the URDF moves with the robot. If not, the bridge falls back to a
    zero-joint default and the URDF stays in its home pose. Either way the
    rest of the pipeline runs untouched.
  * `with_pose_bridge:=false`: the user is expected to provide
    `/joint_states` themselves (e.g. by running joint_state_publisher_gui or
    a full dsr_bringup2 stack).

Usage:
  ros2 launch depth_digital_twin digital_twin_with_robot.launch.py \\
      model:=m0609 \\
      calibration:=$PWD/src/depth_digital_twin/config/T_exo2base.npy
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
        description='Robot model color (per dsr_description2)')
    name_arg = DeclareLaunchArgument(
        'name', default_value='dsr01',
        description='Robot namespace (matches dsr_bringup2 default)')
    bridge_arg = DeclareLaunchArgument(
        'with_pose_bridge', default_value='true',
        description='Bridge /dsr01/joint_states → /joint_states with default fallback')
    calibration_arg = DeclareLaunchArgument(
        'calibration', default_value='',
        description='Absolute path to T_exo2base.npy / T_hand2base.npy (empty ⇒ floor-fit)')
    rviz_arg = DeclareLaunchArgument(
        'rviz', default_value='true', description='Launch RViz')
    rviz_config_arg = DeclareLaunchArgument(
        'rviz_config',
        default_value=PathJoinSubstitution(
            [pkg_dt, 'rviz', 'digital_twin.rviz']),
        description='RViz config')

    # ----- robot URDF (xacro) -----
    robot_description_content = Command([
        PathJoinSubstitution([FindExecutable(name='xacro')]),
        ' ',
        PathJoinSubstitution([pkg_dsr, 'xacro', LaunchConfiguration('model')]),
        '.urdf.xacro',
        ' color:=', LaunchConfiguration('color'),
        ' name:=', LaunchConfiguration('name'),
        # The remaining xacro args are required by dsr_description2 macros
        # but irrelevant for visual-only rendering — pass placeholders.
        ' host:=127.0.0.1 port:=12345 mode:=virtual',
        ' rt_host:=127.0.0.1 update_rate:=100',
        ' model:=', LaunchConfiguration('model'),
    ])
    robot_description = {
        'robot_description': ParameterValue(
            robot_description_content, value_type=str),
    }

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[robot_description],
        output='screen',
    )

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
        output='screen',
    )

    # world ↔ base_0: identity static TF when calibrated mode places `world`
    # on the robot base (the eye-to-hand convention). This lets the URDF
    # graft onto the digital_twin TF tree without a second world frame.
    # Use the new-style flag args; positional `[x y z qx qy qz qw parent child]`
    # is deprecated in ROS 2 Humble (still works but logs a WARN).
    world_to_base_tf = Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='world_to_base_tf',
        arguments=[
            '--x', '0', '--y', '0', '--z', '0',
            '--qx', '0', '--qy', '0', '--qz', '0', '--qw', '1',
            '--frame-id', 'world', '--child-frame-id', 'base_0',
        ],
        output='screen',
    )

    # ----- digital_twin pipeline -----
    digital_twin = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([pkg_dt, 'launch', 'digital_twin.launch.py']),
        ]),
        launch_arguments={
            'rviz': LaunchConfiguration('rviz'),
            'rviz_config': LaunchConfiguration('rviz_config'),
            'calibration': LaunchConfiguration('calibration'),
        }.items(),
    )

    return LaunchDescription([
        model_arg, color_arg, name_arg, bridge_arg, calibration_arg,
        rviz_arg, rviz_config_arg,
        robot_state_publisher, pose_bridge, world_to_base_tf, digital_twin,
    ])
