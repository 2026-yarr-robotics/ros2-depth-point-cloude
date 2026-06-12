"""playback.launch.py — replay a recorded sequence in RViz (#2 debugger).

Shows:
  • the recorded exo + hand colour/depth streams (RViz Image displays)
  • the M0609 model animated by the recorded joint trajectory
  • the recorded EE (TCP) 6D pose as TF `ee_recorded` + an axes/sphere marker
  • a Tk control panel: Stop / Resume / Replay / goto-step+Apply

No robot hardware or controller is needed — robot_state_publisher + the
recorded /joint_states is enough to visualise the arm.

Usage:
  ros2 launch recode_sequence playback.launch.py \\
      sequence:=/home/eunwoosong/Projects/record_sequence/0001
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import (Command, FindExecutable,
                                  LaunchConfiguration, PathJoinSubstitution)
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def _player(context, *_, **__):
    pkg = get_package_share_directory('recode_sequence')
    params = os.path.join(pkg, 'config', 'recode_params.yaml')
    seq = LaunchConfiguration('sequence').perform(context)
    if not seq:
        raise RuntimeError(
            'sequence:=/path/to/record_sequence/0001 is required')
    return [Node(
        package='recode_sequence', executable='sequence_player_node',
        name='sequence_player_node', output='screen',
        parameters=[params, {'sequence_dir': seq}])]


def generate_launch_description() -> LaunchDescription:
    pkg = FindPackageShare('recode_sequence')
    pkg_dsr = FindPackageShare('dsr_description2')

    robot_description = Command([
        PathJoinSubstitution([FindExecutable(name='xacro')]), ' ',
        PathJoinSubstitution([pkg_dsr, 'xacro',
                              LaunchConfiguration('model')]),
        '.urdf.xacro',
        ' color:=', LaunchConfiguration('color'),
        ' name:=', LaunchConfiguration('name'),
        ' host:=127.0.0.1 port:=12345 mode:=virtual',
        ' rt_host:=127.0.0.1 update_rate:=100',
        ' model:=', LaunchConfiguration('model'),
    ])

    return LaunchDescription([
        DeclareLaunchArgument('sequence', default_value='',
                              description='Absolute path to a sequence dir'),
        DeclareLaunchArgument('model', default_value='m0609'),
        DeclareLaunchArgument('color', default_value='white'),
        DeclareLaunchArgument('name', default_value='dsr01'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('control_panel', default_value='true'),
        DeclareLaunchArgument(
            'rviz_config',
            default_value=PathJoinSubstitution(
                [pkg, 'rviz', 'playback.rviz'])),

        Node(package='robot_state_publisher',
             executable='robot_state_publisher',
             name='robot_state_publisher', output='screen',
             parameters=[{'robot_description': ParameterValue(
                 robot_description, value_type=str)}]),

        OpaqueFunction(function=_player),

        Node(package='rviz2', executable='rviz2', name='rviz2',
             output='log',
             arguments=['-d', LaunchConfiguration('rviz_config')],
             condition=IfCondition(LaunchConfiguration('rviz'))),

        Node(package='recode_sequence', executable='playback_control',
             name='playback_control', output='screen',
             condition=IfCondition(LaunchConfiguration('control_panel'))),
    ])
