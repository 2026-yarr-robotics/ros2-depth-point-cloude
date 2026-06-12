"""record.launch.py — start the two cameras and record IMMEDIATELY.

The Doosan robot is brought up separately by you, e.g.:

  ros2 launch dsr_bringup2 dsr_bringup2_rviz.launch.py \\
      mode:=real model:=m0609 host:=192.168.1.100

The recorder subscribes to that stack's /dsr01/joint_states and the
EE RT topics (no get_current_posx service call).

Usage:
  ros2 launch recode_sequence record.launch.py
  ros2 launch recode_sequence record.launch.py with_cameras:=false   # cams already up
  ros2 launch recode_sequence record.launch.py output_root:=/data/seq

Stop with Ctrl-C — the sequence (meta.json + trajectory.pkl) is then
finalised and closed.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            OpaqueFunction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _recorder(context, *_, **__):
    pkg = get_package_share_directory('recode_sequence')
    params = os.path.join(pkg, 'config', 'recode_params.yaml')
    overrides = {}
    out = LaunchConfiguration('output_root').perform(context)
    if out:
        overrides['output_root'] = out
    return [Node(
        package='recode_sequence', executable='recorder_node',
        name='recorder_node', output='screen',
        parameters=[params, overrides])]


def generate_launch_description() -> LaunchDescription:
    pkg = get_package_share_directory('recode_sequence')
    return LaunchDescription([
        DeclareLaunchArgument('with_cameras', default_value='true',
                              description='Also start the two D435i cameras'),
        DeclareLaunchArgument('output_root', default_value='',
                              description='Override recorder output_root'),
        DeclareLaunchArgument(
            'cameras_yaml',
            default_value=os.path.join(pkg, 'config', 'cameras.yaml')),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg, 'launch', 'cameras_only.launch.py')),
            launch_arguments={
                'cameras_yaml': LaunchConfiguration('cameras_yaml'),
            }.items(),
            condition=IfCondition(LaunchConfiguration('with_cameras'))),
        OpaqueFunction(function=_recorder),
    ])
