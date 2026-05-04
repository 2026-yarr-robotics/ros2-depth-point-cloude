"""Cube R-CNN digital-twin pipeline (RGB-only 3D detection → PC + 6D box).

Launch arguments:
    config_file:=/abs/path/to/cubercnn_DLA34_FPN.yaml
    weights:=/abs/path/to/cubercnn_DLA34_FPN.pth
    intrinsics:=/abs/path/to/intrinsics.yaml
    device:=cuda|cpu

Falls back to $CUBERCNN_CONFIG / $CUBERCNN_WEIGHTS env vars exported by
dependence/activate.bash when the corresponding launch arg is omitted.

Implementation note: parameter values are passed via `-p key:=value` ROS 2 CLI
overrides (highest precedence) instead of `parameters=[{...}]` dicts. The dict
form has been observed to silently drop entries on Humble when combined with a
`--params-file` for the same node, which produced spurious "config_file is
EMPTY" failures.
"""
from __future__ import annotations

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _spawn_nodes(context, *args, **kwargs):
    intrinsics = LaunchConfiguration('intrinsics').perform(context)
    params = LaunchConfiguration('params').perform(context)
    cfg = (LaunchConfiguration('config_file').perform(context)
           or os.environ.get('CUBERCNN_CONFIG', ''))
    wts = (LaunchConfiguration('weights').perform(context)
           or os.environ.get('CUBERCNN_WEIGHTS', ''))
    device = LaunchConfiguration('device').perform(context)
    rviz_cfg = LaunchConfiguration('rviz_config').perform(context)
    rviz_on = LaunchConfiguration('rviz')

    def ros_p_args(extras: dict[str, str]) -> list[str]:
        # NOTE: do NOT prepend '--ros-args' here — `ros_arguments=` already
        # wraps the entries inside the single --ros-args block that Node
        # creates for `name=`/remappings, avoiding the "two blocks, last wins"
        # quirk that swallows -p overrides in Humble.
        a = ['--params-file', params,
             '-p', f'intrinsics_path:={intrinsics}']
        for k, v in extras.items():
            a += ['-p', f'{k}:={v}']
        return a

    world_origin = Node(
        package='depth_digital_twin', executable='world_origin_node',
        name='world_origin_node', output='screen',
        ros_arguments=ros_p_args({}))
    cube_rcnn = Node(
        package='depth_digital_twin', executable='cube_rcnn_node',
        name='cube_rcnn_node', output='screen',
        ros_arguments=ros_p_args({
            'config_file': cfg,
            'weights': wts,
            'device': device,
        }))
    cube_pc = Node(
        package='depth_digital_twin', executable='cube_point_cloud_node',
        name='cube_point_cloud_node', output='screen',
        ros_arguments=ros_p_args({}))
    rviz = Node(
        package='rviz2', executable='rviz2', name='rviz2_cube',
        arguments=['-d', rviz_cfg], output='screen',
        condition=IfCondition(rviz_on))

    return [world_origin, cube_rcnn, cube_pc, rviz]


def generate_launch_description() -> LaunchDescription:
    pkg_share = FindPackageShare('depth_digital_twin')

    default_intrinsics = PathJoinSubstitution([pkg_share, 'config', 'intrinsics.yaml'])
    default_params = PathJoinSubstitution([pkg_share, 'config', 'params.yaml'])
    default_rviz = PathJoinSubstitution([pkg_share, 'rviz', 'cube_rcnn.rviz'])

    args = [
        DeclareLaunchArgument('intrinsics', default_value=default_intrinsics,
                              description='Absolute path to intrinsics.yaml'),
        DeclareLaunchArgument('params', default_value=default_params,
                              description='Absolute path to ROS params YAML'),
        DeclareLaunchArgument('config_file', default_value='',
                              description='Cube R-CNN config (.yaml). Falls '
                              'back to $CUBERCNN_CONFIG when empty.'),
        DeclareLaunchArgument('weights', default_value='',
                              description='Cube R-CNN pretrained weights '
                              '(.pth). Falls back to $CUBERCNN_WEIGHTS.'),
        DeclareLaunchArgument('device', default_value='cuda',
                              description="'cuda' or 'cpu'"),
        DeclareLaunchArgument('rviz', default_value='true',
                              description='Launch RViz2 with cube_rcnn config'),
        DeclareLaunchArgument('rviz_config', default_value=default_rviz,
                              description='RViz2 config file'),
    ]

    return LaunchDescription([*args, OpaqueFunction(function=_spawn_nodes)])
