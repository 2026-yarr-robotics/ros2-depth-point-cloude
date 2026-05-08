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
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _make_nodes(context, *args, **kwargs):
    """Resolve LaunchConfigurations to concrete strings, then build the Node
    list. We do this at OpaqueFunction time because nesting a
    `ParameterValue(LaunchConfiguration(...), value_type=str)` inside a
    Node's `parameters=[yaml_file, dict]` overlay is unreliable in ROS 2
    Humble — the substitution did not actually override the YAML's empty
    `calibration_matrix_path` value at runtime. Materialising the strings
    here sidesteps that gap entirely."""
    intrinsics = LaunchConfiguration('intrinsics').perform(context)
    params = LaunchConfiguration('params').perform(context)
    calibration = LaunchConfiguration('calibration').perform(context)
    rviz_cfg = LaunchConfiguration('rviz_config').perform(context)

    overlay = {'intrinsics_path': intrinsics}
    # Only include calibration_matrix_path when non-empty; otherwise let the
    # YAML's default (empty ⇒ floor-plane fit) stand.
    if calibration:
        overlay['calibration_matrix_path'] = calibration

    common_params = [params, overlay]

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
        arguments=['-d', rviz_cfg],
        condition=IfCondition(LaunchConfiguration('rviz')),
        output='screen')

    return [world_origin, detection, point_cloud, rviz]


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
    # Optional hand-eye result. Empty string ⇒ legacy floor-plane fit
    # (see world_origin_node.calibration_matrix_path).
    calibration_arg = DeclareLaunchArgument(
        'calibration', default_value='',
        description='Absolute path to T_exo2base.npy / T_hand2base.npy')

    return LaunchDescription([
        intrinsics_arg, params_arg, rviz_arg, rviz_config_arg, calibration_arg,
        OpaqueFunction(function=_make_nodes),
    ])
