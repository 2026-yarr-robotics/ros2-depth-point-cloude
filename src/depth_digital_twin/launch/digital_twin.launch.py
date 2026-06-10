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
import glob
import os
import re
import tempfile

import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


# Mirror digital_twin_fusion.launch.py: overlay the newest panel-saved tuning
# snapshot (params_YYMMDD_HHMMSS.yaml) on top of params.yaml so cup_fusion_node
# slider tuning + debug-plot toggles persist across restarts. Matches ONLY the
# timestamped name (never params.yaml / params_back.yaml). Snapshots are PARTIAL
# (per-node sections) — they OVERLAY, never replace.
_TUNING_RE = re.compile(r'^params_\d{6}_\d{6}\.yaml$')


def _latest_tuning(cfg_dir: str):
    snaps = [f for f in glob.glob(os.path.join(cfg_dir, 'params_*.yaml'))
             if _TUNING_RE.match(os.path.basename(f))]
    return max(snaps, key=os.path.basename) if snaps else None


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

    # Newest panel-saved tuning snapshot, overlaid on cup_fusion_node so the
    # slider tuning + debug-plot toggles survive a restart (parity with
    # digital_twin_fusion.launch.py). Only applies in fusion mode (cup_fusion).
    tuning: list = []
    if LaunchConfiguration('load_latest_tuning').perform(context) == 'true':
        latest = _latest_tuning(os.path.dirname(params))
        if latest:
            tuning = [latest]
            print(f'[digital_twin] tuning overlay → {latest}')
        else:
            print('[digital_twin] load_latest_tuning: no params_*.yaml '
                  'snapshot found')

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
    # Insert cup_dedup_node between detection and point_cloud: detection_node
    # publishes the RAW array on /digital_twin/detections_raw, the dedup node
    # collapses same-cup duplicate boxes (default NMS iou=0.7 + ByteTrack lets
    # one far cup survive as two ids) and republishes on /digital_twin/detections
    # — the topic point_cloud_node already subscribes to (no PC change needed).
    detection_remaps = cam_remaps + [
        ('/digital_twin/detections', '/digital_twin/detections_raw')]
    detection = Node(
        package='depth_digital_twin', executable='detection_node',
        name='detection_node', output='screen', parameters=detection_params,
        remappings=detection_remaps)
    cup_dedup = Node(
        package='depth_digital_twin', executable='cup_dedup_node',
        name='cup_dedup_node', output='screen', parameters=[params,
            {'detections_in': '/digital_twin/detections_raw',
             'detections_out': '/digital_twin/detections'}])
    # Fusion (exo-only): point_cloud runs as a producer feeding cup_fusion_node,
    # which owns /digital_twin/boxes + /vision/cups_on_table. Default false keeps
    # the standalone single-camera behaviour (rollback baseline) unchanged.
    fusion = LaunchConfiguration('fusion').perform(context).strip().lower() \
        in ('true', '1')
    pc_params = list(common_params)
    if fusion:
        # Per-camera rim topics: the producer defaults are SHARED names --
        # with hand_fusion_add.launch.py running too, both cameras would
        # interleave frames on one rim_debug topic (violent flicker) and
        # cup_fusion_node would never see the obs (it subscribes _exo/_hand).
        pc_params.append({'role': 'producer',
                          'world_clouds_topic': '/digital_twin/cups_exo',
                          'camera_name': 'exo',
                          'cup_obs_topic': '/digital_twin/cup_obs_exo',
                          'rim_debug_topic': '/digital_twin/rim_debug_exo'})
    point_cloud = Node(
        package='depth_digital_twin', executable='point_cloud_node',
        name='point_cloud_node', output='screen', parameters=pc_params,
        remappings=cam_remaps)
    cup_fusion = Node(
        package='depth_digital_twin', executable='cup_fusion_node',
        name='cup_fusion_node', output='screen', parameters=[params, *tuning],
        condition=IfCondition(LaunchConfiguration('fusion')))
    rviz = Node(
        package='rviz2', executable='rviz2', name='rviz2',
        arguments=['-d', rviz_cfg],
        condition=IfCondition(LaunchConfiguration('rviz')),
        output='screen')
    if fusion:
        # Integrated dual-cam panel (exo+hand RGB/Depth/3D + ArUco redetect),
        # remapped to this setup's live topics/service. Replaces the old
        # per-node world_origin_control popup.
        control_panel = Node(
            package='depth_digital_twin', executable='digital_twin_panel',
            name='digital_twin_panel', output='screen',
            parameters=[{
                'exo_color_topic': '/exo/exo/color/image_raw',
                'exo_depth_topic': '/exo/exo/aligned_depth_to_color/image_raw',
                # 3D pane = rim-fit overlay (the actual live measurement);
            # raw YOLO stays on /digital_twin/detection_debug*
            'exo_debug_topic': '/digital_twin/rim_debug_exo',
            'exo_pc_node': 'point_cloud_node',
                'hand_color_topic': '/hand/hand/color/image_raw',
                'hand_depth_topic':
                    '/hand/hand/aligned_depth_to_color/image_raw',
                'hand_debug_topic': '/digital_twin/rim_debug_hand',
                'exo_redetect_srv': '/world_origin_node/redetect',
                # Hand redetect is the handeye_aruco node added by
                # hand_fusion_add.launch.py (VISION_MODE=fusion_dual). In
                # exo-only fusion that node is absent, so the Hand button is a
                # no-op there — expected, since there is no hand camera.
                'hand_redetect_srv': '/world_origin_node_hand/redetect',
            }],
            condition=IfCondition(LaunchConfiguration('control_panel')))
    else:
        control_panel = Node(
            package='depth_digital_twin', executable='world_origin_control',
            name='world_origin_control', output='screen',
            condition=IfCondition(LaunchConfiguration('control_panel')))

    return [world_origin, detection, cup_dedup, point_cloud, cup_fusion, rviz,
            control_panel]


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
        DeclareLaunchArgument(
            'fusion', default_value='false',
            description='exo-only fusion: point_cloud as producer + '
                        'cup_fusion_node owns /digital_twin/boxes + '
                        '/vision/cups_on_table (default false = standalone)'),
        DeclareLaunchArgument(
            'load_latest_tuning', default_value='true',
            description='Overlay the newest panel-saved params_<ts>.yaml '
                        'snapshot on cup_fusion_node (false = params.yaml '
                        'only).'),
        OpaqueFunction(function=_make_nodes),
    ])
