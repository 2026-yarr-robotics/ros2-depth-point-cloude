"""Add the HAND camera as a second producer feeding the already-running
cup_fusion_node (exo-only fusion must be up). Uses HAND-only topics so it does
NOT cross-contaminate the exo pipeline, the hand-specialised YOLO (model_hand),
and reuses the LIVE robot TF (dsr base_link->link_6) + static eye-in-hand TF
(no second robot_state_publisher -> no /tf fight).

TF chain:  hand_color_optical_frame --(static handeye)--> link_6
           link_6 --(live dsr FK)--> base_link
           base_link --(static world<->base, x/y/z + yaw knobs)--> world

world_base_{x,y,z,yaw_deg} align the robot/hand subtree to the exo ArUco
`world`. Defaults are the measured correction (exo cup vs hand cup) so a single
cup fuses to ONE box. If hand-fused cups still split from exo, nudge these
(measure exo vs hand centroid for the same cup; the diff is the residual).
handeye from T_gripper2camera.npy (mm->m, ~178deg about Z).
"""
import glob
import math
import os
import re

import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


# Mirror digital_twin_fusion.launch.py: overlay the newest panel-saved tuning
# snapshot (params_YYMMDD_HHMMSS.yaml) on top of params.yaml so slider tuning
# persists across restarts. Matches ONLY the timestamped name so params.yaml /
# params_back.yaml are never picked. Snapshots are PARTIAL (per-node sections) —
# they OVERLAY, never replace.
_TUNING_RE = re.compile(r'^params_\d{6}_\d{6}\.yaml$')


def _latest_tuning(cfg_dir: str):
    snaps = [f for f in glob.glob(os.path.join(cfg_dir, 'params_*.yaml'))
             if _TUNING_RE.match(os.path.basename(f))]
    return max(snaps, key=os.path.basename) if snaps else None


def _setup(context, *a, **k):
    pkg = FindPackageShare('depth_digital_twin').perform(context)
    params_path = f'{pkg}/config/params.yaml'
    intr = f'{pkg}/config/intrinsics.yaml'
    with open(params_path) as f:
        P = yaml.safe_load(f) or {}
    det_p = (P.get('detection_node') or {}).get('ros__parameters', {})
    pc_p = (P.get('point_cloud_node') or {}).get('ros__parameters', {})
    # ArUco / hand-eye params live under `world_origin_node:`. The node below is
    # RENAMED (world_origin_node_hand) so that section would NOT auto-apply —
    # pass it inline (same trick as digital_twin_fusion.launch.py).
    wo = (P.get('world_origin_node') or {}).get('ros__parameters', {})
    model_hand = det_p.get('model_hand') or det_p.get('model')

    def L(n):
        return LaunchConfiguration(n).perform(context)

    # Newest panel-saved tuning snapshot, overlaid on the hand producer so the
    # point_cloud_node_hand slider tuning (e.g. hand depth_grad) survives a
    # restart — same behaviour as digital_twin_fusion.launch.py.
    tuning: list = []
    if L('load_latest_tuning') == 'true':
        latest = _latest_tuning(f'{pkg}/config')
        if latest:
            tuning = [latest]
            print(f'[hand_fusion_add] tuning overlay → {latest}')
        else:
            print('[hand_fusion_add] load_latest_tuning: no params_*.yaml '
                  'snapshot found')
    yaw = str(math.radians(float(L('world_base_yaw_deg'))))

    # Hand eye-in-hand calibration, ArUco-driven. world_origin_node in
    # `handeye_aruco` mode computes link_6 → hand_color_optical_frame from the
    # workspace ArUco marker (ID 0) seen by the WRIST camera + the live
    # base_link→link_6 FK (dsr). This REPLACES the old hardcoded static TF so the
    # panel's "ArUco Hand" button (/world_origin_node_hand/redetect) actually
    # re-runs the live hand-eye calibration.
    #   • startup: handeye_use_param_offset=true (from params.yaml) publishes the
    #     known-good offset immediately → hand fusion works without waiting on a
    #     marker sighting (no regression vs the old static TF).
    #   • ~/redetect: subscribes to color and recomputes the hand-eye live.
    # handeye_emit_world_to_base stays False (default) → this node does NOT emit
    # world→base_link; the static `world_base` node below owns that edge.
    handeye = Node(
        package='depth_digital_twin', executable='world_origin_node',
        name='world_origin_node_hand', output='screen',
        parameters=[wo, {
            'world_origin_mode': 'handeye_aruco',
            'intrinsics_path': intr,
            'color_topic': '/hand/hand/color/image_raw',
            'camera_frame': 'hand_color_optical_frame',
            'world_frame': 'world',
            'base_frame': 'base_link',
            'ee_frame': 'link_6',
            'aruco_timeout_then_floor': False,
        }])
    world_base = Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='world_base_link',
        arguments=[L('world_base_x'), L('world_base_y'), L('world_base_z'),
                   yaw, '0', '0', 'world', 'base_link'])

    det_hand = Node(
        package='depth_digital_twin', executable='detection_node',
        name='detection_node_hand', output='screen',
        parameters=[{**det_p,
                     'intrinsics_path': intr,
                     'image_topic': '/hand/hand/color/image_raw',
                     'detections_topic': '/digital_twin/detections_hand',
                     'debug_topic': '/digital_twin/detection_debug_hand',
                     'model': model_hand}])
    pc_hand = Node(
        package='depth_digital_twin', executable='point_cloud_node',
        name='point_cloud_node_hand', output='screen',
        parameters=[{**pc_p,
                     'intrinsics_path': intr,
                     'rgb_topic': '/hand/hand/color/image_raw',
                     'depth_topic':
                         '/hand/hand/aligned_depth_to_color/image_raw',
                     'detections_topic': '/digital_twin/detections_hand',
                     'box_debug_topic': '/digital_twin/box_debug_hand',
                     'depth_debug_topic': '/digital_twin/depth_debug_hand',
                     'camera_frame': 'hand_color_optical_frame',
                     'world_frame': 'world',
                     'role': 'producer',
                     'camera_name': 'hand',
                     'world_clouds_topic': '/digital_twin/cups_hand',
                     # Suppress the hand cloud while the wrist is moving so a
                     # motion-smeared cloud is not fused (parity with
                     # digital_twin_fusion.launch.py). Doosan publishes joints
                     # under /dsr01 (no global /joint_states in start.sh).
                     'hand_motion_gating': True,
                     'joint_states_topic': '/dsr01/joint_states'},
                    *tuning])
    return [handeye, world_base, det_hand, pc_hand]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('world_base_x', default_value='0.016'),
        DeclareLaunchArgument('world_base_y', default_value='-0.020'),
        DeclareLaunchArgument('world_base_z', default_value='-0.065'),
        DeclareLaunchArgument('world_base_yaw_deg', default_value='0'),
        DeclareLaunchArgument(
            'load_latest_tuning', default_value='true',
            description='Overlay the newest panel-saved params_<ts>.yaml '
                        'snapshot on the hand producer (false = params.yaml '
                        'only).'),
        OpaqueFunction(function=_setup),
    ])
