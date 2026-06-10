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
import math
import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _setup(context, *a, **k):
    pkg = FindPackageShare('depth_digital_twin').perform(context)
    params_path = f'{pkg}/config/params.yaml'
    intr = f'{pkg}/config/intrinsics.yaml'
    with open(params_path) as f:
        P = yaml.safe_load(f) or {}
    det_p = (P.get('detection_node') or {}).get('ros__parameters', {})
    pc_p = (P.get('point_cloud_node') or {}).get('ros__parameters', {})
    model_hand = det_p.get('model_hand') or det_p.get('model')

    def L(n):
        return LaunchConfiguration(n).perform(context)
    yaw = str(math.radians(float(L('world_base_yaw_deg'))))

    handeye = Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='handeye_link6_hand',
        arguments=['0.0365', '0.0600', '0.0059',
                   '0.00245', '-0.00300', '0.99986', '-0.01621',
                   'link_6', 'hand_color_optical_frame'])
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
                     'world_clouds_topic': '/digital_twin/cups_hand'}])
    return [handeye, world_base, det_hand, pc_hand]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('world_base_x', default_value='0.016'),
        DeclareLaunchArgument('world_base_y', default_value='-0.020'),
        DeclareLaunchArgument('world_base_z', default_value='-0.065'),
        DeclareLaunchArgument('world_base_yaw_deg', default_value='0'),
        OpaqueFunction(function=_setup),
    ])
