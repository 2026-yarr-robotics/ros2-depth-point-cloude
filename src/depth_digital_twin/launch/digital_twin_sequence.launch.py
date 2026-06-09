"""Phase 2a — run the cup-detection pipeline from a RECORDED sequence.

Choose which recorded camera feeds the pipeline with `view:=exo|hand`.

  view:=exo  (default)
    ArUco-based world calibration — same as live-camera mode.
    Frame: camera_color_optical_frame → world (via world_origin_node).

  view:=hand
    Runtime hand-eye calibration via ArUco (replaces the prior static npy
    chain).  The recorded sequence MUST include frames where the hand camera
    sees the same workspace ArUco marker as exo.  Requires dsr_description2:
      source ~/ros2_ws/install/setup.bash
    TF chain (after handeye_aruco calibration completes):
      world ──(world_origin_node, identity)──► base_link
      base_link ──(URDF FK from /joint_states)──► link_6
      link_6 ──(world_origin_node handeye_aruco)──► hand_color_optical_frame
    Each per-sample hand-eye is computed as
      T_link6_cam = inv(T_base_link6) · inv(T_cam_marker · T_marker_base)
    over `world_marker_samples_required` ArUco detections, then SE(3)-averaged.

Prerequisites:
  source ~/Projects/ros2-recode-sequence/install/setup.bash
  source ~/Projects/ros2-depth-point-cloude/install/setup.bash
  source ~/ros2_ws/install/setup.bash   # view:=hand only (dsr_description2)

Usage:
  ros2 launch depth_digital_twin digital_twin_sequence.launch.py \\
      sequence:=/home/eunwoosong/Projects/record_sequence/0010
  ros2 launch depth_digital_twin digital_twin_sequence.launch.py \\
      sequence:=/home/eunwoosong/Projects/record_sequence/0010 view:=hand
  # isolate the replay so it can run next to a live bringup:
  ros2 launch depth_digital_twin digital_twin_sequence.launch.py \\
      sequence:=/home/eunwoosong/Projects/record_sequence/0010 ns:=record

Args:
  sequence    : absolute path to a recorded sequence dir (REQUIRED)
  view        : exo | hand — camera into pipeline (default exo)
  ns          : namespace for the WHOLE replay stack (default empty = root).
                ns:=record ⇒ joint_states / TF / camera / /digital_twin/* all
                move under /record, so a recorded replay coexists with a live
                bringup.  A dedicated RViz comes up reading /record/tf.
  yolo_model  : override detection_node.model (empty = params.yaml selection)
  handeye_npy : DEPRECATED — view:=hand now uses runtime ArUco-based hand-eye.
                Argument kept for backwards compat with older invocations; the
                value is ignored.
  model       : Doosan URDF model name (default m0609, view:=hand only)
  color       : URDF color (default white, view:=hand only)
  name        : robot instance name (default dsr01, view:=hand only)
  loop        : replay loop (default false)
  autostart   : start playing immediately (default true)
  params      : pipeline params.yaml (default: package config)
  rviz        : launch RViz2 (default true)
"""
import json
import os
import tempfile

import yaml


def _write_node_param_override(node_name: str, params: dict) -> str:
    """Write a temp YAML pinning per-node parameters under `<node_name>:
    ros__parameters: …` and return its path.

    Why we need this: when params.yaml stores a key in a node-specific section
    (`detection_node: ros__parameters: model: …`), ROS 2's parameter loader
    treats node-specific entries with higher precedence than any wildcard
    inline-dict that launch_ros serialises for us — so passing the override
    via `parameters=[{'model': new_path}]` (which becomes `/**: model: new`)
    silently loses to the user's yaml.  A second --params-file with the SAME
    node-specific structure DOES override the first (rcl applies them in
    order), so we generate one on the fly and append it AFTER params.yaml.
    """
    fd, path = tempfile.mkstemp(
        suffix=f'_{node_name}_override.yaml', text=True)
    with os.fdopen(fd, 'w') as f:
        yaml.safe_dump({node_name: {'ros__parameters': params}}, f)
    return path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, GroupAction,
                            IncludeLaunchDescription, OpaqueFunction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (Command, FindExecutable, LaunchConfiguration,
                                  PathJoinSubstitution)
from launch_ros.actions import Node, PushRosNamespace, SetRemap
from launch_ros.descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

_HANDEYE_DEFAULT = ('/home/eunwoosong/Projects/ros2-recode-sequence/'
                    'src/recode_sequence/config/T_gripper2camera.npy')

_PIPE_COLOR = '/camera/camera/color/image_raw'
_PIPE_DEPTH = '/camera/camera/aligned_depth_to_color/image_raw'
_PIPE_INFO = '/camera/camera/color/camera_info'

# Absolute names the replay stack pub/subs that ALSO exist in a live bringup.
# When `ns:=<name>` is set we redirect each to /<ns>/… so a recorded replay
# can run next to the real robot without fighting over /joint_states, the TF
# tree, the camera feed, or the /digital_twin/* pipeline topics.  PushRosNamespace
# moves node names + relative/private names; these absolute ones (hardcoded here
# and in params.yaml) ignore the namespace, so they need explicit remaps.
#   • /tf, /tf_static — isolating the TF *topic* gives the replay its own TF
#     buffer, so identical frame ids (world/base_link/link_6) no longer clash;
#     no frame-prefix needed.
#   • /world_origin_node/redetect — world_origin_control calls this absolute
#     service name, so it must follow the namespaced server.
_NS_REMAP_NAMES = (
    _PIPE_COLOR, _PIPE_DEPTH, _PIPE_INFO,
    '/joint_states',
    '/tf', '/tf_static',
    '/digital_twin/detections', '/digital_twin/detection_debug',
    '/digital_twin/points', '/digital_twin/boxes', '/digital_twin/box_debug',
    '/vision/cups_on_table',
    '/world_origin_node/redetect',
)


def _namespace_group(ns: str, actions: list):
    """Wrap every replay action under `/<ns>`: push the namespace (node names +
    relative names) and remap the absolute colliding names above into it.
    SetRemap/PushRosNamespace both propagate into the included
    digital_twin.launch.py pipeline, so no node/params.yaml edits are needed."""
    remaps = [SetRemap(src=n, dst=f'/{ns}{n}') for n in _NS_REMAP_NAMES]
    return GroupAction([PushRosNamespace(ns), *remaps, *actions])


_IDLE = {
    'exo': {'c': '/exo/exo/color/image_raw',
            'd': '/exo/exo/aligned_depth_to_color/image_raw',
            'i': '/exo/exo/color/camera_info', 'f': 'exo_color_optical_frame'},
    'hand': {'c': '/hand/hand/color/image_raw',
             'd': '/hand/hand/aligned_depth_to_color/image_raw',
             'i': '/hand/hand/color/camera_info',
             'f': 'hand_color_optical_frame'},
}


def _export_intrinsics(seq_dir: str, view: str) -> str:
    with open(os.path.join(seq_dir, 'meta.json')) as f:
        meta = json.load(f)
    cam = (meta.get('cameras') or {}).get(view) or {}
    K = cam.get('K') or []
    if len(K) != 9 or not cam.get('width'):
        raise RuntimeError(
            f'sequence {seq_dir} has no {view} camera intrinsics in '
            'meta.json (was that camera_info topic recorded?)')
    dist = cam.get('dist') or [0.0] * 5
    out = {
        'image_width': int(cam['width']),
        'image_height': int(cam['height']),
        'camera_matrix': {'rows': 3, 'cols': 3,
                          'data': [float(v) for v in K]},
        'distortion_coefficients': {'rows': 1, 'cols': len(dist),
                                    'data': [float(v) for v in dist]},
        'reprojection_error': 0.0,
    }
    path = os.path.join(seq_dir, f'{view}_intrinsics.yaml')
    with open(path, 'w') as f:
        yaml.safe_dump(out, f, sort_keys=False)
    return path


def _setup(context, *_, **__):
    seq = LaunchConfiguration('sequence').perform(context)
    view = LaunchConfiguration('view').perform(context).strip().lower()
    if view not in ('exo', 'hand'):
        raise RuntimeError(f"view must be 'exo' or 'hand' (got {view!r})")
    if not seq or not os.path.isdir(seq):
        raise RuntimeError(
            'sequence:=/abs/path/to/record_sequence/NNNN is required '
            f'(got {seq!r})')
    seq = os.path.abspath(seq)
    intr = _export_intrinsics(seq, view)
    print(f'[seq] view={view}  intrinsics → {intr}')

    pkg = get_package_share_directory('depth_digital_twin')
    params = LaunchConfiguration('params').perform(context)
    with open(params) as f:
        dn = ((yaml.safe_load(f) or {}).get('detection_node') or {}).get(
            'ros__parameters') or {}
    explicit = LaunchConfiguration('yolo_model').perform(context).strip()
    model = explicit or dn.get(f'model_{view}') or dn.get('model') or ''
    print(f'[seq] view={view}  yolo model → {model or "(params.yaml default)"}')

    autostart = LaunchConfiguration('autostart').perform(context) == 'true'
    loop_flag = LaunchConfiguration('loop').perform(context) == 'true'
    common_params = [params, {'intrinsics_path': intr}]

    # Empty (default) ⇒ run at root namespace exactly as before.  Set
    # ns:=record (etc.) to isolate the whole replay stack under /record so it
    # can coexist with a live bringup — see _namespace_group above.
    ns = LaunchConfiguration('ns').perform(context).strip().strip('/')
    if ns:
        print(f'[seq] namespace → /{ns}  (isolated from live bringup)')

    if view == 'exo':
        actions = _setup_exo(context, seq, intr, pkg, params, model,
                             autostart, loop_flag, ns)
    else:
        actions = _setup_hand(context, seq, intr, pkg, params, model,
                              common_params, autostart, loop_flag, ns)

    return [_namespace_group(ns, actions)] if ns else actions


def _setup_exo(context, seq, intr, pkg, params, model, autostart, loop_flag,
               ns):
    """Exo view: ArUco world calibration via world_origin_node (existing approach)."""
    player = Node(
        package='recode_sequence', executable='sequence_player_node',
        name='sequence_player_node', output='screen',
        parameters=[{
            'sequence_dir': seq,
            'exo_color_topic': _PIPE_COLOR,
            'exo_depth_topic': _PIPE_DEPTH,
            'exo_info_topic': _PIPE_INFO,
            'exo_frame': 'camera_color_optical_frame',
            'hand_color_topic': _IDLE['hand']['c'],
            'hand_depth_topic': _IDLE['hand']['d'],
            'hand_info_topic': _IDLE['hand']['i'],
            'hand_frame': _IDLE['hand']['f'],
            'autostart': autostart,
            'loop': loop_flag,
        }])

    pipeline = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg, 'launch', 'digital_twin.launch.py')),
        launch_arguments={
            'intrinsics': intr,
            'params': params,
            'yolo_model': model,
            'rviz': LaunchConfiguration('rviz'),
            'control_panel': 'true',
        }.items())

    playback_ctrl = Node(
        package='recode_sequence', executable='playback_control',
        name='playback_control', output='screen',
        parameters=[{'player_ns': f'/{ns}/sequence_player_node'}] if ns
        else [])

    return [player, pipeline, playback_ctrl]


def _setup_hand(context, seq, intr, pkg, params, model, common_params,
                autostart, loop_flag, ns):
    """Hand view: FK-based world transform (joint_states → RSP → handeye_tf)."""
    try:
        get_package_share_directory('dsr_description2')
    except Exception:
        raise RuntimeError(
            "dsr_description2 not found — the m0609 URDF (joint FK for "
            "the hand camera) needs the doosan workspace:\n"
            "    source ~/ros2_ws/install/setup.bash")

    player = Node(
        package='recode_sequence', executable='sequence_player_node',
        name='sequence_player_node', output='screen',
        parameters=[{
            'sequence_dir': seq,
            'hand_color_topic': _PIPE_COLOR,
            'hand_depth_topic': _PIPE_DEPTH,
            'hand_info_topic': _PIPE_INFO,
            'hand_frame': 'hand_color_optical_frame',
            'exo_color_topic': _IDLE['exo']['c'],
            'exo_depth_topic': _IDLE['exo']['d'],
            'exo_info_topic': _IDLE['exo']['i'],
            'exo_frame': _IDLE['exo']['f'],
            'joint_states_topic': '/joint_states',
            'autostart': autostart,
            'loop': loop_flag,
            # Slow the replay (Hz; 0 = recording rate) so the raw image stream
            # doesn't saturate DDS and delay /tf — handeye/point_cloud look up
            # base_link→link_6 at the image stamp and a lagging /tf breaks it.
            'rate_hz': float(
                LaunchConfiguration('playback_rate').perform(context) or 0),
        }])

    robot_description = Command([
        PathJoinSubstitution([FindExecutable(name='xacro')]), ' ',
        PathJoinSubstitution([FindPackageShare('dsr_description2'),
                              'xacro', LaunchConfiguration('model')]),
        '.urdf.xacro',
        ' color:=', LaunchConfiguration('color'),
        ' name:=', LaunchConfiguration('name'),
        ' host:=127.0.0.1 port:=12345 mode:=virtual',
        ' rt_host:=127.0.0.1 update_rate:=100',
        ' model:=', LaunchConfiguration('model'),
    ])
    rsp = Node(
        package='robot_state_publisher', executable='robot_state_publisher',
        name='robot_state_publisher', output='screen',
        parameters=[{'robot_description': ParameterValue(
            robot_description, value_type=str)}])

    # Runtime hand-eye calibration via ArUco (replaces the pre-calibrated
    # T_gripper2camera.npy). world_origin_node in 'handeye_aruco' mode sees
    # the SAME marker as exo, looks up base_link→link_6 via tf2 at each
    # ArUco sample, and publishes:
    #   world      → base_link            (identity)
    #   link_6     → hand_color_optical_frame  (computed hand-eye)
    # NOTE: these MUST go through a node-specific override file, not an inline
    # dict.  params.yaml pins `world_origin_node: world_origin_mode: aruco`
    # (and aruco_timeout_then_floor: true) in a node-specific section, which
    # ROS 2 ranks ABOVE any wildcard inline dict launch_ros serialises — so an
    # inline {'world_origin_mode': 'handeye_aruco'} silently loses and the node
    # boots in exo `aruco` mode (camera frozen to world, never follows link_6).
    # See _write_node_param_override docstring.
    handeye = Node(
        package='depth_digital_twin', executable='world_origin_node',
        name='world_origin_node', output='screen',
        parameters=common_params + [_write_node_param_override(
            'world_origin_node', {
                'world_origin_mode': 'handeye_aruco',
                'color_topic': _PIPE_COLOR,
                'camera_frame': 'hand_color_optical_frame',
                'world_frame': 'world',
                'base_frame': 'base_link',
                'ee_frame': 'link_6',
                # Floor-plane fallback doesn't make sense for the hand camera —
                # without an ArUco view we have no way to anchor world.
                'aruco_timeout_then_floor': False,
            })])

    # See _write_node_param_override docstring — node-specific yaml beats
    # wildcard inline dict, so the override has to be its own --params-file
    # with the matching node-namespace structure.
    det_params = list(common_params)
    if model:
        det_params.append(_write_node_param_override(
            'detection_node', {'model': model}))
    detection = Node(
        package='depth_digital_twin', executable='detection_node',
        name='detection_node', output='screen',
        parameters=det_params)

    point_cloud = Node(
        package='depth_digital_twin', executable='point_cloud_node',
        name='point_cloud_node', output='screen',
        parameters=common_params + [{
            'rgb_topic': _PIPE_COLOR,
            'depth_topic': _PIPE_DEPTH,
            'camera_frame': 'hand_color_optical_frame',
            'world_frame': 'world',
            'aruco_overlay': False,
        }])

    rviz = Node(
        package='rviz2', executable='rviz2', name='rviz2', output='log',
        arguments=['-d', os.path.join(pkg, 'rviz', 'digital_twin.rviz')],
        condition=IfCondition(LaunchConfiguration('rviz')))

    playback_ctrl = Node(
        package='recode_sequence', executable='playback_control',
        name='playback_control', output='screen',
        parameters=[{'player_ns': f'/{ns}/sequence_player_node'}] if ns
        else [])

    # Redetect ArUco popup — same UI as exo (digital_twin.launch.py wires it
    # by default).  Calls /world_origin_node/redetect (std_srvs/Trigger);
    # handeye_aruco mode clears both aruco_samples and the paired
    # handeye_link6_samples, re-subscribes to color, and recomputes.
    world_origin_ctrl = Node(
        package='depth_digital_twin', executable='world_origin_control',
        name='world_origin_control', output='screen')

    return [player, rsp, handeye, detection, point_cloud, rviz,
            playback_ctrl, world_origin_ctrl]


def generate_launch_description() -> LaunchDescription:
    pkg = get_package_share_directory('depth_digital_twin')
    return LaunchDescription([
        DeclareLaunchArgument('sequence', default_value='',
                              description='Absolute path to a sequence dir'),
        DeclareLaunchArgument('view', default_value='exo',
                              description='exo | hand — camera into pipeline'),
        DeclareLaunchArgument('yolo_model', default_value='',
                              description='Override detection_node.model'),
        DeclareLaunchArgument('handeye_npy', default_value=_HANDEYE_DEFAULT,
                              description='T_gripper2camera.npy (view:=hand)'),
        DeclareLaunchArgument('model', default_value='m0609',
                              description='Doosan URDF model (view:=hand)'),
        DeclareLaunchArgument('color', default_value='white',
                              description='URDF color (view:=hand)'),
        DeclareLaunchArgument('name', default_value='dsr01',
                              description='Robot instance name (view:=hand)'),
        DeclareLaunchArgument('loop', default_value='false'),
        DeclareLaunchArgument('autostart', default_value='true'),
        DeclareLaunchArgument(
            'playback_rate', default_value='0',
            description='Replay rate in Hz (0 = recording rate). Lower it '
                        '(e.g. 5) if the raw image stream saturates DDS and '
                        '/tf arrives late, breaking the image-stamp FK lookup '
                        'in handeye_aruco / point_cloud.'),
        DeclareLaunchArgument(
            'ns', default_value='',
            description='Empty = root namespace (current behaviour). Set e.g. '
                        'ns:=record to isolate the whole replay stack under '
                        '/record (joint_states, TF, camera, /digital_twin/*) so '
                        'it can run alongside a live bringup without conflicts.'),
        DeclareLaunchArgument(
            'params',
            default_value=os.path.join(pkg, 'config', 'params.yaml')),
        DeclareLaunchArgument('rviz', default_value='true'),
        OpaqueFunction(function=_setup),
    ])
