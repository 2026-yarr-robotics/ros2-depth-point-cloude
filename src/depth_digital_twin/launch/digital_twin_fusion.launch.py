"""Phase 2b — dual-camera (exo + hand) fusion, LIVE or RECORD.

Two parallel pipelines (exo + hand) feed `cup_fusion_node` in a common `world`
frame. The SAME perception nodes run in both modes — only the TRAJECTORY +
camera source differ. The `ns` arg picks the mode (the user's rule:
"no ns + bringup running ⇒ live; ns set ⇒ use the recorded trajectory"):

  ns EMPTY + NO sequence  → LIVE   (default): defer the trajectory to the real
      robot — `robot_pose_bridge_node` mirrors `/dsr01/joint_states`→`/joint_states`,
      our robot_state_publisher does the m0609 FK, perception consumes the LIVE
      `/camera_exo|hand/*` feeds. NO sequence_player. Intrinsics come from the
      `intr_exo:=`/`intr_hand:=` static YAMLs (no recording → no meta.json).
      Run dsr_bringup2 + the live cameras first.

  ns EMPTY + sequence:=…  → REPLAY at root (legacy standalone playback). Do NOT
      run dsr_bringup2 at the same time (its /joint_states + /tf would fight the
      replay's).

  ns:=record sequence:=…  → REPLAY isolated under /record (joint_states, TF,
      camera, /digital_twin/*, the fusion services). Coexists with a live stack
      at root without conflict (see _namespace_group).

Frame / TF tree (identical in both modes — only the /joint_states SOURCE moves)
  exo_color_optical_frame ──(world_origin_node_exo, ArUco)──► world
  world ──(robot_state_publisher m0609, world_fixed=identity)──► base_link
  base_link ──(URDF FK from /joint_states)──► … ──► link_6
  link_6 ──(world_origin_node_hand, handeye ArUco)──► hand_color_optical_frame

Prerequisites (THREE workspaces sourced — dsr_description2 provides the m0609
URDF used for the joint FK):
  source ~/ros2_ws/install/setup.bash
  source ~/Projects/ros2-recode-sequence/install/setup.bash
  source ~/Projects/ros2-depth-point-cloude/install/setup.bash

Usage:
  # LIVE (verify scan-lock motion against the real robot): bringup + cameras up,
  ros2 launch depth_digital_twin digital_twin_fusion.launch.py \\
      intr_exo:=/abs/exo_intrinsics.yaml intr_hand:=/abs/hand_intrinsics.yaml
  # RECORD alongside live:
  ros2 launch depth_digital_twin digital_twin_fusion.launch.py \\
      ns:=record sequence:=/abs/record_sequence/0010

Args:
  ns          : '' (default) ⇒ LIVE (or root replay if sequence given);
                set (e.g. record) ⇒ namespaced replay alongside a live stack.
  sequence    : recorded sequence dir — REQUIRED for replay, ignored for live.
  intr_exo    : LIVE exo intrinsics YAML (default: package config/intrinsics.yaml)
  intr_hand   : LIVE hand intrinsics YAML (default: package config/intrinsics.yaml)
  exo_cam_ns  : exo camera topic namespace (default /camera_exo — the replay
                sequence_player layout). LIVE feeds that publish elsewhere
                (cameras_only.launch.py / Isaac sim → /exo/exo) pass their ns
                here; every exo subscriber AND the replay player follow it.
  hand_cam_ns : hand camera topic namespace (default /camera_hand; live
                RealSense/Isaac → /hand/hand).
  world_marker_timeout_s : override world_origin_node's ArUco timeout (s).
                Empty (default) keeps the params.yaml value. Slow-to-start
                camera sources (Isaac Sim) need a large value or the exo
                origin falls back to the floor plane fit.
  dsr_joint_topic : LIVE source joints bridged to /joint_states (default
                    /dsr01/joint_states)
  with_pose_bridge: LIVE — run robot_pose_bridge_node (default true)
  with_rsp    : run the m0609 robot_state_publisher (default true). Set false
                when a dsr bringup already broadcasts the identical chain on
                the global /tf (duplicate-stamp TF spam otherwise).
  loop        : replay loop (default false)        autostart : play now (default true)
  params      : pipeline params.yaml (default: package config)
  rviz        : launch RViz2 overlay (default true)
"""
import glob
import json
import os
import re

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, GroupAction, OpaqueFunction)
from launch.conditions import IfCondition
from launch.substitutions import (Command, FindExecutable,
                                  LaunchConfiguration, PathJoinSubstitution)
from launch_ros.actions import Node, PushRosNamespace, SetRemap
from launch_ros.descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

_HANDEYE_DEFAULT = ('/home/eunwoosong/Projects/ros2-recode-sequence/'
                    'src/recode_sequence/config/T_gripper2camera.npy')

# Absolute names the replay stack pub/subs that ALSO exist in a live bringup.
# When `ns:=<name>` is set, each is redirected to /<ns>/… so a recorded replay
# runs next to the real robot without fighting over /joint_states, the TF tree,
# the camera feeds, the /digital_twin/* pipeline, or the fusion services.
# (PushRosNamespace moves node names + relative names; these absolute ones —
# hardcoded here and in params.yaml — ignore the namespace, so they need
# explicit remaps. SetRemap also remaps the service NAMES the panel/skill_manager
# build from params, so the namespaced servers are reached.)
# Camera topics are added per-launch from the exo_cam_ns/hand_cam_ns args —
# see _ns_remap_names().
_NS_REMAP_NAMES = (
    '/joint_states', '/tf', '/tf_static',
    '/digital_twin/detections_exo', '/digital_twin/detection_debug_exo',
    '/digital_twin/detections_hand', '/digital_twin/detection_debug_hand',
    '/digital_twin/cups_exo', '/digital_twin/cups_hand',
    '/digital_twin/cup_obs_exo', '/digital_twin/cup_obs_hand',
    '/digital_twin/rim_debug_exo', '/digital_twin/rim_debug_hand',
    '/digital_twin/box_debug_exo', '/digital_twin/box_debug_hand',
    '/digital_twin/depth_debug_exo', '/digital_twin/depth_debug_hand',
    '/digital_twin/boxes_rim_dbg', '/digital_twin/fusion_health',
    '/digital_twin/points_exo', '/digital_twin/points_hand',
    '/digital_twin/dbg_boxes_exo', '/digital_twin/dbg_boxes_hand',
    '/cup_fusion_node/capture_scan_now',
    '/handeye_markers', '/handeye_camera_pose_in_base',
    '/vision/cups_on_table', '/stack_track_ids',
    '/cup_fusion_node/reset_bias',
    '/digital_twin/boxes',
    '/world_origin_node_exo/redetect', '/world_origin_node_hand/redetect',
    '/cup_fusion_node/clear_scan', '/cup_fusion_node/set_parameters',
    # The panel's producer tuning sliders call these absolute param services.
    '/point_cloud_node_exo/set_parameters',
    '/point_cloud_node_hand/set_parameters',
)


def _ns_remap_names(exo_ns: str, hand_ns: str) -> tuple:
    """Full absolute-name collision list: the static names above plus the
    camera topics under the launch-selected exo/hand namespaces."""
    cams = tuple(
        f'{ns}/{leaf}' for ns in (exo_ns, hand_ns)
        for leaf in ('color/image_raw', 'aligned_depth_to_color/image_raw',
                     'color/camera_info'))
    return cams + _NS_REMAP_NAMES


def _namespace_group(ns: str, actions: list, remap_names: tuple):
    """Wrap every replay action under `/<ns>`: push the namespace (node names +
    relative names) and remap the absolute colliding names above into it."""
    remaps = [SetRemap(src=n, dst=f'/{ns}{n}') for n in remap_names]
    return GroupAction([PushRosNamespace(ns), *remaps, *actions])


_TUNING_RE = re.compile(r'^params_\d{6}_\d{6}\.yaml$')


def _latest_tuning(cfg_dir: str):
    """Newest panel-saved tuning snapshot (params_YYMMDD_HHMMSS.yaml) in
    cfg_dir, or None. Matches ONLY the timestamped name so params.yaml and
    params_back.yaml (which sorts AFTER digits!) are never picked. Snapshots
    are PARTIAL — the caller must OVERLAY them on params.yaml, not replace it."""
    snaps = [f for f in glob.glob(os.path.join(cfg_dir, 'params_*.yaml'))
             if _TUNING_RE.match(os.path.basename(f))]
    return max(snaps, key=os.path.basename) if snaps else None


def _export_intrinsics(seq_dir: str, view: str) -> str:
    with open(os.path.join(seq_dir, 'meta.json')) as f:
        meta = json.load(f)
    cam = (meta.get('cameras') or {}).get(view) or {}
    K = cam.get('K') or []
    if len(K) != 9 or not cam.get('width'):
        raise RuntimeError(
            f'sequence {seq_dir}: no {view} intrinsics in meta.json')
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
    p = os.path.join(seq_dir, f'{view}_intrinsics.yaml')
    with open(p, 'w') as f:
        yaml.safe_dump(out, f, sort_keys=False)
    return p


def _setup(context, *_, **__):
    pkg = get_package_share_directory('depth_digital_twin')
    ns = LaunchConfiguration('ns').perform(context).strip().strip('/')
    seq = LaunchConfiguration('sequence').perform(context)
    seq_valid = bool(seq and os.path.isdir(seq))

    # Camera topic namespaces: every exo/hand subscriber below (and the replay
    # publisher) builds its topics from these, so one arg rewires the whole
    # pipeline to whatever the camera source publishes (replay /camera_exo,
    # live RealSense or Isaac sim /exo/exo + /hand/hand).
    exo_ns = '/' + LaunchConfiguration(
        'exo_cam_ns').perform(context).strip().strip('/')
    hand_ns = '/' + LaunchConfiguration(
        'hand_cam_ns').perform(context).strip().strip('/')
    exo_color = f'{exo_ns}/color/image_raw'
    exo_depth = f'{exo_ns}/aligned_depth_to_color/image_raw'
    exo_info = f'{exo_ns}/color/camera_info'
    hand_color = f'{hand_ns}/color/image_raw'
    hand_depth = f'{hand_ns}/aligned_depth_to_color/image_raw'
    hand_info = f'{hand_ns}/color/camera_info'
    # LIVE = no namespace AND no sequence → defer the trajectory to the real
    # robot (dsr_bringup2). A namespace, or a sequence, means REPLAY.
    live = (not ns) and (not seq_valid)

    # The m0609 URDF FK (joints → link_6) is needed in BOTH modes for the
    # robot_state_publisher below.
    try:
        get_package_share_directory('dsr_description2')
    except Exception:
        raise RuntimeError(
            "dsr_description2 not found — the m0609 URDF (joint FK for the "
            "hand camera) needs the doosan workspace:\n"
            "    source ~/ros2_ws/install/setup.bash")

    if live:
        # Intrinsics: no recording → take them from static YAML args.
        intr_exo = os.path.abspath(
            LaunchConfiguration('intr_exo').perform(context))
        intr_hand = os.path.abspath(
            LaunchConfiguration('intr_hand').perform(context))
        for v, p in (('exo', intr_exo), ('hand', intr_hand)):
            if not os.path.isfile(p):
                raise RuntimeError(
                    f'LIVE mode {v} intrinsics not found: {p}\n'
                    f'    pass intr_{v}:=/abs/path/to/{v}_intrinsics.yaml')
        print(f'[fusion] MODE=LIVE  joints←{LaunchConfiguration("dsr_joint_topic").perform(context)}'
              f'  intr_exo={intr_exo}  intr_hand={intr_hand}')
    else:
        if not seq_valid:
            raise RuntimeError(
                'REPLAY needs sequence:=/abs/path/to/record_sequence/NNNN '
                f'(got {seq!r}). For LIVE, omit both ns and sequence and run '
                'dsr_bringup2 + the live cameras.')
        seq = os.path.abspath(seq)
        intr_exo = _export_intrinsics(seq, 'exo')
        intr_hand = _export_intrinsics(seq, 'hand')
        print(f'[fusion] MODE=REPLAY  sequence={seq}'
              f'{("  namespace=/" + ns) if ns else "  (root)"}')

    params = LaunchConfiguration('params').perform(context)
    # Optionally OVERLAY the newest panel-saved tuning snapshot on top of
    # params.yaml (snapshots are partial → later files win, base stays intact).
    base_params = [params]
    if LaunchConfiguration('load_latest_tuning').perform(context) == 'true':
        latest = _latest_tuning(os.path.dirname(params))
        if latest:
            base_params.append(latest)
            print(f'[fusion] tuning overlay → {latest}')
        else:
            print('[fusion] load_latest_tuning: no params_*.yaml snapshot found')
    # --release: propagate release_mode to every node via the shared base
    # (detection/point_cloud read it; others harmlessly ignore the override,
    # same as the params.yaml /**: wildcard already does).
    release = LaunchConfiguration('release').perform(context).strip().lower() \
        in ('true', '1')
    base_params.append({'release_mode': release})
    with open(params) as f:
        _y = yaml.safe_load(f) or {}
    dn = (_y.get('detection_node') or {}).get('ros__parameters') or {}
    # ArUco / hand-eye params live under `world_origin_node:`, which would NOT
    # apply to the renamed _exo/_hand nodes — re-pass them as an inline dict.
    wo = (_y.get('world_origin_node') or {}).get('ros__parameters') or {}
    # Optional ArUco-timeout override (slow-starting camera sources, e.g.
    # Isaac Sim, need far more than the params.yaml 15 s before the exo
    # origin falls back to the floor plane fit).
    wm_timeout = LaunchConfiguration(
        'world_marker_timeout_s').perform(context).strip()
    if wm_timeout:
        wo = dict(wo, world_marker_timeout_s=float(wm_timeout))
    # Cup model lives under point_cloud_node: — pass the geometry to
    # cup_fusion_node so the fused fit/frustum use the SAME Speed Stack cup.
    pn = (_y.get('point_cloud_node') or {}).get('ros__parameters') or {}
    # The params.yaml `point_cloud_node:` section does NOT match the RENAMED
    # producer nodes (point_cloud_node_exo/_hand), so its window_period_s=0.1
    # (and approx_sync_slop / depth filters) were silently ignored → producers
    # ran at the 0.5s code default (~1.5 Hz), which flickered /points. Pass `pn`
    # explicitly, positioned BEFORE the tuning snapshot so per-camera panel
    # tuning (point_cloud_node_exo/_hand sections) still wins.
    prod_base = [params, pn, *base_params[1:]]
    cup_geom = {k: pn[k] for k in (
        'cup_top_diameter_m', 'cup_bottom_diameter_m', 'cup_height_m',
        'cup_fit_residual_max', 'cup_polygon_segments', 'cup_class_names',
        'box_standing_ratio', 'box_min_elongation') if k in pn}
    model_exo = dn.get('model_exo') or dn.get('model') or ''
    model_hand = dn.get('model_hand') or dn.get('model') or ''
    # Like `wo`/`pn` above: the params.yaml `detection_node:` section does
    # NOT match the RENAMED detection_node_exo/_hand, so its class filter /
    # confidence / device silently fell back to code defaults (the 3-class
    # hand model then leaked its extra class into the producers). Re-pass
    # the runtime keys inline; model/imgsz/topics stay per-node below.
    det_common = {k: dn[k] for k in
                  ('target_classes', 'confidence', 'device') if k in dn}
    imgsz = int(LaunchConfiguration('imgsz').perform(context))

    EXO_F = 'exo_color_optical_frame'
    HAND_F = 'hand_color_optical_frame'

    # ---- trajectory source: REPLAY player vs LIVE joint bridge ----------
    if live:
        traj = [Node(
            package='depth_digital_twin', executable='robot_pose_bridge_node',
            name='robot_pose_bridge', output='screen',
            condition=IfCondition(LaunchConfiguration('with_pose_bridge')),
            parameters=[{
                'input_topic':
                    LaunchConfiguration('dsr_joint_topic').perform(context),
                'output_topic': '/joint_states',
            }])]
    else:
        traj = [Node(
            package='recode_sequence', executable='sequence_player_node',
            name='sequence_player_node', output='screen',
            parameters=[{
                'sequence_dir': seq,
                'exo_color_topic': exo_color,
                'exo_depth_topic': exo_depth,
                'exo_info_topic': exo_info,
                'hand_color_topic': hand_color,
                'hand_depth_topic': hand_depth,
                'hand_info_topic': hand_info,
                'exo_frame': EXO_F, 'hand_frame': HAND_F,
                'joint_states_topic': '/joint_states',
                'autostart':
                    LaunchConfiguration('autostart').perform(context) == 'true',
                'loop': LaunchConfiguration('loop').perform(context) == 'true',
            }]),
            Node(package='recode_sequence', executable='playback_control',
                 name='playback_control', output='screen')]

    # ---- m0609 URDF FK: world → base_link → … → link_6 (BOTH modes) -----
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
    # with_rsp:=false skips it: when a dsr bringup is ALREADY broadcasting
    # the identical world→base_link→…→link_6 chain on the global /tf (same
    # m0609 xacro), a second publisher emits duplicate-stamp transforms
    # (TF_REPEATED_DATA spam in every listener) and the pose-bridge's
    # zero-joint idle fallback would flap the chain to the home pose on a
    # joint-stream stall. Keep true for replay/standalone (no bringup).
    rsp = Node(
        package='robot_state_publisher', executable='robot_state_publisher',
        name='robot_state_publisher', output='screen',
        condition=IfCondition(LaunchConfiguration('with_rsp')),
        parameters=[{'robot_description': ParameterValue(
            robot_description, value_type=str)}])

    # ---- exo: ArUco world origin + detection + point cloud (producer) ----
    common_exo = [*base_params, {'intrinsics_path': intr_exo}]
    world_origin_exo = Node(
        package='depth_digital_twin', executable='world_origin_node',
        name='world_origin_node_exo', output='screen',
        parameters=common_exo + [wo, {
            'world_origin_mode': 'aruco',
            'color_topic': exo_color,
            'depth_topic': exo_depth,
            'camera_frame': EXO_F,
            'world_frame': 'world',
        }])
    common_hand = [*base_params, {'intrinsics_path': intr_hand}]
    world_origin_hand = Node(
        package='depth_digital_twin', executable='world_origin_node',
        name='world_origin_node_hand', output='screen',
        parameters=common_hand + [wo, {
            'world_origin_mode': 'handeye_aruco',
            'color_topic': hand_color,
            'camera_frame': HAND_F,
            'world_frame': 'world',
            'base_frame': 'base_link',
            'ee_frame': 'link_6',
            'aruco_timeout_then_floor': False,
        }])
    det_exo = Node(
        package='depth_digital_twin', executable='detection_node',
        name='detection_node_exo', output='screen',
        parameters=common_exo + [det_common, {
            'image_topic': exo_color,
            'detections_topic': '/digital_twin/detections_exo',
            'debug_topic': '/digital_twin/detection_debug_exo',
            'model': model_exo,
            'imgsz': imgsz,
        }])
    pc_exo = Node(
        package='depth_digital_twin', executable='point_cloud_node',
        name='point_cloud_node_exo', output='screen',
        parameters=prod_base + [{'intrinsics_path': intr_exo}, {
            'rgb_topic': exo_color,
            'depth_topic': exo_depth,
            'detections_topic': '/digital_twin/detections_exo',
            'camera_frame': EXO_F,
            'world_frame': 'world',
            'role': 'producer',
            'camera_name': 'exo',
            'world_clouds_topic': '/digital_twin/cups_exo',
            'cup_obs_topic': '/digital_twin/cup_obs_exo',
            'rim_debug_topic': '/digital_twin/rim_debug_exo',
            # split the remaining shared-default debug topics too — two
            # cameras interleaving on one Image topic flickers violently
            'box_debug_topic': '/digital_twin/box_debug_exo',
            'depth_debug_topic': '/digital_twin/depth_debug_exo',
            # rim is the upright measurement (cup_fusion fit_source=rim) —
            # in rim mode the fusion never FITS upright clouds, but they
            # still feed the /digital_twin/points DISPLAY (RViz/debug).
            # Set false to also skip building them (CPU saving).
            'upright_clouds': True,
        }])
    det_hand = Node(
        package='depth_digital_twin', executable='detection_node',
        name='detection_node_hand', output='screen',
        parameters=common_hand + [det_common, {
            'image_topic': hand_color,
            'detections_topic': '/digital_twin/detections_hand',
            'debug_topic': '/digital_twin/detection_debug_hand',
            'model': model_hand,
            'imgsz': imgsz,
        }])
    pc_hand = Node(
        package='depth_digital_twin', executable='point_cloud_node',
        name='point_cloud_node_hand', output='screen',
        parameters=prod_base + [{'intrinsics_path': intr_hand}, {
            'rgb_topic': hand_color,
            'depth_topic': hand_depth,
            'detections_topic': '/digital_twin/detections_hand',
            'camera_frame': HAND_F,
            'world_frame': 'world',
            'role': 'producer',
            'camera_name': 'hand',
            'world_clouds_topic': '/digital_twin/cups_hand',
            'cup_obs_topic': '/digital_twin/cup_obs_hand',
            'rim_debug_topic': '/digital_twin/rim_debug_hand',
            'box_debug_topic': '/digital_twin/box_debug_hand',
            'depth_debug_topic': '/digital_twin/depth_debug_hand',
            'upright_clouds': True,
            'hand_motion_gating': True,
            'joint_states_topic': '/joint_states',
            'aruco_overlay': False,
        }])

    # ---- fusion: associate + merge + fit + KF (+ scan&lock) → /boxes,/points
    fusion = Node(
        package='depth_digital_twin', executable='cup_fusion_node',
        name='cup_fusion_node', output='screen',
        parameters=[*base_params, dict({
            'exo_clouds_topic': '/digital_twin/cups_exo',
            'hand_clouds_topic': '/digital_twin/cups_hand',
            'exo_obs_topic': '/digital_twin/cup_obs_exo',
            'hand_obs_topic': '/digital_twin/cup_obs_hand',
            'boxes_topic': '/digital_twin/boxes',
            'world_frame': 'world',
            # scan-lock reads /joint_states (remapped under /<ns> in replay).
            'scan_joint_states_topic': '/joint_states',
        }, **cup_geom)])

    panel = Node(
        package='depth_digital_twin', executable='digital_twin_panel',
        name='digital_twin_panel', output='screen',
        parameters=[{
            # This launch uses the exo_cam_ns/hand_cam_ns feeds and the
            # _exo/_hand world-origin nodes. Set image topics EXPLICITLY (do
            # not rely on the panel's code defaults, which target start.sh's
            # /exo/exo, /hand/hand split-launch topology).
            'exo_color_topic': exo_color,
            'exo_depth_topic': exo_depth,
            # 3D pane = the rim-fit overlay (observed contour green, fitted
            # silhouette cyan, depth init red, per-cup iou/rms/b/cov text) —
            # the actual live measurement. Raw YOLO boxes remain on
            # /digital_twin/detection_debug_* if needed.
            'exo_debug_topic': '/digital_twin/rim_debug_exo',
            'hand_color_topic': hand_color,
            'hand_depth_topic': hand_depth,
            'hand_debug_topic': '/digital_twin/rim_debug_hand',
            'exo_redetect_srv': '/world_origin_node_exo/redetect',
            'hand_redetect_srv': '/world_origin_node_hand/redetect',
        }])

    rviz = Node(
        package='rviz2', executable='rviz2', name='rviz2', output='log',
        arguments=['-d', os.path.join(pkg, 'rviz', 'fusion.rviz')],
        condition=IfCondition(LaunchConfiguration('rviz')))

    entities = traj + [
        rsp, world_origin_exo, world_origin_hand,
        det_exo, pc_exo, det_hand, pc_hand,
        fusion, panel, rviz]

    # ns set ⇒ isolate the whole replay stack under /<ns> (coexist with live).
    if ns:
        return [_namespace_group(ns, entities, _ns_remap_names(exo_ns, hand_ns))]
    return entities


def generate_launch_description() -> LaunchDescription:
    pkg = get_package_share_directory('depth_digital_twin')
    return LaunchDescription([
        DeclareLaunchArgument(
            'ns', default_value='',
            description='Empty (default) = LIVE (or root replay if sequence '
                        'given). Set e.g. ns:=record to run a namespaced '
                        'replay alongside a live bringup.'),
        DeclareLaunchArgument(
            'sequence', default_value='',
            description='Recorded sequence dir — REQUIRED for replay, ignored '
                        'for live.'),
        DeclareLaunchArgument(
            'intr_exo',
            default_value=os.path.join(pkg, 'config', 'intrinsics.yaml'),
            description='LIVE exo intrinsics YAML (no meta.json in live mode).'),
        DeclareLaunchArgument(
            'intr_hand',
            default_value=os.path.join(pkg, 'config', 'intrinsics.yaml'),
            description='LIVE hand intrinsics YAML.'),
        DeclareLaunchArgument(
            'exo_cam_ns', default_value='/camera_exo',
            description='Exo camera topic namespace (<ns>/color/image_raw, '
                        '<ns>/aligned_depth_to_color/image_raw). Default = '
                        'replay sequence_player layout; live cameras_only/'
                        'Isaac publish /exo/exo.'),
        DeclareLaunchArgument(
            'hand_cam_ns', default_value='/camera_hand',
            description='Hand camera topic namespace (live RealSense/Isaac '
                        'publish /hand/hand).'),
        DeclareLaunchArgument(
            'world_marker_timeout_s', default_value='',
            description='Override world_origin ArUco timeout in seconds '
                        '(empty = params.yaml). Use a large value for '
                        'slow-starting camera sources (Isaac Sim).'),
        DeclareLaunchArgument(
            'dsr_joint_topic', default_value='/dsr01/joint_states',
            description='LIVE source joints bridged → /joint_states.'),
        DeclareLaunchArgument('with_pose_bridge', default_value='true'),
        DeclareLaunchArgument(
            'with_rsp', default_value='true',
            description='Run the m0609 robot_state_publisher. Set false when '
                        'a dsr bringup already broadcasts the same chain on '
                        'the global /tf (avoids duplicate-stamp TF spam).'),
        DeclareLaunchArgument('handeye_npy', default_value=_HANDEYE_DEFAULT),
        DeclareLaunchArgument('model', default_value='m0609'),
        DeclareLaunchArgument('color', default_value='white'),
        DeclareLaunchArgument('name', default_value='dsr01'),
        DeclareLaunchArgument('loop', default_value='false'),
        DeclareLaunchArgument('autostart', default_value='true'),
        DeclareLaunchArgument(
            'imgsz', default_value='640',
            description='YOLO inference size for BOTH detectors (dual = heavy; '
                        '640 default, set 1280 if GPU allows)'),
        DeclareLaunchArgument(
            'release', default_value='false',
            description='Release mode: every node skips ALL debug-image '
                        'synthesis (detection/box/depth/rim debug topics). '
                        'Measurement outputs unaffected. Same as env '
                        'DPC_RELEASE=1.'),
        DeclareLaunchArgument(
            'params',
            default_value=os.path.join(pkg, 'config', 'params.yaml')),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument(
            'load_latest_tuning', default_value='true',
            description='Overlay the newest panel-saved params_<ts>.yaml '
                        'snapshot on top of params.yaml at startup '
                        '(false = base params.yaml only).'),
        OpaqueFunction(function=_setup),
    ])
