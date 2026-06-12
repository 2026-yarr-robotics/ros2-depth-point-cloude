"""Bring up the two D435i cameras only (exo + hand-eye), nothing else.

Connected device serials are auto-enumerated (``rs-enumerate-devices -s``)
and EVERY node is bound to a DISTINCT serial — never start two unbound
RealSense nodes, or they fight over the same device ("Device or resource
busy" → SIGSEGV).

  • BOUND mode  — config/cameras.yaml has exo_serial + hand_serial:
    /exo and /hand are bound to those serials.
  • IDENTIFY mode — serials not set yet: connected serials are assigned
    to /cam_a and /cam_b (distinct), so `camera_id_tool` can show both
    feeds and let you assign exo/hand → cameras.yaml.

Topics: /<ns>/<ns>/color/image_raw,
        /<ns>/<ns>/aligned_depth_to_color/image_raw,
        /<ns>/<ns>/color/camera_info

The IMU (gyro/accel) and initial_reset are disabled — they are not
needed and were the source of the HID "device busy / permission denied"
crashes with two D435i on one host.
"""
import os
import re
import subprocess

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _connected_serials() -> list[str]:
    """Return serials of connected RealSense devices, in enumeration order."""
    try:
        out = subprocess.run(
            ['rs-enumerate-devices', '-s'],
            capture_output=True, text=True, timeout=15).stdout
    except Exception as e:  # noqa: BLE001
        print(f'[cameras_only] rs-enumerate-devices failed: {e}')
        return []
    serials: list[str] = []
    for line in out.splitlines():
        if 'Serial Number' in line or 'ERROR' in line:
            continue
        # Serial = the long all-digit token (firmware has dots, not 8+ run).
        m = re.findall(r'\b(\d{8,})\b', line)
        if m:
            serials.append(m[0])
    return serials


def _camera_node(namespace: str, serial: str,
                 color_profile: str, depth_profile: str) -> Node:
    return Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        name=namespace,
        namespace=namespace,
        parameters=[{
            'camera_name': namespace,
            'camera_namespace': namespace,
            'serial_no': str(serial),       # plain string, distinct per node
            'enable_color': True,
            'enable_depth': True,
            'enable_infra1': False,
            'enable_infra2': False,
            'enable_gyro': False,           # no IMU — avoids HID busy crash
            'enable_accel': False,
            'enable_sync': False,
            'initial_reset': False,         # reset caused instability w/ 2 cams
            'rgb_camera.color_profile': color_profile,
            'rgb_camera.color_format': 'RGB8',
            'depth_module.depth_profile': depth_profile,
            'align_depth.enable': True,
            'pointcloud.enable': False,
        }],
        output='screen')


def _setup(context, *_, **__):
    cameras_yaml = LaunchConfiguration('cameras_yaml').perform(context)
    color_profile = LaunchConfiguration('color_profile').perform(context)
    depth_profile = LaunchConfiguration('depth_profile').perform(context)
    view = LaunchConfiguration('view').perform(context).strip().lower()
    if view not in ('both', 'exo', 'hand'):
        raise RuntimeError(f"view must be 'both', 'exo', or 'hand' (got {view!r})")

    exo_serial = hand_serial = ''
    if os.path.isfile(cameras_yaml):
        with open(cameras_yaml) as f:
            data = yaml.safe_load(f) or {}
        exo_serial = str(data.get('exo_serial', '') or '')
        hand_serial = str(data.get('hand_serial', '') or '')

    connected = _connected_serials()
    print(f'[cameras_only] connected serials: {connected or "(none)"}')
    if not connected:
        print('[cameras_only] ERROR: no RealSense devices found — '
              'check USB / `rs-enumerate-devices`.')
        return []

    # BOUND mode: both roles configured.
    if exo_serial and hand_serial:
        roles = [('exo', exo_serial), ('hand', hand_serial)]
        if view != 'both':
            roles = [(ns, ser) for ns, ser in roles if ns == view]
        nodes = []
        for ns, ser in roles:
            if ser not in connected:
                print(f'[cameras_only] WARNING: {ns} serial {ser} not '
                      f'connected — skipping {ns}.')
                continue
            nodes.append(_camera_node(ns, ser, color_profile, depth_profile))
        print(f'[cameras_only] BOUND mode  view={view}  '
              f'exo={exo_serial} hand={hand_serial}')
        return nodes

    # IDENTIFY mode: bind connected serials to distinct cam_a / cam_b.
    if len(connected) >= 2:
        print('[cameras_only] IDENTIFY mode — cam_a=%s cam_b=%s. '
              'Run camera_id_tool to assign exo/hand.'
              % (connected[0], connected[1]))
        return [
            _camera_node('cam_a', connected[0],
                         color_profile, depth_profile),
            _camera_node('cam_b', connected[1],
                         color_profile, depth_profile),
        ]
    print('[cameras_only] WARNING: only ONE camera connected '
          f'({connected[0]}). Recording needs TWO. Starting cam_a only.')
    return [_camera_node('cam_a', connected[0],
                         color_profile, depth_profile)]


def generate_launch_description() -> LaunchDescription:
    pkg = get_package_share_directory('recode_sequence')
    return LaunchDescription([
        DeclareLaunchArgument(
            'cameras_yaml',
            default_value=os.path.join(pkg, 'config', 'cameras.yaml'),
            description='Path to cameras.yaml (serial→role mapping)'),
        DeclareLaunchArgument(
            'view', default_value='both',
            description='both | exo | hand — which camera(s) to start'),
        DeclareLaunchArgument(
            'color_profile', default_value='1280x720x30',
            description='RealSense color profile WxHxFPS'),
        DeclareLaunchArgument(
            'depth_profile', default_value='1280x720x30',
            description='RealSense depth profile WxHxFPS'),
        OpaqueFunction(function=_setup),
    ])
