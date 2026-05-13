"""aruco_calibrate.launch.py — bring up the Doosan stack + the **capture** tool.

The capture CLI consumes `/dsr01/system/get_current_posx` (or its
`aux_control` mirror) — it does not produce it. So either you run
`dsr_bringup2` separately, or use this launch to start both at once.

Two-step workflow:
  1. Capture (this launch): press `s` repeatedly to save image+posx pairs
     under `output_dir`. Quit with `q`.
  2. Solve (offline): run `ros2 run depth_digital_twin aruco_handeye ...`
     pointing at the same directory. Produces T_exo2base.npy / T_hand2base.npy.

Prerequisites:
  source ~/ros2_ws/install/setup.bash      # provides dsr_bringup2 + dsr_msgs2 + dsr_description2
  source ~/Projects/ros2-depth-point-cloude/install/setup.bash

RealSense must be running in a separate terminal (this launch keeps it
out of scope so it can be reused with any camera bringup):

  ros2 launch realsense2_camera rs_align_depth_launch.py \\
      depth_module.depth_profile:=640x480x30 \\
      rgb_camera.color_profile:=640x480x30 \\
      initial_reset:=true align_depth.enable:=true

Usage:
  # Virtual robot (emulator) — fastest path, no hardware:
  ros2 launch depth_digital_twin aruco_calibrate.launch.py

  # Real robot:
  ros2 launch depth_digital_twin aruco_calibrate.launch.py \\
      bringup_mode:=real bringup_host:=192.168.137.100

  # If bringup is already running elsewhere, just run the tool:
  ros2 launch depth_digital_twin aruco_calibrate.launch.py \\
      with_bringup:=false startup_delay_s:=0.0

Custom marker / output:
  ros2 launch depth_digital_twin aruco_calibrate.launch.py \\
      marker_length:=0.04 marker_id:=0 dict:=4X4_50 \\
      output_dir:=$PWD/data/aruco

Argument naming (deliberately split to avoid ambiguity):
  - `bringup_mode` : virtual | real    — Doosan robot stack
  - capture has no `mode` arg; eye-to-hand vs eye-in-hand is decided at
    solve time via `aruco_handeye --mode exo|hand`.
"""
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, IncludeLaunchDescription, TimerAction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    pkg_dt = FindPackageShare('depth_digital_twin')

    # ----- Calibration tool args (forwarded to aruco_calibrate CLI) -----
    intrinsics_arg = DeclareLaunchArgument(
        'intrinsics',
        default_value=PathJoinSubstitution([pkg_dt, 'config', 'intrinsics.yaml']),
        description='Path to intrinsics.yaml (output of `calibrate` CLI)')
    # Capture writes a directory (PNGs + calibrate_data.json), not a .npy.
    # The .npy is produced later by `aruco_handeye` from the same directory.
    output_dir_arg = DeclareLaunchArgument(
        'output_dir',
        default_value='/home/eunwoosong/Projects/ros2-depth-point-cloude/record/exo',
        description='Directory for captured images + calibrate_data.json '
                    '(ingested later by aruco_handeye). '
                    'Default: project-root/record/exo.')
    marker_length_arg = DeclareLaunchArgument(
        'marker_length', default_value='0.05',
        description='ArUco side length in metres (for the live axes overlay)')
    dict_arg = DeclareLaunchArgument(
        'dict', default_value='4X4_50',
        description='ArUco dictionary (e.g. 4X4_50, 5X5_100)')
    marker_id_arg = DeclareLaunchArgument(
        'marker_id', default_value='0',
        description='Marker id to track (default 0; -1 = first detected)')
    pose_source_arg = DeclareLaunchArgument(
        'pose_source', default_value='service',
        description='Where to read robot posx: service (default; DSR '
                    'GetCurrentPosx — same as sample/data_recording.py), '
                    'tf (via /tf, control-authority-independent fallback), '
                    'or manual (stdin prompt).')
    base_frame_arg = DeclareLaunchArgument(
        'base_frame', default_value='base_0',
        description='Base TF frame for pose_source=tf')
    ee_frame_arg = DeclareLaunchArgument(
        'ee_frame', default_value='link6',
        description='End-effector TF frame for pose_source=tf')

    # ----- dsr_bringup2 args -----
    with_bringup_arg = DeclareLaunchArgument(
        'with_bringup', default_value='true',
        description='Auto-start dsr_bringup2_rviz inside this launch. '
                    'Set false if bringup is already running elsewhere.')
    name_arg = DeclareLaunchArgument(
        'name', default_value='dsr01',
        description='Robot namespace (must match dsr_bringup2 default)')
    model_arg = DeclareLaunchArgument(
        'model', default_value='m0609',
        description='Doosan model (m0609, m1013, ...)')
    bringup_mode_arg = DeclareLaunchArgument(
        'bringup_mode', default_value='virtual',
        description='Doosan operation mode: virtual (emulator) | real')
    bringup_host_arg = DeclareLaunchArgument(
        'bringup_host', default_value='127.0.0.1',
        description='Robot IP (only used when bringup_mode:=real)')
    bringup_port_arg = DeclareLaunchArgument(
        'bringup_port', default_value='12345',
        description='Robot service port')
    rt_host_arg = DeclareLaunchArgument(
        'rt_host', default_value='192.168.137.50',
        description='Robot RT IP (real mode only)')
    startup_delay_arg = DeclareLaunchArgument(
        'startup_delay_s', default_value='6.0',
        description='Seconds to wait before starting the tool. With internal '
                    'bringup, gives the emulator + services time to come up. '
                    'Set 0 when --with-bringup:=false.')

    # ----- dsr_bringup2_rviz include (conditional) -----
    bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('dsr_bringup2'),
                'launch', 'dsr_bringup2_rviz.launch.py']),
        ]),
        launch_arguments={
            'name': LaunchConfiguration('name'),
            'model': LaunchConfiguration('model'),
            'mode': LaunchConfiguration('bringup_mode'),
            'host': LaunchConfiguration('bringup_host'),
            'port': LaunchConfiguration('bringup_port'),
            'rt_host': LaunchConfiguration('rt_host'),
        }.items(),
        condition=IfCondition(LaunchConfiguration('with_bringup')),
    )

    # ----- aruco_calibrate CLI exposed as a launch_ros Node -----
    # cv2.imshow window pops up; keystrokes go to cv2, prints go to launch
    # stdout. argparse parses the `arguments` list verbatim.
    aruco_node = Node(
        package='depth_digital_twin',
        executable='aruco_calibrate',
        name='aruco_calibrate',
        output='screen',
        # Force unbuffered Python stdout so the calibration tool's logs (s/c
        # key feedback, save banner) appear in the launch terminal in real
        # time. Without this, ros2 launch keeps stdout in a pipe and Python
        # block-buffers writes — the user only sees logs at process exit.
        additional_env={'PYTHONUNBUFFERED': '1'},
        arguments=[
            '--intrinsics',     LaunchConfiguration('intrinsics'),
            '--output-dir',     LaunchConfiguration('output_dir'),
            '--marker-length',  LaunchConfiguration('marker_length'),
            '--dict',           LaunchConfiguration('dict'),
            '--marker-id',      LaunchConfiguration('marker_id'),
            '--pose-source',    LaunchConfiguration('pose_source'),
            '--base-frame',     LaunchConfiguration('base_frame'),
            '--ee-frame',       LaunchConfiguration('ee_frame'),
        ],
    )
    delayed_aruco = TimerAction(
        period=LaunchConfiguration('startup_delay_s'),
        actions=[aruco_node],
    )

    return LaunchDescription([
        intrinsics_arg, output_dir_arg, marker_length_arg, dict_arg,
        marker_id_arg, pose_source_arg, base_frame_arg, ee_frame_arg,
        with_bringup_arg, name_arg, model_arg,
        bringup_mode_arg, bringup_host_arg, bringup_port_arg, rt_host_arg,
        startup_delay_arg,
        bringup, delayed_aruco,
    ])
