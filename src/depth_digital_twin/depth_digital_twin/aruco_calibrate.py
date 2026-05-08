"""ArUco hand-eye **capture** CLI (online, with robot).

Mirrors the `capture_chessboard` + `calibrate` two-step pattern. This file
only captures: every press of <s> snapshots the current RGB frame *and* the
current robot posx into `<output_dir>/`, the same JSON layout as
`sample/Calibration_Tutorial/data_recording.py`. The actual hand-eye solve
runs offline via `aruco_handeye` (a separate CLI) which reads those files
and produces `T_exo2base.npy` / `T_hand2base.npy`.

Output layout:
  <output_dir>/
    calibrate_data.json   # {"poses": [[x,y,z,rx,ry,rz], ...], "file_name": ["aruco_000.png", ...]}
    aruco_000.png
    aruco_001.png
    ...

Workflow:
  1. Move robot to a varied pose (rotation + translation diversity matters).
  2. The display *freezes* on the latest frame in which ArUco was detected
     — so even when detection flickers off you still see the last good
     view (status badge: `frozen (last detect Nf ago)`).
  3. Press <s> — saves the cached PNG + the current robot posx.
  4. Repeat ≥10 times across the workspace.
  5. <q> / ESC — quit.
  6. Run `ros2 run depth_digital_twin aruco_handeye --data-dir <output_dir> ...`
     to compute and save the calibration matrix.

Robot pose is read via Doosan's `dsr_msgs2/srv/GetCurrentPosx` (pose returned
as [x, y, z, rx, ry, rz] with ZYZ Euler in degrees, translation in mm — the
same convention as `data_recording.py` and `handeye_calibration.py`).

Hot keys:
  s — save current frame + robot posx
  u — undo last save (removes PNG + JSON entry)
  q / ESC — quit (capture is autosaved as you go; nothing extra to flush)

Usage:
  ros2 run depth_digital_twin aruco_calibrate \
      --intrinsics src/depth_digital_twin/config/intrinsics.yaml \
      --output-dir ./data/aruco \
      --marker-length 0.05 --dict 4X4_50 --marker-id 0
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rclpy
import tf2_ros
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

from depth_digital_twin.intrinsics import load_intrinsics


# Re-exported for the offline solver (`aruco_handeye`). Lives here so capture
# and solve share a single source of truth for detector params + maths.
__all__ = (
    'main',
    '_ARUCO_DICT_NAMES',
    '_make_aruco_detector',
    '_detect_marker',
    '_solve_marker_pose',
    '_calibrate_exo',
    '_calibrate_hand',
    '_posx_to_T_base2ee',
    '_zyz_deg_to_R',
)


# ----------------------------------------------------------------------
# ArUco helpers
# ----------------------------------------------------------------------
_ARUCO_DICT_NAMES = {
    '4X4_50': cv2.aruco.DICT_4X4_50,
    '4X4_100': cv2.aruco.DICT_4X4_100,
    '4X4_250': cv2.aruco.DICT_4X4_250,
    '5X5_50': cv2.aruco.DICT_5X5_50,
    '5X5_100': cv2.aruco.DICT_5X5_100,
    '5X5_250': cv2.aruco.DICT_5X5_250,
    '6X6_50': cv2.aruco.DICT_6X6_50,
    '6X6_100': cv2.aruco.DICT_6X6_100,
    '6X6_250': cv2.aruco.DICT_6X6_250,
    '7X7_50': cv2.aruco.DICT_7X7_50,
}


def _make_aruco_detector(dict_name: str):
    if dict_name not in _ARUCO_DICT_NAMES:
        raise ValueError(
            f'Unknown ArUco dict {dict_name!r}; choose from '
            f'{sorted(_ARUCO_DICT_NAMES)}')
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(_ARUCO_DICT_NAMES[dict_name])
    # Use the new ArucoDetector API when available (OpenCV ≥4.7); fall back
    # to the legacy detectMarkers function otherwise.
    if hasattr(aruco, 'ArucoDetector'):
        params = aruco.DetectorParameters()
        # Subpixel corner refinement → corners stop jittering at the pixel
        # boundary, which makes the live overlay (and the saved samples)
        # noticeably more stable.
        if hasattr(aruco, 'CORNER_REFINE_SUBPIX'):
            params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
            params.cornerRefinementWinSize = 5
            params.cornerRefinementMaxIterations = 30
            params.cornerRefinementMinAccuracy = 0.01
        # Wider adaptive-threshold sweep + lower minimum perimeter so the
        # detector doesn't drop out a frame on glare / motion / distance.
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 53
        params.adaptiveThreshWinSizeStep = 10
        params.minMarkerPerimeterRate = 0.02
        return aruco.ArucoDetector(dictionary, params), dictionary
    return None, dictionary


def _detect_marker(detector, dictionary, gray):
    if detector is not None:
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
    return corners, ids


def _solve_marker_pose(corners: np.ndarray, marker_length_m: float,
                       K: np.ndarray, dist: np.ndarray):
    """Pose of marker in camera frame via solvePnP (planar object). Returns
    (R_cam_marker, t_cam_marker[m]) or (None, None)."""
    half = float(marker_length_m) * 0.5
    obj_pts = np.array([
        [-half,  half, 0.0],
        [ half,  half, 0.0],
        [ half, -half, 0.0],
        [-half, -half, 0.0],
    ], dtype=np.float32)
    img_pts = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    if img_pts.shape != (4, 2):
        return None, None
    ok, rvec, tvec = cv2.solvePnP(
        obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None, None
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.reshape(3)


# ----------------------------------------------------------------------
# DSR pose client (lazy / optional)
# ----------------------------------------------------------------------
def _try_import_get_current_posx_service(node: Node):
    """Return a callable (lambda → posx[6]) using the GetCurrentPosx service,
    or None if dsr_msgs2 isn't available / robot isn't running."""
    try:
        from dsr_msgs2.srv import GetCurrentPosx  # type: ignore
    except ImportError:
        node.get_logger().warn(
            'dsr_msgs2 not importable. Source ros2_ws first '
            '(`source ~/ros2_ws/install/setup.bash`) to use live robot pose.')
        return None

    cli = node.create_client(GetCurrentPosx, '/dsr01/system/get_current_posx')
    # Some installations expose the service under aux_control; try both.
    if not cli.wait_for_service(timeout_sec=1.0):
        cli = node.create_client(
            GetCurrentPosx, '/dsr01/aux_control/get_current_posx')
    if not cli.wait_for_service(timeout_sec=1.0):
        node.get_logger().warn(
            'GetCurrentPosx service not available. Is dsr_bringup2 running?')
        return None

    def _read():
        req = GetCurrentPosx.Request()
        req.ref = 0  # DR_BASE
        fut = cli.call_async(req)
        # Don't spin from the caller: a background MultiThreadedExecutor is
        # already spinning the node (see main()). Calling
        # `spin_until_future_complete` from a second thread races with that
        # spinner and trips `IndexError: wait set index too big` inside
        # rclpy.qos_event.is_ready. Poll the future instead — the executor
        # thread will mark it done as soon as the response arrives.
        deadline = time.monotonic() + 2.0
        while not fut.done() and time.monotonic() < deadline:
            time.sleep(0.005)
        if not fut.done() or fut.result() is None:
            return None
        res = fut.result()
        if not res.success or not res.task_pos_info:
            return None
        data = res.task_pos_info[0].data
        return tuple(float(v) for v in data[:6])  # x,y,z,rx,ry,rz
    return _read


# ----------------------------------------------------------------------
# Pose math (matches sample/Calibration_Tutorial conventions)
# ----------------------------------------------------------------------
def _zyz_deg_to_R(rx: float, ry: float, rz: float) -> np.ndarray:
    """Doosan posx convention: ZYZ Euler angles in degrees."""
    from scipy.spatial.transform import Rotation
    return Rotation.from_euler('ZYZ', [rx, ry, rz], degrees=True).as_matrix()


def _posx_to_T_base2ee(posx_mm_deg) -> np.ndarray:
    """[x, y, z (mm), rx, ry, rz (deg)] → T_base2ee (translation in mm).
    Pose of EE expressed in base frame, matches sample handeye_calibration.py.
    """
    x, y, z, rx, ry, rz = posx_mm_deg
    T = np.eye(4)
    T[:3, :3] = _zyz_deg_to_R(rx, ry, rz)
    T[:3, 3] = [x, y, z]
    return T


# ----------------------------------------------------------------------
# Single capture node — owns the image subscription AND the GetCurrentPosx
# service client (when DSR is available). Combining both onto one node — and
# spinning it through a single MultiThreadedExecutor in a background thread
# — is what avoids:
#   * the rosout "Publisher already registered for provided node name"
#     warning, which fires when the launch's `__node:=aruco_calibrate`
#     global remap aliases two separately-named nodes onto the same name; and
#   * the `IndexError: wait set index too big` race that occurs when a second
#     thread calls `rclpy.spin_until_future_complete` on a different node
#     while this thread is already spinning.
# ----------------------------------------------------------------------
class _CaptureNode(Node):
    def __init__(self, topic: str):
        super().__init__('aruco_calibrate')
        self.bridge = CvBridge()
        self.frame: Optional[np.ndarray] = None
        self.create_subscription(Image, topic, self._on_image, 10)
        # TF listener for `--pose-source tf`. Buffer is fed by the executor
        # in main(), same as the image subscription, so it stays current
        # without any extra spin.
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

    def _on_image(self, msg: Image) -> None:
        self.frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def lookup_posx_from_tf(self, base_frame: str,
                            ee_frame: str) -> Optional[tuple]:
        """Read T_base→ee from /tf (published by robot_state_publisher) and
        convert to Doosan's `posx` convention: translation in mm,
        ZYZ Euler angles in degrees.

        Works regardless of control authority — the broadcast publishes
        joint_states whenever the controller is alive, so RViz, downstream
        FK, and this function all see the same up-to-date pose.
        Returns None if the transform isn't available (yet)."""
        from scipy.spatial.transform import Rotation
        try:
            tfm = self.tf_buffer.lookup_transform(
                base_frame, ee_frame, rclpy.time.Time())
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(
                f'TF lookup {base_frame}<-{ee_frame} failed: {e}',
                throttle_duration_sec=2.0)
            return None
        t = tfm.transform.translation
        q = tfm.transform.rotation
        rot = Rotation.from_quat([q.x, q.y, q.z, q.w])  # scipy uses xyzw
        zyz = rot.as_euler('ZYZ', degrees=True)
        return (
            float(t.x) * 1000.0, float(t.y) * 1000.0, float(t.z) * 1000.0,
            float(zyz[0]), float(zyz[1]), float(zyz[2]),
        )


# ----------------------------------------------------------------------
# Calibration solvers
# ----------------------------------------------------------------------
def _calibrate_exo(samples: list[dict]) -> np.ndarray:
    """Eye-to-hand: ArUco marker is on the EE; camera is fixed in space.

    cv2.calibrateHandEye normally solves AX = XB for eye-in-hand. To repurpose
    it for eye-to-hand we feed the inverse of the gripper poses (so X becomes
    T_base2cam instead of T_gripper2cam). Convention follows
    sample/Calibration_Tutorial/handeye_calibration.py (which already does
    this swap for its own eye-to-hand setup).

    Returns T_exo2base (4x4, translation in mm).
    """
    R_g2b_inv_list, t_g2b_inv_list = [], []
    R_t2c_list, t_t2c_list = [], []
    for s in samples:
        T_b2g = _posx_to_T_base2ee(s['posx'])
        T_g2b = np.linalg.inv(T_b2g)
        R_g2b_inv_list.append(T_g2b[:3, :3])
        t_g2b_inv_list.append(T_g2b[:3, 3].reshape(3, 1))
        # marker→cam (input wants R_target2cam, t_target2cam) — solvePnP gave
        # us R_cam_marker so we already have what's needed (cam-frame pose
        # of the marker).
        R_t2c_list.append(s['R_cam_marker'])
        t_t2c_list.append((s['t_cam_marker_m'] * 1000.0).reshape(3))  # m→mm

    R_x, t_x = cv2.calibrateHandEye(
        R_g2b_inv_list, t_g2b_inv_list,
        R_t2c_list, t_t2c_list,
        method=cv2.CALIB_HAND_EYE_PARK)
    T = np.eye(4)
    T[:3, :3] = R_x
    T[:3, 3] = np.asarray(t_x).reshape(3)
    return T


def _calibrate_hand(samples: list[dict]) -> np.ndarray:
    """Eye-in-hand: camera mounted on EE, ArUco marker is fixed in the world.

    Standard cv2.calibrateHandEye: A = base→gripper, B = cam→target.
    Returns T_hand2base = T_gripper2cam (translation in mm) — actually the
    matrix relating EE to camera; with our naming `T_hand2base` denotes the
    EE-relative camera pose used at runtime as `cam_in_base = base2ee @
    T_hand2base`. NOTE: cv2 returns gripper2cam, so consumers compose:
        cam_in_base = T_base2ee @ T_hand2base
    """
    R_g2b_list, t_g2b_list = [], []
    R_t2c_list, t_t2c_list = [], []
    for s in samples:
        T_b2g = _posx_to_T_base2ee(s['posx'])
        R_g2b_list.append(T_b2g[:3, :3])
        t_g2b_list.append(T_b2g[:3, 3].reshape(3, 1))
        R_t2c_list.append(s['R_cam_marker'])
        t_t2c_list.append((s['t_cam_marker_m'] * 1000.0).reshape(3))  # m→mm

    R_x, t_x = cv2.calibrateHandEye(
        R_g2b_list, t_g2b_list,
        R_t2c_list, t_t2c_list,
        method=cv2.CALIB_HAND_EYE_PARK)
    T = np.eye(4)
    T[:3, :3] = R_x
    T[:3, 3] = np.asarray(t_x).reshape(3)
    return T


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--intrinsics', type=Path, required=True,
                        help='YAML produced by `calibrate` CLI (used for the '
                             'live ArUco axes overlay only)')
    parser.add_argument('--output-dir', type=Path,
                        default=Path('/home/eunwoosong/Projects/'
                                     'ros2-depth-point-cloude/record/exo'),
                        help='Directory to save aruco_NNN.png + calibrate_data.json '
                             '(default: record/exo under the project root)')
    parser.add_argument('--marker-length', type=float, required=True,
                        help='ArUco marker side length in METRES (e.g. 0.05) '
                             '— for the live axes overlay; the solver re-derives '
                             'pose from each saved image at compute time.')
    parser.add_argument('--dict', dest='aruco_dict', default='4X4_50',
                        help='ArUco dictionary (default 4X4_50)')
    parser.add_argument('--marker-id', type=int, default=0,
                        help='Specific marker id to track (default 0). '
                             '-1 = whichever marker is detected first.')
    parser.add_argument('--topic', default='/camera/camera/color/image_raw')
    parser.add_argument('--pose-source', choices=['service', 'tf', 'manual'],
                        default='service',
                        help='Where to read the robot posx at save time. '
                             '`service` (default): DSR GetCurrentPosx — same '
                             'as sample/Calibration_Tutorial/data_recording.py; '
                             'returns the configured TCP pose. '
                             '`tf`: read /tf base_frame→ee_frame and convert '
                             'to mm + ZYZ deg — control-authority-independent '
                             'fallback. '
                             '`manual`: prompt for posx via stdin each save.')
    parser.add_argument('--base-frame', default='base_0',
                        help='Base TF frame for --pose-source=tf')
    parser.add_argument('--ee-frame', default='link6',
                        help='EE TF frame for --pose-source=tf '
                             '(rigidly attached to the marker)')
    parser.add_argument('--no-robot', action='store_true',
                        help='[deprecated] alias for --pose-source manual')
    args, ros_args = parser.parse_known_args(argv)

    intr = load_intrinsics(args.intrinsics)
    K = intr.K
    dist = intr.dist.reshape(-1, 1)

    detector, dictionary = _make_aruco_detector(args.aruco_dict)

    rclpy.init(args=ros_args)
    node = _CaptureNode(args.topic)
    # MultiThreadedExecutor with 2 worker threads: one services the image
    # subscription callback at ~30 Hz, the other handles GetCurrentPosx
    # service responses without head-of-line blocking on the image queue.
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    # All status output goes through the ROS logger so `output='screen'` in
    # the launch reliably shows it (avoids the Python stdout-buffering pitfall
    # when the entry point is spawned by ros2 launch as a non-TTY process).
    log = node.get_logger()

    # Output directory + JSON registry. Resume is automatic — if the JSON
    # already exists we append, so the user can stop / restart capture
    # without losing prior samples.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.output_dir / 'calibrate_data.json'
    if data_path.is_file():
        try:
            with data_path.open('r') as f:
                write_data = json.load(f)
        except json.JSONDecodeError:
            write_data = {}
    else:
        write_data = {}
    write_data.setdefault('poses', [])
    write_data.setdefault('file_name', [])

    # Resolve pose source. `--no-robot` is a deprecated alias for `manual`.
    pose_source = 'manual' if args.no_robot else args.pose_source

    log.info(f'aruco_calibrate (capture) ready: dict={args.aruco_dict} '
             f'marker_id={args.marker_id} marker_length={args.marker_length} m')
    log.info(f'output dir: {args.output_dir}  (existing samples: '
             f'{len(write_data["file_name"])})')
    log.info(f'JSON: {data_path}')
    if pose_source == 'tf':
        log.info(f'pose source: TF lookup '
                 f'{args.base_frame} ← {args.ee_frame} (mm + ZYZ deg). '
                 f'Independent of robot control authority.')
    elif pose_source == 'service':
        log.info('pose source: DSR GetCurrentPosx service '
                 '(requires control authority).')
    else:
        log.info('pose source: manual (stdin prompt on every save).')

    # DSR service client only needed for `service` mode.
    read_posx_service = (
        _try_import_get_current_posx_service(node)
        if pose_source == 'service' else None)

    win = 'aruco_calibrate (capture) — s=save, u=undo, q=quit'
    cv2.namedWindow(win)

    # Freeze-on-last-detect cache. Display + save both work off these:
    #
    #   cached_raw      — RGB frame at the last successful detection
    #                     (saved verbatim to PNG; the solver re-detects from it)
    #   cached_overlay  — same frame with ArUco overlay drawn (what we show)
    #   cached_posx     — robot pose READ AT THE SAME MOMENT as cached_raw
    #                     (critical: pairing the two avoids the time skew
    #                     between detection and 's' that produced 90°+ rotation
    #                     residuals on prior captures)
    #   cached_R / cached_t — for the status line / save log
    #   frames_since_detect — visualised so the user can see staleness
    cached_raw: Optional[np.ndarray] = None
    cached_overlay: Optional[np.ndarray] = None
    cached_posx: Optional[tuple] = None
    cached_R: Optional[np.ndarray] = None
    cached_t: Optional[np.ndarray] = None
    frames_since_detect = 0

    def _read_posx_now() -> Optional[tuple]:
        """Helper: read the current posx according to the configured source.
        `manual` mode never auto-reads (we'd need stdin); returns None so
        the cache is left untouched and the save path falls through to a
        prompt."""
        if pose_source == 'tf':
            return node.lookup_posx_from_tf(args.base_frame, args.ee_frame)
        if pose_source == 'service' and read_posx_service is not None:
            return read_posx_service()
        return None

    try:
        while rclpy.ok():
            time.sleep(0.01)
            frame = node.frame
            if frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids = _detect_marker(detector, dictionary, gray)
            picked_corners = None
            if ids is not None and len(ids) > 0:
                if args.marker_id < 0:
                    picked_corners = corners[0]
                else:
                    matches = [i for i, mid in enumerate(ids.flatten().tolist())
                               if int(mid) == args.marker_id]
                    if matches:
                        picked_corners = corners[matches[0]]

            fresh = False
            if picked_corners is not None:
                R_cm, t_cm = _solve_marker_pose(
                    picked_corners, args.marker_length, K, dist)
                if R_cm is not None:
                    # Read robot posx NOW so it's paired with this exact
                    # frame. In manual mode posx_now stays None — the save
                    # path will prompt instead. In tf/service mode, if the
                    # read fails (TF not yet available, service rejected),
                    # we *don't* update the cache: better to keep the last
                    # synced pair than to mix this image with a stale posx.
                    posx_now = _read_posx_now() if pose_source != 'manual' else 'manual_pending'

                    if pose_source == 'manual' or posx_now is not None:
                        # Build an overlay snapshot to cache for display + freeze.
                        overlay = frame.copy()
                        if ids is not None and len(ids) > 0:
                            cv2.aruco.drawDetectedMarkers(overlay, corners, ids)
                        pick_col = (0, 255, 255)
                        pts4 = picked_corners.reshape(-1, 2).astype(int)
                        cv2.polylines(overlay, [pts4], True, pick_col, 2,
                                      lineType=cv2.LINE_AA)
                        for ci, (px, py) in enumerate(pts4):
                            cv2.circle(overlay, (int(px), int(py)), 6, pick_col, -1)
                            cv2.circle(overlay, (int(px), int(py)), 7, (0, 0, 0), 1)
                            cv2.putText(overlay, str(ci),
                                        (int(px) + 8, int(py) - 8),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                        pick_col, 1, cv2.LINE_AA)
                        if hasattr(cv2, 'drawFrameAxes'):
                            rvec, _ = cv2.Rodrigues(R_cm)
                            cv2.drawFrameAxes(
                                overlay, K, dist, rvec, t_cm.reshape(3, 1),
                                args.marker_length * 0.5)

                        cached_raw = frame.copy()
                        cached_overlay = overlay
                        cached_posx = (None if posx_now == 'manual_pending'
                                       else posx_now)
                        cached_R = R_cm
                        cached_t = t_cm
                        frames_since_detect = 0
                        fresh = True

            if not fresh:
                frames_since_detect += 1

            # Render: prefer the cached (frozen) overlay; fall back to live
            # frame only if we have never detected anything yet.
            if cached_overlay is not None:
                display = cached_overlay.copy()
            else:
                display = frame.copy()
                if ids is not None and len(ids) > 0:
                    seen = ','.join(str(int(v)) for v in ids.flatten().tolist())
                    cv2.putText(display, f'wanted id={args.marker_id}; seen=[{seen}]',
                                (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 255), 1, cv2.LINE_AA)

            # Posx-pairing badge: green only when image+posx are paired in
            # cache (the only configuration that yields valid hand-eye data).
            if pose_source == 'manual':
                pair_tag = 'posx=manual@save'
            elif cached_posx is not None:
                pair_tag = 'posx PAIRED'
            else:
                pair_tag = 'posx MISSING (read failing)'

            if fresh:
                colour, tag = (0, 255, 0), 'detected (live)'
            elif cached_overlay is not None:
                colour, tag = (180, 180, 180), f'frozen (last detect {frames_since_detect}f ago)'
            else:
                colour, tag = (0, 0, 255), 'NOT detected — move marker into view'
            t_disp = ('—' if cached_t is None
                      else f'({cached_t[0]:.3f},{cached_t[1]:.3f},{cached_t[2]:.3f}) m')
            tracked = ('any' if args.marker_id < 0 else f'id={args.marker_id}')
            n_saved = len(write_data['file_name'])
            status = (f'capture dict={args.aruco_dict} track={tracked} | '
                      f'{tag} | {pair_tag} | saved={n_saved} | '
                      f't_cam_marker={t_disp}')
            cv2.putText(display, status, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
            cv2.putText(display,
                        's=save cached image+posx  u=undo  q=quit',
                        (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (200, 200, 200), 1)
            cv2.imshow(win, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('u'):
                if not write_data['file_name']:
                    log.warn('[u] no saved samples to undo')
                    continue
                last_fname = write_data['file_name'].pop()
                # Pop the matching pose so the two lists stay aligned — value
                # itself isn't needed because we re-write the JSON below.
                write_data['poses'].pop()
                last_path = args.output_dir / last_fname
                try:
                    last_path.unlink()
                except FileNotFoundError:
                    pass
                with data_path.open('w') as f:
                    json.dump(write_data, f, indent=2)
                log.info(
                    f'[u] removed {last_fname} '
                    f'(remaining: {len(write_data["file_name"])})')
            elif key == ord('s'):
                if cached_raw is None:
                    log.warn('[s] save skipped: no detection cached yet — '
                             'move the marker into view first.')
                    continue
                # Use the posx cached AT THE SAME MOMENT as cached_raw — not
                # a fresh read at 's' press time. Earlier captures had ~100
                # mm / 60° per-sample residuals because of the time skew
                # between "marker last detected" and "user pressed s"; even
                # small robot motion in that gap broke the AX=XB pairing.
                if pose_source == 'manual':
                    raw = input(
                        'Enter robot posx for the cached frame '
                        '(x y z rx ry rz, mm/deg ZYZ): ')
                    try:
                        posx = tuple(float(v) for v in raw.replace(',', ' ').split())
                        if len(posx) != 6:
                            raise ValueError
                    except ValueError:
                        log.warn('[s] save skipped: invalid posx input')
                        continue
                else:
                    if cached_posx is None:
                        log.error(
                            f'[s] save skipped: no posx was paired with the '
                            f'cached frame (pose_source={pose_source}). '
                            f'GetCurrentPosx / TF lookup likely failing — '
                            f'check the logs above.')
                        continue
                    posx = cached_posx

                idx = len(write_data['file_name'])
                fname = f'aruco_{idx:03d}.png'
                img_path = args.output_dir / fname
                # Save the *cached* raw frame (the moment ArUco was last
                # detected). The solver re-runs detection on this PNG.
                if not cv2.imwrite(str(img_path), cached_raw):
                    log.error(f'[s] save skipped: failed to write {img_path}')
                    continue
                write_data['file_name'].append(fname)
                write_data['poses'].append(list(posx))
                with data_path.open('w') as f:
                    json.dump(write_data, f, indent=2)
                log.info(
                    f'[s] saved {fname} '
                    f'(total: {len(write_data["file_name"])}, '
                    f'cached {frames_since_detect}f ago) | '
                    f'posx=({posx[0]:.1f},{posx[1]:.1f},{posx[2]:.1f},'
                    f'{posx[3]:.1f},{posx[4]:.1f},{posx[5]:.1f})')
    finally:
        # Log before shutdown — node's logger needs rclpy alive.
        try:
            log.info(
                f'capture done. {len(write_data["file_name"])} samples in '
                f'{args.output_dir}. Run aruco_handeye to compute the '
                f'calibration matrix.')
        except Exception:
            pass
        cv2.destroyAllWindows()
        try:
            executor.shutdown(timeout_sec=1.0)
        except Exception:
            pass
        rclpy.shutdown()


if __name__ == '__main__':
    main(sys.argv[1:])
