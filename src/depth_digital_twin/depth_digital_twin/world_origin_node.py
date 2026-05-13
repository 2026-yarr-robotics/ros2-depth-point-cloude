"""world_origin_node — define `world` frame relative to the camera.

Two modes, selected by `world_origin_mode`:

* **aruco** (default): detect a known ArUco marker placed in the workspace at a
  measured position relative to the robot base. On startup the node collects
  `world_marker_samples_required` good detections, averages them, and publishes
  a single static TF camera→world (world := robot base). The color subscription
  is then released; the TF persists indefinitely.

  Marker frame convention (OpenCV / solvePnP):
    - Z  : out of marker plane, toward camera (= world +Z for flat marker on table)
    - X  : right on marker pattern
    - Y  : top  of marker pattern

  The rigid transform from marker frame to base frame is specified via six
  params (world_marker_offset_*_m and world_marker_rot_*_deg). The offsets are
  the marker's position in the BASE frame (easy to measure physically). The
  rotations describe the marker frame's orientation relative to base frame
  (Euler intrinsic XYZ, degrees). Both default to zero (marker axes = base axes,
  suitable for a flat marker whose normal is world +Z and whose pattern top
  points in the base +Y direction).

* **floor** (fallback): sample a depth patch around (window_center_*_px),
  fit a plane via SVD, build a Z-up basis, publish a static TF. Activated
  when `world_origin_mode: floor` or when ArUco mode times out with
  `aruco_timeout_then_floor: true`.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path

import cv2
import cv2.aruco as aruco
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger
from tf2_ros import StaticTransformBroadcaster

from depth_digital_twin.intrinsics import load_intrinsics


class WorldOriginNode(Node):
    def __init__(self) -> None:
        super().__init__('world_origin_node')

        # ── common params ──────────────────────────────────────────────────
        self.declare_parameter('intrinsics_path', '')
        self.declare_parameter('camera_frame', 'camera_color_optical_frame')
        self.declare_parameter('world_frame', 'world')
        self.declare_parameter('depth_unit', 0.001)
        self.declare_parameter('window_radius', 20)
        self.declare_parameter('window_center_x_px', -1)
        self.declare_parameter('window_center_y_px', -1)

        # ── mode ───────────────────────────────────────────────────────────
        self.declare_parameter('world_origin_mode', 'aruco')  # aruco | floor

        # ── aruco mode params ──────────────────────────────────────────────
        self.declare_parameter('color_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('world_marker_id', 0)
        self.declare_parameter('world_marker_dict', 'DICT_4X4_50')
        self.declare_parameter('world_marker_length_m', 0.05)
        self.declare_parameter('world_marker_samples_required', 30)
        self.declare_parameter('world_marker_reproj_err_max_px', 2.0)
        self.declare_parameter('world_marker_timeout_s', 15.0)
        self.declare_parameter('aruco_timeout_then_floor', True)
        # Marker position in BASE frame (measured physically, metres)
        self.declare_parameter('world_marker_offset_x_m', 0.367)
        self.declare_parameter('world_marker_offset_y_m', 0.003)
        self.declare_parameter('world_marker_offset_z_m', 0.0)
        # Marker frame orientation relative to BASE frame (Euler intrinsic XYZ, degrees).
        # Zero = marker axes aligned with base axes.
        self.declare_parameter('world_marker_rot_x_deg', 0.0)
        self.declare_parameter('world_marker_rot_y_deg', 0.0)
        self.declare_parameter('world_marker_rot_z_deg', 0.0)

        # ── floor mode params ──────────────────────────────────────────────
        self.declare_parameter('depth_topic',
                               '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('min_patch_points', 100)
        self.declare_parameter('max_plane_residual', 0.01)
        self.declare_parameter('samples_required', 10)

        # ── resolve common ─────────────────────────────────────────────────
        path = Path(self.get_parameter('intrinsics_path').value)
        if not path.is_file():
            raise FileNotFoundError(f'intrinsics_path not found: {path}')
        self.intr = load_intrinsics(path)
        self.get_logger().info(
            f'Intrinsics: {self.intr.width}x{self.intr.height} '
            f'fx={self.intr.fx:.2f} cx={self.intr.cx:.2f}')

        self.camera_frame: str = self.get_parameter('camera_frame').value
        self.world_frame: str = self.get_parameter('world_frame').value
        self.depth_unit: float = float(self.get_parameter('depth_unit').value)
        self.window_radius: int = int(self.get_parameter('window_radius').value)
        self.window_cx_px: int = int(self.get_parameter('window_center_x_px').value)
        self.window_cy_px: int = int(self.get_parameter('window_center_y_px').value)

        self.bridge = CvBridge()
        self.broadcaster = StaticTransformBroadcaster(self)
        self.published = False
        self._mode = str(self.get_parameter('world_origin_mode').value).strip().lower()

        # Service: ~/redetect — restart detection without restarting the node.
        self.create_service(Trigger, '~/redetect', self._handle_redetect)

        if self._mode == 'aruco':
            self._setup_aruco_mode()
        else:
            self._setup_floor_mode()

    # ── Redetect service ───────────────────────────────────────────────────

    def _handle_redetect(self, _req: Trigger.Request,
                         resp: Trigger.Response) -> Trigger.Response:
        """Restart detection from scratch without restarting the node.

        ArUco mode: clear samples, re-subscribe to color topic if needed.
        Floor mode: clear plane-fit samples.
        In both cases the previously published TF stays until a new one is sent.
        """
        self.published = False

        if self._mode == 'aruco':
            self.aruco_samples = []
            self.aruco_start = self.get_clock().now()
            if not hasattr(self, '_aruco_sub') or self._aruco_sub is None:
                self._aruco_sub = self.create_subscription(
                    Image,
                    str(self.get_parameter('color_topic').value),
                    self._on_color_aruco, 10)
            self.get_logger().info('Redetect: ArUco detection restarted.')
            resp.success = True
            resp.message = 'ArUco detection restarted'
        else:
            if hasattr(self, 'origin_samples'):
                self.origin_samples.clear()
                self.R_samples.clear()
            self.get_logger().info('Redetect: floor plane-fit reset.')
            resp.success = True
            resp.message = 'Floor fit reset'

        return resp

    # ── ArUco mode ─────────────────────────────────────────────────────────

    def _setup_aruco_mode(self) -> None:
        self.marker_id = int(self.get_parameter('world_marker_id').value)
        dict_name = str(self.get_parameter('world_marker_dict').value)
        self.marker_length = float(self.get_parameter('world_marker_length_m').value)
        self.aruco_samples_req = int(
            self.get_parameter('world_marker_samples_required').value)
        self.aruco_reproj_max = float(
            self.get_parameter('world_marker_reproj_err_max_px').value)
        self.aruco_timeout_s = float(
            self.get_parameter('world_marker_timeout_s').value)
        self.aruco_fallback = bool(
            self.get_parameter('aruco_timeout_then_floor').value)

        dx = float(self.get_parameter('world_marker_offset_x_m').value)
        dy = float(self.get_parameter('world_marker_offset_y_m').value)
        dz = float(self.get_parameter('world_marker_offset_z_m').value)
        rx = float(self.get_parameter('world_marker_rot_x_deg').value)
        ry = float(self.get_parameter('world_marker_rot_y_deg').value)
        rz = float(self.get_parameter('world_marker_rot_z_deg').value)

        # offset = marker origin position in BASE frame  (physical measurement).
        # R_m2b  = "marker axes expressed in BASE frame" — rotation that
        #          transforms a vector from MARKER frame into BASE frame.
        #
        # The pose-of-marker-in-base-frame transform is [R_m2b | offset]
        # (this transforms MARKER points into BASE coordinates).
        # We need its INVERSE — the transform that turns BASE points into MARKER
        # coordinates — because the chain is:
        #     p_cam = R_cm · p_marker + tvec                  (solvePnP)
        # so to map p_base → p_cam we first need p_base → p_marker.
        R_m2b = _euler_xyz_to_R(rx, ry, rz)
        self.T_marker_base = _invert_se3(R_m2b, np.array([dx, dy, dz]))
        # Net effect: base_origin_in_cam = tvec − R_cm · R_m2b^T · offset
        # i.e. start at marker, move OPPOSITE to the offset (rotated into the
        # camera frame).  Since offset = "marker position from base", the base
        # is at −offset from the marker.
        # Stored for aruco TF broadcast (world → aruco)
        self._aruco_R_m2b = R_m2b
        self._aruco_offset = np.array([dx, dy, dz])

        # ArUco detector — tuned for small markers
        if not hasattr(aruco, dict_name):
            raise ValueError(
                f'Unknown ArUco dict {dict_name!r}. '
                'Use e.g. DICT_4X4_50 or DICT_5X5_50.')
        aruco_dict = aruco.getPredefinedDictionary(getattr(aruco, dict_name))
        det_params = aruco.DetectorParameters()
        det_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        # Allow small markers (default minMarkerPerimeterRate=0.03 misses tiny markers)
        det_params.minMarkerPerimeterRate = 0.01
        det_params.adaptiveThreshWinSizeMin = 3
        det_params.adaptiveThreshWinSizeMax = 23
        det_params.adaptiveThreshWinSizeStep = 10
        self.aruco_detector = aruco.ArucoDetector(aruco_dict, det_params)

        # Object points: marker corners in marker frame (Z=0 plane, Z toward camera)
        L = self.marker_length / 2.0
        self.obj_pts = np.array([
            [-L,  L, 0.0],
            [ L,  L, 0.0],
            [ L, -L, 0.0],
            [-L, -L, 0.0],
        ], dtype=np.float32)

        self.aruco_samples: list[np.ndarray] = []
        self.aruco_start = self.get_clock().now()
        self._aruco_sub = self.create_subscription(
            Image,
            str(self.get_parameter('color_topic').value),
            self._on_color_aruco, 10)

        self.get_logger().info(
            f'ArUco mode: ID={self.marker_id} dict={dict_name} '
            f'length={self.marker_length*100:.1f}cm '
            f'target={self.aruco_samples_req} samples. '
            f'Marker in base frame: ({dx:.3f}, {dy:.3f}, {dz:.3f}) m  '
            f'rot_xyz=({rx:.1f}, {ry:.1f}, {rz:.1f}) deg')

    def _on_color_aruco(self, msg: Image) -> None:
        if self.published:
            return

        elapsed = (self.get_clock().now() - self.aruco_start).nanoseconds * 1e-9

        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.aruco_detector.detectMarkers(gray)

        if ids is None or self.marker_id not in ids.flatten():
            if elapsed > self.aruco_timeout_s:
                if self.aruco_fallback:
                    self.get_logger().warn(
                        f'ArUco ID={self.marker_id} not detected after '
                        f'{elapsed:.0f}s — falling back to floor-plane fit.')
                    self.destroy_subscription(self._aruco_sub)
                    self._setup_floor_mode()
                else:
                    self.get_logger().error(
                        f'ArUco ID={self.marker_id} not detected after {elapsed:.0f}s.',
                        throttle_duration_sec=5.0)
            return

        idx = int(np.where(ids.flatten() == self.marker_id)[0][0])
        corner = corners[idx].reshape(4, 2).astype(np.float32)

        # IPPE_SQUARE always returns two solutions.  We must pick the one where:
        #   (A) tvec[2] > 0  — marker is in FRONT of the camera
        #   (B) marker normal (R[:,2]) faces the camera, i.e.
        #       R[:,2] · (−tvec) > 0
        #
        # Checking only (A) is insufficient when the camera views the marker
        # from the side: both solutions can satisfy (A) while one has the
        # marker's Z axis pointing INTO the table instead of toward the camera.
        # That wrong solution makes the camera appear below the world XY plane.
        n_sols, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            self.obj_pts, corner,
            self.intr.K, self.intr.dist,
            flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if n_sols == 0:
            return

        best_rvec, best_tvec, best_facing = rvecs[0], tvecs[0], -float('inf')
        for k in range(n_sols):
            t_k = tvecs[k].flatten()
            if float(t_k[2]) <= 0:          # (A) must be in front
                continue
            R_k, _ = cv2.Rodrigues(rvecs[k])
            # (B) marker normal dot (camera→marker direction) — higher = more facing
            facing = float(np.dot(R_k[:, 2], -t_k))
            if facing > best_facing:
                best_facing = facing
                best_rvec, best_tvec = rvecs[k], tvecs[k]

        rvec, tvec = best_rvec, best_tvec

        if best_facing == -float('inf'):
            self.get_logger().warn(
                'All IPPE solutions have marker behind camera — skipping.',
                throttle_duration_sec=2.0)
            return
        if best_facing <= 0:
            self.get_logger().warn(
                f'Best IPPE solution has marker facing away (facing={best_facing:.3f}) '
                '— marker may be viewed too obliquely.',
                throttle_duration_sec=2.0)

        proj, _ = cv2.projectPoints(
            self.obj_pts, rvec, tvec, self.intr.K, self.intr.dist)
        reproj_err = float(np.mean(
            np.linalg.norm(proj.reshape(4, 2) - corner, axis=1)))
        if reproj_err > self.aruco_reproj_max:
            self.get_logger().warn(
                f'Marker reproj err {reproj_err:.1f}px '
                f'> {self.aruco_reproj_max:.1f}px — rejected.',
                throttle_duration_sec=1.0)
            return

        R_cm, _ = cv2.Rodrigues(rvec)
        T_cam_marker = np.eye(4)
        T_cam_marker[:3, :3] = R_cm
        T_cam_marker[:3, 3] = tvec.flatten()
        self.aruco_samples.append(T_cam_marker)

        n = len(self.aruco_samples)
        dist_cm = float(np.linalg.norm(tvec)) * 100.0
        self.get_logger().info(
            f'Marker [{n}/{self.aruco_samples_req}] '
            f'reproj={reproj_err:.2f}px dist={dist_cm:.1f}cm',
            throttle_duration_sec=0.5)

        if n < self.aruco_samples_req:
            return

        # Average N samples → T_cam→base
        T_avg = _se3_average(self.aruco_samples)
        T_cam_base = T_avg @ self.T_marker_base

        # Log for sanity check
        euler_deg = _R_to_euler_xyz(T_avg[:3, :3])
        t_avg = T_avg[:3, 3]
        stds_mm = np.std([T[:3, 3] for T in self.aruco_samples], axis=0) * 1000.0

        # Computed world (base) origin in the CAMERA frame
        t_base_in_cam = T_cam_base[:3, 3]
        # Camera position in the WORLD frame (what RViz displays as camera pose)
        t_cam_in_world = -T_cam_base[:3, :3].T @ T_cam_base[:3, 3]
        # World axes in camera frame  (columns of R_cam_base)
        Rcw = T_cam_base[:3, :3]
        # Active R_m2b — if rot_z changes, this matrix changes.  If you see
        # the SAME world_x/y/z lines across two runs with different rot_z values,
        # the parameter is NOT being loaded.
        R_m2b_euler = _R_to_euler_xyz(self._aruco_R_m2b)

        self.get_logger().info(
            f'Calibration complete:\n'
            f'  ── params actually in use ──\n'
            f'    offset (m)       = ({self._aruco_offset[0]:.3f}, {self._aruco_offset[1]:.3f}, {self._aruco_offset[2]:.3f})\n'
            f'    R_m2b euler_xyz  = ({R_m2b_euler[0]:.1f}, {R_m2b_euler[1]:.1f}, {R_m2b_euler[2]:.1f}) deg\n'
            f'  ── marker pose (cam frame) ──\n'
            f'    pos (cm)         = ({t_avg[0]*100:.1f}, {t_avg[1]*100:.1f}, {t_avg[2]*100:.1f})\n'
            f'    euler_xyz (deg)  = ({euler_deg[0]:.1f}, {euler_deg[1]:.1f}, {euler_deg[2]:.1f})\n'
            f'    pos std (mm)     = ({stds_mm[0]:.1f}, {stds_mm[1]:.1f}, {stds_mm[2]:.1f})\n'
            f'  ── world axes in CAM frame (should rotate when rot_z changes) ──\n'
            f'    world_x = ({Rcw[0,0]:+.3f}, {Rcw[1,0]:+.3f}, {Rcw[2,0]:+.3f})\n'
            f'    world_y = ({Rcw[0,1]:+.3f}, {Rcw[1,1]:+.3f}, {Rcw[2,1]:+.3f})\n'
            f'    world_z = ({Rcw[0,2]:+.3f}, {Rcw[1,2]:+.3f}, {Rcw[2,2]:+.3f})\n'
            f'  ── world origin & camera pos ──\n'
            f'    base in cam (cm) = ({t_base_in_cam[0]*100:.1f}, {t_base_in_cam[1]*100:.1f}, {t_base_in_cam[2]*100:.1f})\n'
            f'    cam in world(cm) = ({t_cam_in_world[0]*100:.1f}, {t_cam_in_world[1]*100:.1f}, {t_cam_in_world[2]*100:.1f})\n'
            f'                       Z sign={"+" if t_cam_in_world[2] >= 0 else "−"} (must be + for camera above table)\n'
            f'  → Compare "R_m2b euler_xyz" against params.yaml rot_xyz_deg.')

        self._publish_aruco_origin(T_cam_base)
        self.destroy_subscription(self._aruco_sub)
        self._aruco_sub = None

    # ── Floor mode ─────────────────────────────────────────────────────────

    def _setup_floor_mode(self) -> None:
        self.min_patch_points = int(self.get_parameter('min_patch_points').value)
        self.max_plane_residual = float(self.get_parameter('max_plane_residual').value)
        self.samples_required = int(self.get_parameter('samples_required').value)
        self.origin_samples: deque[np.ndarray] = deque(maxlen=self.samples_required)
        self.R_samples: deque[np.ndarray] = deque(maxlen=self.samples_required)
        self.create_subscription(
            Image,
            str(self.get_parameter('depth_topic').value),
            self._on_depth, 10)
        self.get_logger().info('Floor mode: waiting for depth samples…')

    def _on_depth(self, msg: Image) -> None:
        if self.published:
            return
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        h, w = depth.shape[:2]
        cx_px = w // 2 if self.window_cx_px < 0 else self.window_cx_px
        cy_px = h // 2 if self.window_cy_px < 0 else self.window_cy_px
        cx_px = int(np.clip(cx_px, 0, w - 1))
        cy_px = int(np.clip(cy_px, 0, h - 1))
        r = self.window_radius
        y0, y1 = max(0, cy_px - r), min(h, cy_px + r + 1)
        x0, x1 = max(0, cx_px - r), min(w, cx_px + r + 1)
        patch = depth[y0:y1, x0:x1].astype(np.float32)
        if patch.size == 0:
            return
        ys, xs = np.mgrid[y0:y1, x0:x1]
        valid = patch > 0
        if int(valid.sum()) < self.min_patch_points:
            return

        zs = patch[valid] * self.depth_unit
        us = xs[valid].astype(np.float32)
        vs = ys[valid].astype(np.float32)
        X = (us - self.intr.cx) * zs / self.intr.fx
        Y = (vs - self.intr.cy) * zs / self.intr.fy
        pts = np.stack([X, Y, zs], axis=1)

        p0 = pts.mean(axis=0)
        centred = pts - p0
        _, S, Vt = np.linalg.svd(centred, full_matrices=False)
        normal = Vt[-1]
        residual = float(S[-1] / np.sqrt(max(1, pts.shape[0])))
        if residual > self.max_plane_residual:
            self.get_logger().warn(
                f'Plane residual {residual:.4f} m > {self.max_plane_residual:.4f}; skip',
                throttle_duration_sec=2.0)
            return

        if float(np.dot(normal, -p0)) < 0.0:
            normal = -normal
        normal = normal / (np.linalg.norm(normal) + 1e-12)
        cam_x = np.array([1.0, 0.0, 0.0])
        x_world = cam_x - float(np.dot(cam_x, normal)) * normal
        if np.linalg.norm(x_world) < 1e-3:
            cam_y = np.array([0.0, 1.0, 0.0])
            x_world = cam_y - float(np.dot(cam_y, normal)) * normal
        x_world = x_world / (np.linalg.norm(x_world) + 1e-12)
        y_world = np.cross(normal, x_world)
        R = np.column_stack([x_world, y_world, normal])

        self.origin_samples.append(p0)
        self.R_samples.append(R)
        if len(self.origin_samples) < self.samples_required:
            return

        origin = np.median(np.stack(self.origin_samples, axis=0), axis=0)
        R_mean = np.mean(np.stack(self.R_samples, axis=0), axis=0)
        U, _, Vt2 = np.linalg.svd(R_mean)
        D = np.eye(3)
        D[2, 2] = float(np.sign(np.linalg.det(U @ Vt2)))
        R_final = U @ D @ Vt2

        T = np.eye(4)
        T[:3, :3] = R_final
        T[:3, 3] = origin
        self._publish_static_tf(T, mode='floor-plane-fit')

    # ── TF publish ─────────────────────────────────────────────────────────

    def _publish_aruco_origin(self, T_cam_base: np.ndarray) -> None:
        """Send camera→world and world→aruco static TFs in one call.

        StaticTransformBroadcaster replaces ALL previously published transforms
        on each sendTransform() call, so both TFs must be sent together.
        """
        stamp = self.get_clock().now().to_msg()

        R_cw = T_cam_base[:3, :3]
        t_cw = T_cam_base[:3, 3]
        qx, qy, qz, qw = _rot_to_quat(R_cw)
        tf_world = TransformStamped()
        tf_world.header.stamp = stamp
        tf_world.header.frame_id = self.camera_frame
        tf_world.child_frame_id = self.world_frame
        tf_world.transform.translation.x = float(t_cw[0])
        tf_world.transform.translation.y = float(t_cw[1])
        tf_world.transform.translation.z = float(t_cw[2])
        tf_world.transform.rotation.x = qx
        tf_world.transform.rotation.y = qy
        tf_world.transform.rotation.z = qz
        tf_world.transform.rotation.w = qw

        qx2, qy2, qz2, qw2 = _rot_to_quat(self._aruco_R_m2b)
        tf_aruco = TransformStamped()
        tf_aruco.header.stamp = stamp
        tf_aruco.header.frame_id = self.world_frame
        tf_aruco.child_frame_id = 'aruco'
        tf_aruco.transform.translation.x = float(self._aruco_offset[0])
        tf_aruco.transform.translation.y = float(self._aruco_offset[1])
        tf_aruco.transform.translation.z = float(self._aruco_offset[2])
        tf_aruco.transform.rotation.x = qx2
        tf_aruco.transform.rotation.y = qy2
        tf_aruco.transform.rotation.z = qz2
        tf_aruco.transform.rotation.w = qw2

        self.broadcaster.sendTransform([tf_world, tf_aruco])
        self.published = True
        self.get_logger().info(
            f'[aruco-origin] Static TFs: '
            f'{self.camera_frame}→{self.world_frame} '
            f'origin=({t_cw[0]:.3f},{t_cw[1]:.3f},{t_cw[2]:.3f})m  |  '
            f'{self.world_frame}→aruco '
            f'pos=({self._aruco_offset[0]:.3f},{self._aruco_offset[1]:.3f},'
            f'{self._aruco_offset[2]:.3f})m')

    def _publish_static_tf(self, T_cam_world: np.ndarray, *, mode: str) -> None:
        """Publish static TF camera_frame → world_frame (floor fallback mode).

        T_cam_world is the 4×4 pose of the world (base) frame in camera frame.
        """
        R_cw = T_cam_world[:3, :3]
        t_cw = T_cam_world[:3, 3]
        qx, qy, qz, qw = _rot_to_quat(R_cw)
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = self.camera_frame
        tf.child_frame_id = self.world_frame
        tf.transform.translation.x = float(t_cw[0])
        tf.transform.translation.y = float(t_cw[1])
        tf.transform.translation.z = float(t_cw[2])
        tf.transform.rotation.x = qx
        tf.transform.rotation.y = qy
        tf.transform.rotation.z = qz
        tf.transform.rotation.w = qw
        self.broadcaster.sendTransform(tf)
        self.published = True
        self.get_logger().info(
            f'[{mode}] Static TF published: '
            f'{self.camera_frame} → {self.world_frame}  '
            f'origin=({t_cw[0]:.3f}, {t_cw[1]:.3f}, {t_cw[2]:.3f}) m')


# ── Pure functions ──────────────────────────────────────────────────────────

def _euler_xyz_to_R(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """Intrinsic XYZ Euler (degrees) → 3×3 rotation matrix."""
    rx, ry, rz = np.radians([rx_deg, ry_deg, rz_deg])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx  # intrinsic XYZ = extrinsic ZYX


def _R_to_euler_xyz(R: np.ndarray) -> tuple[float, float, float]:
    """3×3 rotation → intrinsic XYZ Euler angles (degrees). For logging only."""
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    if sy > 1e-6:
        rx = float(np.degrees(np.arctan2(R[2, 1], R[2, 2])))
        ry = float(np.degrees(np.arctan2(-R[2, 0], sy)))
        rz = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    else:
        rx = float(np.degrees(np.arctan2(-R[1, 2], R[1, 1])))
        ry = float(np.degrees(np.arctan2(-R[2, 0], sy)))
        rz = 0.0
    return rx, ry, rz


def _invert_se3(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Inverse of SE(3) [R | t]: returns [Rᵀ | -Rᵀ t]."""
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def _se3_average(matrices: list[np.ndarray]) -> np.ndarray:
    """Average a list of 4×4 SE(3) transforms (translation mean + quaternion mean)."""
    t_mean = np.mean([T[:3, 3] for T in matrices], axis=0)
    quats = np.array([_rot_to_quat(T[:3, :3]) for T in matrices])  # (N, 4) xyzw
    q0 = quats[0]
    for i in range(1, len(quats)):
        if float(np.dot(quats[i], q0)) < 0.0:
            quats[i] = -quats[i]
    q_mean = quats.mean(axis=0)
    q_mean /= float(np.linalg.norm(q_mean))
    R_mean = _quat_to_R(*q_mean)
    out = np.eye(4)
    out[:3, :3] = R_mean
    out[:3, 3] = t_mean
    return out


def _quat_to_R(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Quaternion (x, y, z, w) → 3×3 rotation matrix."""
    n = float(np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw))
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    return np.array([
        [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
    ])


def _rot_to_quat(R: np.ndarray) -> tuple[float, float, float, float]:
    """3×3 rotation → quaternion (x, y, z, w). Numerically stable."""
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    tr = m00 + m11 + m22
    if tr > 0.0:
        s = float(np.sqrt(tr + 1.0)) * 2.0
        qw = 0.25 * s
        qx = (m21 - m12) / s
        qy = (m02 - m20) / s
        qz = (m10 - m01) / s
    elif (m00 > m11) and (m00 > m22):
        s = float(np.sqrt(1.0 + m00 - m11 - m22)) * 2.0
        qw = (m21 - m12) / s
        qx = 0.25 * s
        qy = (m01 + m10) / s
        qz = (m02 + m20) / s
    elif m11 > m22:
        s = float(np.sqrt(1.0 + m11 - m00 - m22)) * 2.0
        qw = (m02 - m20) / s
        qx = (m01 + m10) / s
        qy = 0.25 * s
        qz = (m12 + m21) / s
    else:
        s = float(np.sqrt(1.0 + m22 - m00 - m11)) * 2.0
        qw = (m10 - m01) / s
        qx = (m02 + m20) / s
        qy = (m12 + m21) / s
        qz = 0.25 * s
    return float(qx), float(qy), float(qz), float(qw)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = WorldOriginNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
