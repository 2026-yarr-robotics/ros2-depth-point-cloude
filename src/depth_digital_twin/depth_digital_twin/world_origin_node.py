"""world_origin_node — define `world` frame relative to the camera.

Two modes, picked at startup based on `calibration_matrix_path`:

* **calibrated** — load a 4×4 hand-eye result that places `world` on the
  robot base. Two sub-modes per file naming convention:
    - `T_exo2base*.npy`  : eye-to-hand. Camera fixed in space; the matrix is
      already T_camera→base. Publish a single static TF, world := robot base.
    - `T_hand2base*.npy` : eye-in-hand. Camera moves with the EE. Requires a
      live source of T_base→hand (currently `/dsr01/joint_states` + a
      flange transform listener). Without that, the node logs and falls
      back to the floor-fit mode below.
  Translation unit defaults to mm (Doosan convention) and is converted to m
  for ROS via `calibration_translation_unit`.

* **floor-plane fit** (legacy default; activated when no calibration file is
  given): sample a depth patch around `(window_center_*_px)`, fit a plane via
  SVD, build a Z-up basis aligned with that plane, publish a static TF
  camera→world such that the floor patch lands at Z ≈ 0.

Downstream nodes (point_cloud_node) read tf2 with target=world, source=camera
and consume the result without knowing which mode is active.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import Image
from tf2_ros import StaticTransformBroadcaster

from depth_digital_twin.intrinsics import load_intrinsics


class WorldOriginNode(Node):
    def __init__(self) -> None:
        super().__init__('world_origin_node')
        self.declare_parameter('intrinsics_path', '')
        self.declare_parameter('depth_topic',
                               '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_frame', 'camera_color_optical_frame')
        self.declare_parameter('world_frame', 'world')
        self.declare_parameter('depth_unit', 0.001)  # mm → m
        self.declare_parameter('window_radius', 20)  # px patch radius for plane fit
        # Patch centre in pixels. -1 (default) means "image centre"; set to a
        # specific (u, v) when the floor is not under the image centre (e.g.
        # tilted camera, off-axis viewpoint).
        self.declare_parameter('window_center_x_px', -1)
        self.declare_parameter('window_center_y_px', -1)
        self.declare_parameter('min_patch_points', 100)  # min valid pixels to fit a plane
        self.declare_parameter('max_plane_residual', 0.01)  # m, reject noisy fits
        self.declare_parameter('samples_required', 10)
        # ----- Calibrated mode -----
        # Absolute path to a 4x4 hand-eye matrix (.npy). Naming convention:
        #   T_exo2base*.npy  → eye-to-hand (camera fixed; matrix = T_cam2base)
        #   T_hand2base*.npy → eye-in-hand (camera on EE; needs live robot pose)
        # Empty string disables calibrated mode (legacy floor-plane fit).
        self.declare_parameter('calibration_matrix_path', '')
        # Translation unit in the .npy file. Doosan saves mm; ROS uses m.
        self.declare_parameter('calibration_translation_unit', 'mm')  # mm | m
        # When eye-in-hand calibration is loaded but no robot pose is
        # available, the node would otherwise be stuck. Falling back to the
        # floor-fit mode keeps the rest of the pipeline running.
        self.declare_parameter('calibration_fallback_to_floor_fit', True)

        path = Path(self.get_parameter('intrinsics_path').value)
        if not path.is_file():
            raise FileNotFoundError(f'intrinsics_path not found: {path}')
        self.intr = load_intrinsics(path)
        self.get_logger().info(
            f'Loaded intrinsics {self.intr.width}x{self.intr.height} '
            f'fx={self.intr.fx:.2f} cx={self.intr.cx:.2f}')

        self.camera_frame: str = self.get_parameter('camera_frame').value
        self.world_frame: str = self.get_parameter('world_frame').value
        self.depth_unit: float = float(self.get_parameter('depth_unit').value)
        self.window_radius: int = int(self.get_parameter('window_radius').value)
        self.window_cx_px: int = int(self.get_parameter('window_center_x_px').value)
        self.window_cy_px: int = int(self.get_parameter('window_center_y_px').value)
        self.min_patch_points: int = int(self.get_parameter('min_patch_points').value)
        self.max_plane_residual: float = float(
            self.get_parameter('max_plane_residual').value)
        self.samples_required: int = int(self.get_parameter('samples_required').value)

        self.bridge = CvBridge()
        self.broadcaster = StaticTransformBroadcaster(self)
        # Accumulate (origin, R) samples; final TF uses median origin + averaged
        # rotation re-orthogonalised via SVD.
        self.origin_samples: deque[np.ndarray] = deque(maxlen=self.samples_required)
        self.R_samples: deque[np.ndarray] = deque(maxlen=self.samples_required)
        self.published = False

        # Try calibrated mode first. If the file exists and the convention is
        # eye-to-hand, we publish a static TF immediately and skip the floor
        # plane fit entirely. Otherwise drop through to the depth subscription.
        cal_path = str(self.get_parameter('calibration_matrix_path').value)
        # Log the raw parameter value so the user can see at startup whether
        # the launch's `calibration:=...` actually reached this node. If this
        # prints empty, the calibration arg never made it through; check the
        # launch wiring and re-build with --symlink-install.
        self.get_logger().info(
            f'calibration_matrix_path = {cal_path!r}')
        if cal_path:
            handled = self._setup_calibrated_mode(Path(cal_path))
            if handled:
                return  # Static TF published; depth subscription not needed.
            self.get_logger().warn(
                'Calibration mode did not handle startup; falling through to '
                'floor-plane fit. Check the WARN/ERROR above for the reason.')

        self.create_subscription(
            Image, self.get_parameter('depth_topic').value, self._on_depth, 10)
        self.get_logger().info('Waiting for centre depth samples…')

    # ------------------------------------------------------------------
    def _setup_calibrated_mode(self, cal_path: Path) -> bool:
        """Returns True iff calibrated mode handled startup (TF published)."""
        if not cal_path.is_file():
            self.get_logger().warn(
                f'calibration_matrix_path not found: {cal_path} — '
                'falling back to floor-plane fit.')
            return False
        try:
            T = np.load(cal_path)
        except Exception as e:  # noqa: BLE001 — surface load errors
            self.get_logger().error(
                f'Failed to load calibration matrix {cal_path}: {e} — '
                'falling back to floor-plane fit.')
            return False
        T = np.asarray(T, dtype=np.float64)
        if T.shape != (4, 4):
            self.get_logger().error(
                f'Calibration matrix must be 4x4, got {T.shape} — falling back.')
            return False
        R = T[:3, :3]
        t = T[:3, 3].copy()
        if not np.allclose(R @ R.T, np.eye(3), atol=1e-3):
            self.get_logger().error(
                'Calibration rotation is not orthogonal (R Rᵀ ≠ I) — '
                'falling back to floor-plane fit.')
            return False
        if abs(np.linalg.det(R) - 1.0) > 1e-3:
            self.get_logger().error(
                f'Calibration rotation det(R)={np.linalg.det(R):.4f} ≠ 1 — '
                'falling back to floor-plane fit.')
            return False

        unit = str(self.get_parameter('calibration_translation_unit').value).lower()
        if unit == 'mm':
            t = t * 1e-3
        elif unit != 'm':
            self.get_logger().warn(
                f'Unknown calibration_translation_unit={unit!r}; assuming "m".')
        if float(np.linalg.norm(t)) > 5.0:
            self.get_logger().warn(
                f'Calibration translation |t|={np.linalg.norm(t):.3f} m looks '
                f'unusually large — check `calibration_translation_unit` (got '
                f'{unit!r}).')

        name = cal_path.name.lower()
        if name.startswith('t_exo2base'):
            mode = 'exo'
        elif name.startswith('t_hand2base'):
            mode = 'hand'
        else:
            self.get_logger().warn(
                f'Calibration filename {cal_path.name!r} does not match '
                'T_exo2base*.npy or T_hand2base*.npy; assuming exo (eye-to-hand).')
            mode = 'exo'

        if mode == 'exo':
            # Eye-to-hand: camera is fixed; T = T_cam2base = pose of camera
            # in the base frame. Publishing camera→world (parent=camera,
            # child=world) means the static transform stored is "pose of
            # world in camera". world := base ⇒ pose of base in camera =
            # inv(T_cam2base).
            T_camera_world = _invert_se3(R, t)
            R_pub = T_camera_world[:3, :3]
            t_pub = T_camera_world[:3, 3]
            self._publish_calibrated(t_pub, R_pub, mode='exo (eye-to-hand)')
            self.get_logger().info(
                f'Loaded {cal_path.name} (exo mode). World = robot base.')
            return True

        # Eye-in-hand: requires live robot pose to assemble camera→base.
        # Phase 3 will wire up `/dsr01/joint_states` + a TF listener and
        # publish a *dynamic* camera→world. Until that lands, gracefully
        # degrade so the rest of the pipeline keeps running.
        if bool(self.get_parameter('calibration_fallback_to_floor_fit').value):
            self.get_logger().warn(
                f'{cal_path.name} is eye-in-hand (T_hand2base) but live robot '
                'pose integration is not implemented yet — falling back to '
                'floor-plane fit.')
            return False
        self.get_logger().error(
            f'{cal_path.name} requires live robot pose; calibrated_fallback '
            'is disabled. Node will not publish any TF.')
        return True  # don't subscribe to depth; refuse to fall back

    def _publish_calibrated(self, origin_cam: np.ndarray,
                            R_cw: np.ndarray, *, mode: str) -> None:
        qx, qy, qz, qw = _rot_to_quat(R_cw)
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = self.camera_frame
        tf.child_frame_id = self.world_frame
        tf.transform.translation.x = float(origin_cam[0])
        tf.transform.translation.y = float(origin_cam[1])
        tf.transform.translation.z = float(origin_cam[2])
        tf.transform.rotation.x = qx
        tf.transform.rotation.y = qy
        tf.transform.rotation.z = qz
        tf.transform.rotation.w = qw
        self.broadcaster.sendTransform(tf)
        self.published = True
        self.get_logger().info(
            f'Published static TF {self.camera_frame} -> {self.world_frame} '
            f'[{mode}]. origin=({origin_cam[0]:.3f}, {origin_cam[1]:.3f}, '
            f'{origin_cam[2]:.3f}) m')

    def _on_depth(self, msg: Image) -> None:
        if self.published:
            return
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        h, w = depth.shape[:2]
        cx_px = w // 2 if self.window_cx_px < 0 else self.window_cx_px
        cy_px = h // 2 if self.window_cy_px < 0 else self.window_cy_px
        # Clamp so the requested centre is at least within the image.
        cx_px = int(np.clip(cx_px, 0, w - 1))
        cy_px = int(np.clip(cy_px, 0, h - 1))
        r = self.window_radius

        # 1) Sample depth in a square patch around the image centre.
        y0, y1 = max(0, cy_px - r), min(h, cy_px + r + 1)
        x0, x1 = max(0, cx_px - r), min(w, cx_px + r + 1)
        patch = depth[y0:y1, x0:x1].astype(np.float32)
        if patch.size == 0:
            return
        ys, xs = np.mgrid[y0:y1, x0:x1]
        valid = patch > 0
        if int(valid.sum()) < self.min_patch_points:
            return

        # 2) Deproject every valid pixel into the camera optical frame.
        zs = patch[valid] * self.depth_unit
        us = xs[valid].astype(np.float32)
        vs = ys[valid].astype(np.float32)
        X = (us - self.intr.cx) * zs / self.intr.fx
        Y = (vs - self.intr.cy) * zs / self.intr.fy
        pts = np.stack([X, Y, zs], axis=1)  # (N, 3)

        # 3) Plane fit via SVD on centred points.
        p0 = pts.mean(axis=0)
        centred = pts - p0
        # Right-singular-vectors of the centred cloud; smallest = plane normal.
        _, S, Vt = np.linalg.svd(centred, full_matrices=False)
        normal = Vt[-1]
        # Reject noisy fits: smallest singular value scaled by sqrt(N) ≈ residual.
        residual = float(S[-1] / np.sqrt(max(1, pts.shape[0])))
        if residual > self.max_plane_residual:
            self.get_logger().warn(
                f'Plane fit residual {residual:.4f} m exceeds threshold '
                f'{self.max_plane_residual:.4f}; skipping frame',
                throttle_duration_sec=2.0)
            return

        # Orient normal toward the camera origin (so it points "up" out of the
        # floor). The camera optical centre is at the origin, so the vector from
        # the floor patch back to the camera is -p0; align normal with it.
        if float(np.dot(normal, -p0)) < 0.0:
            normal = -normal
        normal = normal / (np.linalg.norm(normal) + 1e-12)

        # 4) Build a Z-up basis on the plane.
        cam_x = np.array([1.0, 0.0, 0.0])
        x_world = cam_x - float(np.dot(cam_x, normal)) * normal
        if np.linalg.norm(x_world) < 1e-3:
            # Camera +X almost parallel to normal — fall back to camera +Y.
            cam_y = np.array([0.0, 1.0, 0.0])
            x_world = cam_y - float(np.dot(cam_y, normal)) * normal
        x_world = x_world / (np.linalg.norm(x_world) + 1e-12)
        y_world = np.cross(normal, x_world)
        # Columns of R = world axes expressed in the camera optical frame.
        # This is exactly T_camera_world's rotation (pose of world IN camera).
        R = np.column_stack([x_world, y_world, normal])

        self.origin_samples.append(p0)
        self.R_samples.append(R)
        if len(self.origin_samples) < self.samples_required:
            return

        origin = np.median(np.stack(self.origin_samples, axis=0), axis=0)
        R_mean = np.mean(np.stack(self.R_samples, axis=0), axis=0)
        # Re-orthogonalise via SVD; force det = +1 (right-handed).
        U, _, Vt2 = np.linalg.svd(R_mean)
        D = np.eye(3)
        D[2, 2] = float(np.sign(np.linalg.det(U @ Vt2)))
        R_final = U @ D @ Vt2
        self._publish(origin, R_final)

    def _publish(self, origin_cam: np.ndarray, R_cw: np.ndarray) -> None:
        # `R_cw` here is "rotation of world axes expressed in the camera frame",
        # i.e. the rotation part of T_camera_world (pose of world in camera).
        qx, qy, qz, qw = _rot_to_quat(R_cw)
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = self.camera_frame
        tf.child_frame_id = self.world_frame
        tf.transform.translation.x = float(origin_cam[0])
        tf.transform.translation.y = float(origin_cam[1])
        tf.transform.translation.z = float(origin_cam[2])
        tf.transform.rotation.x = qx
        tf.transform.rotation.y = qy
        tf.transform.rotation.z = qz
        tf.transform.rotation.w = qw
        self.broadcaster.sendTransform(tf)
        self.published = True
        # Log a useful summary: where world sits + how it is oriented.
        n = R_cw[:, 2]  # world +Z in camera frame
        tilt_deg = float(np.degrees(np.arccos(
            np.clip(np.dot(n, np.array([0.0, -1.0, 0.0])), -1.0, 1.0))))
        self.get_logger().info(
            f'Published static TF {self.camera_frame} -> {self.world_frame} '
            f'(plane-fit Z-up). origin=({origin_cam[0]:.3f}, '
            f'{origin_cam[1]:.3f}, {origin_cam[2]:.3f}) m, '
            f'camera tilt vs vertical-down ≈ {tilt_deg:.1f}°')


def _invert_se3(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Inverse of an SE(3) rigid transform: inv(R, t) = (Rᵀ, -Rᵀ t)."""
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def _rot_to_quat(R: np.ndarray) -> tuple[float, float, float, float]:
    """3x3 rotation → quaternion (x, y, z, w). Numerically stable variant."""
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    tr = m00 + m11 + m22
    if tr > 0.0:
        s = float(np.sqrt(tr + 1.0)) * 2.0  # s = 4w
        qw = 0.25 * s
        qx = (m21 - m12) / s
        qy = (m02 - m20) / s
        qz = (m10 - m01) / s
    elif (m00 > m11) and (m00 > m22):
        s = float(np.sqrt(1.0 + m00 - m11 - m22)) * 2.0  # s = 4x
        qw = (m21 - m12) / s
        qx = 0.25 * s
        qy = (m01 + m10) / s
        qz = (m02 + m20) / s
    elif m11 > m22:
        s = float(np.sqrt(1.0 + m11 - m00 - m22)) * 2.0  # s = 4y
        qw = (m02 - m20) / s
        qx = (m01 + m10) / s
        qy = 0.25 * s
        qz = (m12 + m21) / s
    else:
        s = float(np.sqrt(1.0 + m22 - m00 - m11)) * 2.0  # s = 4z
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
