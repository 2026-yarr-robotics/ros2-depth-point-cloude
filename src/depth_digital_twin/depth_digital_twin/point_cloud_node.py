"""point_cloud_node — fuse RGB + aligned depth + segmentation into:

* `/digital_twin/points`     — coloured PointCloud2 in the `world` frame
* `/digital_twin/boxes`      — MarkerArray of per-object 3D position boxes
                              (CUBE + LINE_LIST outline + TEXT label)
* `/digital_twin/box_debug`  — RGB image with the projected 3D boxes drawn on top

3D box estimation is tailored to cup-like objects: a standing cup is a
near-symmetric cylinder, so PCA yaw is unstable and the box is published
axis-aligned. A fallen cup has a clear elongation in the XY plane, so PCA on
the horizontal projection is used to recover its orientation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point as MsgPoint, Quaternion
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image, JointState, PointCloud2, PointField
from std_msgs.msg import ColorRGBA, Header, Int32MultiArray, String
from visualization_msgs.msg import Marker, MarkerArray

import message_filters
import tf2_ros
from std_srvs.srv import Trigger
from scipy.spatial import cKDTree
import tf2_ros
from std_srvs.srv import Trigger

from rcl_interfaces.msg import SetParametersResult
from geometry_msgs.msg import Vector3
from depth_digital_twin.intrinsics import load_intrinsics
from depth_digital_twin.cup_geometry import (cone_silhouette_px,
                                             edge_snap_fit,
                                             fit_silhouette_xy,
                                             ray_through_point)
from depth_digital_twin_msgs.msg import (CupObservation,
                                         CupObservationArray,
                                         SegmentedObjectArray,
                                         WorldObjectCloud,
                                         WorldObjectCloudArray)


# Edge index pairs for the 12 edges of a box given the 8-corner layout used
# by `_box_corners` below (bottom face 0..3, top face 4..7).
_BOX_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def _palette(i: int) -> tuple[float, float, float]:
    base = [
        (0.95, 0.26, 0.21),
        (0.30, 0.69, 0.31),
        (0.13, 0.59, 0.95),
        (1.00, 0.76, 0.03),
        (0.61, 0.15, 0.69),
        (0.00, 0.74, 0.83),
    ]
    return base[i % len(base)]


def _pack_rgb(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pack uint8 RGB into a float32 (PointCloud2 'rgb' field convention)."""
    rgb_uint = ((r.astype(np.uint32) << 16)
                | (g.astype(np.uint32) << 8)
                | b.astype(np.uint32))
    return np.frombuffer(rgb_uint.astype(np.uint32).tobytes(), dtype=np.float32)


def _classify_color_bgr(bgr: np.ndarray, allowed: list[str]) -> str | None:
    """Bucket the median HSV of a (N,3) BGR pixel block to a color name.

    Returns None if the chosen color isn't in `allowed` (caller will fall back
    to the existing track color). OpenCV hue range is 0–179.
    """
    if bgr.size == 0:
        return None
    hsv = cv2.cvtColor(bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    h_med = float(np.median(hsv[:, 0]))
    s_med = float(np.median(hsv[:, 1]))
    v_med = float(np.median(hsv[:, 2]))

    if v_med < 35:
        cand = 'black'
    elif s_med < 40:
        cand = 'white' if v_med > 200 else 'gray'
    elif h_med < 10 or h_med >= 170:
        cand = 'red'
    elif h_med < 23:
        cand = 'orange'
    elif h_med < 35:
        cand = 'yellow'
    elif h_med < 80:
        cand = 'green'
    elif h_med < 130:
        cand = 'blue'
    else:
        cand = 'purple'

    return cand if cand in allowed else None


class PositionKF:
    """Constant-position Kalman filter on a 3D world-frame centre (metres).

    Replaces the old EMA-smooth + scan-and-lock freeze. A cup is static
    between relocations, so the motion model is "stays put" with a small
    per-window process noise `q` that lets the estimate keep creeping toward
    fresh measurements — this is what continuously corrects a slightly-wrong
    fit, which the frozen LOCKED box never did.

    State x (3,) is the centre; covariance P (3x3) is diagonal in practice.
    `update()` gates each measurement on its Mahalanobis distance: a transient
    depth spike is rejected so the estimate holds, while a *sustained* run of
    rejections (handled by the caller) is treated as a real relocation and the
    filter is `reset()` to re-acquire. Q/R are supplied as per-axis variance
    vectors so the noisier Z axis can be weighted independently.
    """

    def __init__(self, x0: np.ndarray, p0_diag: np.ndarray,
                 q_diag: np.ndarray) -> None:
        self.x = np.asarray(x0, dtype=np.float64).copy()
        self.P = np.diag(np.asarray(p0_diag, dtype=np.float64))
        self.q = np.asarray(q_diag, dtype=np.float64)

    def predict(self) -> None:
        """Constant-position prediction: x unchanged, covariance grows by Q.
        Called once per window (even on a miss) so a re-appearing cup is not
        rejected by a stale, over-confident covariance."""
        self.P = self.P + np.diag(self.q)

    def update(self, z: np.ndarray, r_diag: np.ndarray,
               gate: float) -> tuple[bool, float]:
        """Gated Kalman measurement update. Returns (accepted, mahalanobis²).
        When the innovation's Mahalanobis distance exceeds `gate` (>0) the
        measurement is rejected and the state is left untouched."""
        z = np.asarray(z, dtype=np.float64)
        S = self.P + np.diag(np.asarray(r_diag, dtype=np.float64))
        Sinv = np.linalg.inv(S)
        y = z - self.x
        d2 = float(y @ Sinv @ y)
        if gate > 0.0 and d2 > gate:
            return False, d2
        K = self.P @ Sinv
        self.x = self.x + K @ y
        self.P = (np.eye(3) - K) @ self.P
        return True, d2

    def reset(self, z: np.ndarray, p0_diag: np.ndarray) -> None:
        """Re-acquire at a new measurement (cup relocated). Restores the large
        initial covariance so the next few measurements pull hard."""
        self.x = np.asarray(z, dtype=np.float64).copy()
        self.P = np.diag(np.asarray(p0_diag, dtype=np.float64))

    def position_std(self) -> float:
        """Worst-axis 1σ position uncertainty (m). Small ⇒ estimate settled."""
        return float(np.sqrt(max(0.0, float(np.max(np.diag(self.P))))))


class PointCloudNode(Node):
    def __init__(self) -> None:
        super().__init__('point_cloud_node')

        self.declare_parameter('intrinsics_path', '')
        self.declare_parameter('rgb_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic',
                               '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('detections_topic', '/digital_twin/detections')
        self.declare_parameter('points_topic', '/digital_twin/points')
        self.declare_parameter('boxes_topic', '/digital_twin/boxes')
        self.declare_parameter('box_debug_topic', '/digital_twin/box_debug')
        self.declare_parameter('depth_debug_topic', '/digital_twin/depth_debug')
        self.declare_parameter('camera_frame', 'camera_color_optical_frame')
        self.declare_parameter('world_frame', 'world')
        self.declare_parameter('depth_unit', 0.001)
        self.declare_parameter('downsample', 2)
        # Hard cap on accumulated points per track fed to the per-window fit /
        # density / publish passes (0 = unlimited). Bounds worst-case window
        # cost when a mask spikes; excess points are randomly subsampled.
        self.declare_parameter('max_points_per_track', 6000)
        # role = standalone (default: this node fits + Kalman-filters + emits
        # /boxes, the existing single-camera behaviour) | producer (fusion mode:
        # emit per-object world-frame clouds on WorldObjectCloudArray and let
        # cup_fusion_node do the fit/KF/merge — no fit/KF/markers here).
        self.declare_parameter('role', 'standalone')
        self.declare_parameter('camera_name', 'exo')
        self.declare_parameter('world_clouds_topic', '/digital_twin/world_clouds')
        # Hand motion gate (Phase C): when enabled, producer stamps moving=True
        # on its clouds whenever a joint velocity exceeds the threshold, so the
        # fusion node can drop a wrist-camera frame captured mid-motion.
        self.declare_parameter('hand_motion_gating', False)
        self.declare_parameter('joint_states_topic', '/joint_states')
        self.declare_parameter('joint_vel_thresh', 0.05)   # rad/s, max joint
        self.declare_parameter('z_min', 0.1)
        self.declare_parameter('z_max', 4.0)
        self.declare_parameter('box_line_width', 0.0015)
        self.declare_parameter('box_alpha', 0.25)
        self.declare_parameter('box_standing_ratio', 0.8)
        self.declare_parameter('box_min_elongation', 1.5)
        self.declare_parameter('box_force_aabb', False)
        # Pixels to erode the YOLO mask before sampling depth points for the
        # 3D box. YOLO seg boundaries are noisy and depth at object edges is
        # frequently a mixed/foreground+background pixel. 0 disables.
        self.declare_parameter('mask_erode_px', 3)
        # Depth-Laplacian threshold (m) for the mixed-pixel filter. A pixel is
        # rejected when |∇²z| exceeds this — i.e. it sits on a depth discontinuity
        # (cup silhouette vs background table) where the stereo correlator
        # averaged two surfaces and the deprojected point lands outside the
        # true geometry. 0 disables.
        self.declare_parameter('depth_gradient_max_m', 0.015)
        # Per-axis MAD-based outlier filter on the world-frame point cluster
        # before fitting the box. Drops points whose deviation from the median
        # on any axis exceeds k * 1.4826 * MAD (k≈3 ⇒ 3σ for Gaussian noise).
        # Catches single-pixel depth spikes (specular/transparent/mixed pixel)
        # that otherwise inflate the AABB. 0 disables.
        self.declare_parameter('box_outlier_mad_k', 3.0)
        self.declare_parameter('approx_sync_slop', 0.05)
        self.declare_parameter('objects_only', True)
        # World-frame re-association distance (m). When ByteTrack issues a new
        # instance_id for a detection whose world-frame centroid is within this
        # distance of an existing track, the new id is silently merged into the
        # existing track. Prevents ghost boxes when the camera moves fast and
        # ByteTrack re-ids the same physical cup.
        self.declare_parameter('track_world_merge_dist_m', 0.08)

        # ── Kalman position filter (replaces EMA + scan-and-lock) ───────────
        # Per-track constant-position Kalman filter on the cup/object centre.
        # The estimate is never frozen: every in-gate window nudges it toward
        # the fresh fit, so a slightly-wrong pose self-corrects over a few
        # windows instead of sticking until the cup physically moves >3 cm.
        #
        # Process noise (per window) — how fast the estimate may drift toward
        # new measurements. Larger ⇒ faster correction, slightly more jitter.
        self.declare_parameter('kf_process_std_xy_m', 0.002)
        self.declare_parameter('kf_process_std_z_m', 0.004)
        # Base measurement noise — the per-window fit's assumed 1σ error. Z is
        # noisier than XY on a stereo depth camera. Inflated adaptively by the
        # cup-fit residual (see kf_resid_infl).
        self.declare_parameter('kf_meas_std_xy_m', 0.005)
        self.declare_parameter('kf_meas_std_z_m', 0.010)
        # Initial / re-acquire covariance std — large so the first measurement
        # (or a confirmed relocation) is trusted almost fully.
        self.declare_parameter('kf_init_std_m', 0.05)
        # Mahalanobis² gate (3 dof). A measurement beyond this is rejected as a
        # transient spike. 9.0 ≈ χ²(3) at ~97%. 0 disables gating.
        self.declare_parameter('kf_gate_mahalanobis', 9.0)
        # Consecutive gated-out windows before the run is judged a real
        # relocation (robot pick / place) and the filter re-acquires there.
        # Rejects 1-frame spikes while still following a moved cup.
        self.declare_parameter('kf_reacquire_windows', 3)
        # Position 1σ (m) below which the track is reported "settled" ([S] tag).
        # Replaces the old binary LOCKED state — purely informational now.
        self.declare_parameter('kf_settled_std_m', 0.006)
        # Measurement-noise inflation per unit of (residual / cup_fit_residual_max).
        # Down-weights high-residual fits so a poor window can't yank the
        # estimate. 0 disables (constant R).
        self.declare_parameter('kf_resid_infl', 1.0)

        # ── Color classification + cups_on_table publish ──────────────────
        # Per-track color votes are accumulated from each frame's masked
        # pixels (HSV bucket). The argmax is published as the track's color
        # and counted into /cups_on_table — minus any track ids currently in
        # /stack_track_ids (cups vision-node says are stacked).
        self.declare_parameter('cups_on_table_topic', '/cups_on_table')
        self.declare_parameter('stack_track_ids_topic', '/stack_track_ids')
        # Colors recognised by the HSV bucket. Names match the palette used
        # downstream by the bridge / verifier.
        self.declare_parameter(
            'color_classes',
            ['red', 'orange', 'yellow', 'green', 'blue', 'purple',
             'white', 'black'])
        # Min pixel count in the mask before a frame contributes a color
        # vote (avoid tiny masks with mostly edge noise).
        self.declare_parameter('color_min_pixels', 64)

        # ----- Cup model (truncated-cone prior; standing only) -----
        self.declare_parameter('cup_top_diameter_m', 0.054)
        self.declare_parameter('cup_bottom_diameter_m', 0.078)
        self.declare_parameter('cup_height_m', 0.095)
        self.declare_parameter('cup_polygon_segments', 24)
        self.declare_parameter('cup_track_keepalive_frames', 10)
        self.declare_parameter('cup_fit_residual_max', 0.02)
        self.declare_parameter('cup_class_names', ['cup'])

        # ── Rim/silhouette observations (depth-light measurement path) ─────
        # Per-cup 2D silhouette chamfer fit (cup_geometry.fit_silhouette_xy):
        # the YOLO mask contour + calibration constrain the axis (x, y) far
        # more precisely than the depth cloud. Published as CupObservationArray
        # ALONGSIDE the existing cloud path (Phase 1: parallel instrumentation;
        # cup_fusion_node logs/compares, /digital_twin/boxes is unchanged).
        self.declare_parameter('rim_obs_enabled', True)
        self.declare_parameter('cup_obs_topic', '/digital_twin/cup_obs')
        # Min seconds between silhouette fits per track (a fit is ~5-10 ms;
        # throttling bounds worst-case callback cost with many cups).
        self.declare_parameter('rim_fit_period_s', 0.3)
        # Assumed 1σ contour-pixel noise; inflated x3 when the wrist camera is
        # moving and x2 when the mask is cut by the image border.
        self.declare_parameter('rim_sigma_px', 2.0)
        # Drop fits whose rendered-silhouette-vs-mask IoU is below this (the
        # consumer applies its own, stricter gate on top).
        self.declare_parameter('rim_min_iou', 0.2)
        # 3rd fit parameter b: uniform mask-boundary bias (px). Absorbs YOLO
        # over/under-segmentation so it cannot leak into (x, y); reported per
        # observation as boundary_offset_px.
        self.declare_parameter('rim_boundary_offset', True)
        # Edge-snap refinement: re-anchor the boundary to the image-gradient
        # peak along each silhouette normal (the mask only initialises). Off
        # → the chamfer/mask fit is published as-is.
        self.declare_parameter('rim_edge_snap', True)
        self.declare_parameter('rim_edge_min_grad', 8.0)
        # Max seconds a frame may wait for its stamped TF before falling back
        # to latest (fallback frames produce NO rim observation on a moving
        # camera — the image↔FK pairing would be wrong).
        self.declare_parameter('rim_tf_defer_s', 0.25)
        # Emit world point clouds for UPRIGHT cups. False = rim observations
        # are the sole upright measurement (fusion fit_source=rim) and the
        # noisy upright clouds are not built at all — skips the per-window
        # MAD/density/serialize passes for those tracks. Clouds for other
        # classes (fallen-cup → OBB) are always kept.
        self.declare_parameter('upright_clouds', True)
        # Image-space debug overlay of every ATTEMPTED silhouette fit:
        # observed contour (green), fitted silhouette (cyan), depth init
        # (red dot), per-cup text incl. the failure reason when a fit drops.
        self.declare_parameter('rim_debug_topic', '/digital_twin/rim_debug')

        # Accumulating-window pipeline. Per-frame depth is too noisy at one
        # shot; we ingest into per-track buffers and only fit + publish at
        # `window_period_s` cadence. With more samples per cluster the MAD
        # filter removes flicker outliers far more reliably and the cup-axis
        # fit becomes stable.
        self.declare_parameter('window_period_s', 0.5)

        # Mirror the floor-patch parameters (owned by world_origin_node, shared
        # via the /**: scope in params.yaml) so we can draw the same patch
        # rectangle on /digital_twin/depth_debug for visual sanity-checking.
        self.declare_parameter('window_radius', 30)
        self.declare_parameter('window_center_x_px', -1)
        self.declare_parameter('window_center_y_px', -1)

        # ArUco marker axes overlay on box_debug (uses calibrated TF, not real-time detect)
        self.declare_parameter('aruco_overlay', True)
        self.declare_parameter('world_marker_length_m', 0.05)

        # ── Top-rim & spatial-density filtering (point-cloud noise suppression) ──
        # Top-rim: keep only points within `top_rim_band_m` of the cluster's
        # `top_rim_percentile`-th height — isolates the cup rim for a clean
        # rendered cloud. Set band ≤ 0 to keep the whole cluster (no top-rim cut).
        self.declare_parameter('top_rim_band_m', 0.008)
        self.declare_parameter('top_rim_percentile', 95.0)
        self.declare_parameter('use_spatial_density_filter', True)
        self.declare_parameter('spatial_density_radius_m', 0.03)
        self.declare_parameter('spatial_density_min_neighbors', 5)

        path = Path(self.get_parameter('intrinsics_path').value)
        if not path.is_file():
            raise FileNotFoundError(f'intrinsics_path not found: {path}')
        self.intr = load_intrinsics(path)
        self.K = self.intr.K

        self.camera_frame: str = self.get_parameter('camera_frame').value
        self.world_frame: str = self.get_parameter('world_frame').value
        self.depth_unit: float = float(self.get_parameter('depth_unit').value)
        self.downsample: int = max(1, int(self.get_parameter('downsample').value))
        self.max_points_per_track: int = max(
            0, int(self.get_parameter('max_points_per_track').value))
        self.role: str = str(self.get_parameter('role').value).strip().lower()
        self.camera_name: str = str(self.get_parameter('camera_name').value)
        # Set by the joint-velocity gate (Phase C); producer stamps it onto each
        # emitted cloud so the fusion node can down-weight a moving hand view.
        self._joints_moving: bool = False
        self._motion_gating: bool = bool(
            self.get_parameter('hand_motion_gating').value)
        self._joint_vel_thresh: float = float(
            self.get_parameter('joint_vel_thresh').value)
        self._last_js: tuple | None = None
        self.z_min: float = float(self.get_parameter('z_min').value)
        self.z_max: float = float(self.get_parameter('z_max').value)
        self.box_line_w: float = float(self.get_parameter('box_line_width').value)
        self.box_alpha: float = float(self.get_parameter('box_alpha').value)
        self.standing_ratio: float = float(self.get_parameter('box_standing_ratio').value)
        self.min_elongation: float = float(self.get_parameter('box_min_elongation').value)
        self.force_aabb: bool = bool(self.get_parameter('box_force_aabb').value)
        self.mask_erode_px: int = max(0, int(self.get_parameter('mask_erode_px').value))
        self.depth_grad_max: float = max(
            0.0, float(self.get_parameter('depth_gradient_max_m').value))
        self.outlier_mad_k: float = max(
            0.0, float(self.get_parameter('box_outlier_mad_k').value))
        self.objects_only: bool = bool(self.get_parameter('objects_only').value)
        self.cup_top_d: float = float(self.get_parameter('cup_top_diameter_m').value)
        self.cup_bot_d: float = float(self.get_parameter('cup_bottom_diameter_m').value)
        self.cup_h: float = float(self.get_parameter('cup_height_m').value)
        self.cup_n_seg: int = max(8, int(self.get_parameter('cup_polygon_segments').value))
        self.cup_keepalive: int = int(self.get_parameter('cup_track_keepalive_frames').value)
        self._track_merge_dist: float = float(
            self.get_parameter('track_world_merge_dist_m').value)
        # ── Kalman position-filter coefficients (per-axis variance vectors) ──
        _q_xy = float(self.get_parameter('kf_process_std_xy_m').value)
        _q_z = float(self.get_parameter('kf_process_std_z_m').value)
        self._kf_q = np.array([_q_xy ** 2, _q_xy ** 2, _q_z ** 2],
                              dtype=np.float64)
        _r_xy = float(self.get_parameter('kf_meas_std_xy_m').value)
        _r_z = float(self.get_parameter('kf_meas_std_z_m').value)
        self._kf_r = np.array([_r_xy ** 2, _r_xy ** 2, _r_z ** 2],
                              dtype=np.float64)
        _p0 = float(self.get_parameter('kf_init_std_m').value)
        self._kf_p0 = np.array([_p0 ** 2, _p0 ** 2, _p0 ** 2], dtype=np.float64)
        self._kf_gate: float = float(
            self.get_parameter('kf_gate_mahalanobis').value)
        self._kf_reacquire_windows: int = max(
            1, int(self.get_parameter('kf_reacquire_windows').value))
        self._kf_settled_std: float = float(
            self.get_parameter('kf_settled_std_m').value)
        self._kf_resid_infl: float = max(
            0.0, float(self.get_parameter('kf_resid_infl').value))
        self._color_classes: list[str] = [
            str(c).lower()
            for c in self.get_parameter('color_classes').value]
        self._color_min_pixels: int = max(
            1, int(self.get_parameter('color_min_pixels').value))
        # depth-track ids that vision-node currently reports as stacked. These
        # are subtracted from /cups_on_table so a "stacked" cup is not double
        # counted (verifier owns the slot, depth owns the table count).
        self._stacked_ids: set[int] = set()
        self.cup_resid_max: float = float(self.get_parameter('cup_fit_residual_max').value)
        self.cup_class_names: set[str] = {
            s.lower() for s in self.get_parameter('cup_class_names').value}
        self._rim_enabled: bool = bool(
            self.get_parameter('rim_obs_enabled').value)
        self._rim_fit_period: float = max(
            0.05, float(self.get_parameter('rim_fit_period_s').value))
        self._rim_sigma_px: float = float(
            self.get_parameter('rim_sigma_px').value)
        self._rim_min_iou: float = float(
            self.get_parameter('rim_min_iou').value)
        self._rim_boundary_offset: bool = bool(
            self.get_parameter('rim_boundary_offset').value)
        self._rim_edge_snap: bool = bool(
            self.get_parameter('rim_edge_snap').value)
        self._rim_edge_min_grad: float = float(
            self.get_parameter('rim_edge_min_grad').value)
        self._tf_defer_s: float = float(
            self.get_parameter('rim_tf_defer_s').value)
        self._tf_pending: list = []   # frames awaiting their stamped TF
        self._upright_clouds: bool = bool(
            self.get_parameter('upright_clouds').value)
        if not self._upright_clouds and \
                str(self.get_parameter('role').value).strip().lower() \
                != 'producer':
            self.get_logger().warning(
                'upright_clouds=false in STANDALONE role: upright cups get '
                'no cloud, no fit, no /digital_twin/boxes output. This mode '
                'only makes sense feeding cup_fusion_node (role=producer).')
        self.patch_radius: int = max(1, int(self.get_parameter('window_radius').value))
        self.patch_cx_px: int = int(self.get_parameter('window_center_x_px').value)
        self.patch_cy_px: int = int(self.get_parameter('window_center_y_px').value)
        self.window_period_s: float = max(
            1e-3, float(self.get_parameter('window_period_s').value))

        # ── Top-rim & spatial-density filtering parameters ──────────────────
        self._top_rim_band: float = float(
            self.get_parameter('top_rim_band_m').value)
        self._top_rim_pct: float = float(
            self.get_parameter('top_rim_percentile').value)
        self._use_spatial_density: bool = bool(
            self.get_parameter('use_spatial_density_filter').value)
        self._spatial_density_radius: float = float(
            self.get_parameter('spatial_density_radius_m').value)
        self._spatial_density_min_neighbors: int = int(
            self.get_parameter('spatial_density_min_neighbors').value)

        # Tracks keyed by Ultralytics ByteTrack instance id (forwarded via
        # SegmentedObject.instance_id from detection_node). Per-track state:
        #   class_name              str
        #   kf                      PositionKF | None — 3D centre filter (lazy)
        #   center_xy               np.ndarray — cached kf.x[:2] (re-association)
        #   reacquire_count         int — consecutive gated-out (relocation) windows
        #   settled                 bool — kf position 1σ ≤ kf_settled_std
        #   points_buf, colors_buf  list[np.ndarray] — accumulated within window
        #   miss                    int — windows without any new points
        #   last_state              dict | None — last successful fit / render
        #   last_score, last_display_name, last_residual — for the label
        self._tracks: dict[int, dict] = {}
        self._last_published_ids: set[int] = set()
        self._window_start_stamp = None  # rclpy.time.Time, set on first frame

        slop: float = float(self.get_parameter('approx_sync_slop').value)

        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        latched = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)

        self.points_pub = self.create_publisher(
            PointCloud2, self.get_parameter('points_topic').value, 5)
        self.boxes_pub = self.create_publisher(
            MarkerArray, self.get_parameter('boxes_topic').value, latched)
        self.box_debug_pub = self.create_publisher(
            Image, self.get_parameter('box_debug_topic').value, 1)
        self.depth_debug_pub = self.create_publisher(
            Image, self.get_parameter('depth_debug_topic').value, 1)
        # JSON {color: count} of cups on the table, EXCLUDING any track id
        # vision-node has reported as occupying a stack slot. Latched so a
        # late-joining subscriber sees the most recent snapshot.
        self.cups_on_table_pub = self.create_publisher(
            String, self.get_parameter('cups_on_table_topic').value, latched)
        # Silhouette observations — published in BOTH roles (standalone tools
        # may log them too); consumed by cup_fusion_node in fusion mode.
        self.cup_obs_pub = None
        self.rim_debug_pub = None
        if self._rim_enabled:
            self.cup_obs_pub = self.create_publisher(
                CupObservationArray,
                self.get_parameter('cup_obs_topic').value, 5)
            self.rim_debug_pub = self.create_publisher(
                Image, self.get_parameter('rim_debug_topic').value, 1)
        # Producer-role output: per-object world clouds for cup_fusion_node.
        self.world_clouds_pub = None
        if self.role == 'producer':
            self.world_clouds_pub = self.create_publisher(
                WorldObjectCloudArray,
                self.get_parameter('world_clouds_topic').value, 5)
            self.get_logger().info(
                f"point_cloud_node role=producer (camera='{self.camera_name}') "
                f"→ {self.get_parameter('world_clouds_topic').value}; "
                f"fit/KF/markers deferred to cup_fusion_node")
            if self._motion_gating:
                self.create_subscription(
                    JointState,
                    str(self.get_parameter('joint_states_topic').value),
                    self._on_joint_states, 10)
                self.get_logger().info(
                    f'hand motion gating ON (|q̇| > {self._joint_vel_thresh} '
                    f'rad/s ⇒ moving)')
        self.create_subscription(
            Int32MultiArray,
            self.get_parameter('stack_track_ids_topic').value,
            self._on_stack_track_ids, 10)

        rgb_sub = message_filters.Subscriber(
            self, Image, self.get_parameter('rgb_topic').value)
        depth_sub = message_filters.Subscriber(
            self, Image, self.get_parameter('depth_topic').value)
        det_sub = message_filters.Subscriber(
            self, SegmentedObjectArray, self.get_parameter('detections_topic').value)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub, det_sub], queue_size=10, slop=slop)
        self.sync.registerCallback(self._on_synced)
        self._trigger_scan_srv = self.create_service(
            Trigger, '~/trigger_scan', self._on_trigger_scan)
        self.get_logger().info('point_cloud_node ready (waiting for synced frames)')

        # ── ArUco axis overlay (TF-based, not real-time detection) ────────
        self._aruco_overlay = bool(self.get_parameter('aruco_overlay').value)
        self._aruco_axis_len = float(
            self.get_parameter('world_marker_length_m').value) * 0.8

        # Live-tunable point-extraction knobs — `ros2 param set
        # /point_cloud_node_exo depth_gradient_max_m 0.08` etc. takes effect
        # immediately so each camera can be dialed in without a relaunch.
        self._tunable = {
            'depth_gradient_max_m': ('depth_grad_max', float),
            'max_points_per_track': ('max_points_per_track', int),
            'mask_erode_px': ('mask_erode_px', int),
            'downsample': ('downsample', int),
            'z_min': ('z_min', float),
            'z_max': ('z_max', float),
        }
        self.add_on_set_parameters_callback(self._on_set_params)

    def _on_set_params(self, params):
        for p in params:
            spec = self._tunable.get(p.name)
            if spec is not None:
                attr, cast = spec
                setattr(self, attr, cast(p.value))
        return SetParametersResult(successful=True)

    # ------------------------------------------------------------------
    def _on_joint_states(self, msg: JointState) -> None:
        """Estimate max joint speed; set _joints_moving for the producer to
        stamp onto its clouds. (Phase C hand motion gate.)"""
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        pos = np.asarray(msg.position, dtype=np.float64)
        if self._last_js is not None:
            t0, p0 = self._last_js
            dt = t - t0
            if dt > 1e-3 and pos.shape == p0.shape and pos.size:
                v = float(np.max(np.abs(pos - p0)) / dt)
                # Schmitt trigger: velocity noise hovering AT the threshold
                # made the flag flap, which made the fusion's drop-moving
                # gate discard hand observations intermittently.
                if v > self._joint_vel_thresh:
                    self._joints_moving = True
                elif v < 0.5 * self._joint_vel_thresh:
                    self._joints_moving = False
        self._last_js = (t, pos)

    # ------------------------------------------------------------------
    def _on_synced(self, rgb_msg: Image, depth_msg: Image,
                   det_msg: SegmentedObjectArray) -> None:
        """Per-frame entry: resolve world←camera AT IMAGE-CAPTURE TIME, then
        process. When FK lags the image (wrist camera: /tf for this stamp has
        not arrived yet) the frame is DEFERRED and retried on the next
        callback instead of silently using the latest TF — that fallback
        pairs an old pose with a new image and was a direct injector of
        image↔trajectory desync into every world-frame measurement. Only a
        frame older than `rim_tf_defer_s` falls back to latest, marked
        tf_stamped_ok=False (rim observations from a moving camera skip it).
        """
        retry = self._tf_pending[:3]      # bounded drain: no burst stalls
        carry = self._tf_pending[3:]
        self._tf_pending = []
        for item in retry:
            try:
                self._try_process_frame(*item)
            except Exception as e:           # one bad frame must not kill spin
                self.get_logger().error(
                    f'frame processing failed: {e}', throttle_duration_sec=5.0)
        self._tf_pending.extend(carry)
        try:
            self._try_process_frame(rgb_msg, depth_msg, det_msg,
                                    self._joints_moving)
        except Exception as e:
            self.get_logger().error(
                f'frame processing failed: {e}', throttle_duration_sec=5.0)

    def _try_process_frame(self, rgb_msg, depth_msg, det_msg,
                           moving_at=None) -> None:
        # Latch the motion state AT CAPTURE: a deferred frame retried
        # 0.1-0.25 s later must not be judged by the CURRENT flag (the
        # arm may have stopped — the frame is still mid-motion data).
        if moving_at is None:
            moving_at = self._joints_moving
        stamp = rclpy.time.Time.from_msg(rgb_msg.header.stamp)
        tf_stamped_ok = True
        try:
            tf = self.tf_buffer.lookup_transform(
                self.world_frame, self.camera_frame, stamp,
                # Non-blocking: in a single-threaded executor the /tf callback
                # can't run while THIS callback blocks. Fail fast → defer.
                timeout=rclpy.duration.Duration(seconds=0.0))
        except tf2_ros.ExtrapolationException:
            age_s = (self.get_clock().now() - stamp).nanoseconds * 1e-9
            if age_s < self._tf_defer_s:
                # FK not in the buffer yet — requeue for the next callback.
                if len(self._tf_pending) < 8:
                    self._tf_pending.append(
                        (rgb_msg, depth_msg, det_msg, moving_at))
                else:
                    self.get_logger().warning(
                        'TF defer queue full — dropping a frame',
                        throttle_duration_sec=5.0)
                return
            try:
                tf = self.tf_buffer.lookup_transform(
                    self.world_frame, self.camera_frame, rclpy.time.Time())
                tf_stamped_ok = False
                self.get_logger().info(
                    f'TF fallback→latest (stamp lag {age_s*1e3:.0f}ms, '
                    f'moving={self._joints_moving})',
                    throttle_duration_sec=10.0)
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException):
                self.get_logger().warn(
                    f'TF {self.world_frame}<-{self.camera_frame} not available '
                    f'(neither at stamp nor latest)',
                    throttle_duration_sec=2.0)
                return
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException):
            self.get_logger().warn(
                f'TF {self.world_frame}<-{self.camera_frame} not available yet',
                throttle_duration_sec=2.0)
            return
        self._process_frame(rgb_msg, depth_msg, det_msg, tf,
                            tf_stamped_ok, moving_at)

    def _process_frame(self, rgb_msg, depth_msg, det_msg, tf,
                       tf_stamped_ok: bool, moving_at: bool = False) -> None:
        """Ingest detections into per-track buffers + emit live debug images.
        Heavy work (filter, fit, marker/cloud publish) is deferred to
        `_finalize_window` which fires every `window_period_s`."""
        # tf is target=world, source=camera ⇒ p_world = R_wc @ p_cam + t_wc.
        t_wc = np.array([tf.transform.translation.x,
                         tf.transform.translation.y,
                         tf.transform.translation.z], dtype=np.float64)
        q = tf.transform.rotation
        R_wc = _quat_to_rot(q.x, q.y, q.z, q.w)

        # Diagnostics: print frame-to-frame Δ AND cumulative Δ since the last
        # logged line, plus the tf stamp actually used.  If tf_stamp doesn't
        # advance → tf_buffer isn't receiving fresh /tf (RSP / joint_states /
        # subscription problem).  If tf_stamp advances but pos doesn't →
        # joint values are constant (recorded sequence motionless).
        if not hasattr(self, '_last_t_wc_frame'):
            self._last_t_wc_frame = t_wc.copy()
            self._last_t_wc_log = t_wc.copy()
            self._tf_max_frame_mm = 0.0
        frame_d_mm = float(
            np.linalg.norm(t_wc - self._last_t_wc_frame)) * 1000.0
        self._tf_max_frame_mm = max(self._tf_max_frame_mm, frame_d_mm)
        self._last_t_wc_frame = t_wc.copy()

        cum_d_mm = float(
            np.linalg.norm(t_wc - self._last_t_wc_log)) * 1000.0
        ts = tf.header.stamp
        # Use a local conditional so the throttled INFO does no work most ticks.
        log_now = self.get_logger().info(
            f'world<-{self.camera_frame} '
            f'pos=({t_wc[0]:+.3f},{t_wc[1]:+.3f},{t_wc[2]:+.3f})m '
            f'  Δ_frame={frame_d_mm:.1f}mm '
            f'  Δ_cum={cum_d_mm:.1f}mm '
            f'  max_frame={self._tf_max_frame_mm:.1f}mm '
            f'  tf_stamp={ts.sec}.{ts.nanosec:09d}',
            throttle_duration_sec=2.0)
        # Only reset cumulative tracking after a log actually went out, but
        # rclpy throttle doesn't expose that.  Approximate by resetting every
        # call — cum then = inter-frame.  Use max_frame above to spot any spike.
        self._last_t_wc_log = t_wc.copy()
        self._tf_max_frame_mm = 0.0

        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        h, w = depth.shape[:2]
        if rgb.shape[:2] != (h, w):
            self.get_logger().warning(
                f'RGB ({rgb.shape[:2]}) != depth ({h, w}); requires aligned '
                f'depth', once=True)
            return

        z = depth.astype(np.float32) * self.depth_unit
        valid = (z > self.z_min) & (z < self.z_max)
        if self.depth_grad_max > 0.0:
            # Mixed-pixel reject: |∇²z| at object silhouettes is large because
            # the stereo correlator averaged the cup surface with the background.
            # Killing those pixels collapses the inflated "ring" outside the cup.
            gz = cv2.Laplacian(z, cv2.CV_32F, ksize=3)
            valid &= np.abs(gz) < self.depth_grad_max

        union_mask = np.zeros((h, w), dtype=bool)
        per_object_masks: list[tuple[object, np.ndarray]] = []
        for obj in det_msg.objects:
            m = self.bridge.imgmsg_to_cv2(obj.mask, desired_encoding='mono8')
            if m.shape[:2] != (h, w):
                continue
            mb = m > 0
            union_mask |= mb
            per_object_masks.append((obj, mb))

        # Per-frame: depth debug stream is independent of detection success.
        self._publish_depth_debug(depth_msg, z, valid, union_mask)

        # Per-frame: ingest each detection's world points into its track's
        # accumulating buffer. No fitting / cloud / marker publish here — the
        # window timer below batches that into one ~window_period_s update.
        # (Exception: the throttled per-track silhouette fit below, which must
        # pair a mask with the TF of ITS OWN frame — a moving hand camera
        # cannot defer that pairing to the window.)
        rim_obs: list = []
        # Always-on canvas: the debug stream must be a LIVE video (panel 3D
        # pane), not a sparse one that only updates when a throttled fit
        # fires — that read as "rim detection randomly stops".
        rim_dbg = {'canvas': None, 'gray': None, 'mag': None, 'fresh': set()}
        if (self.rim_debug_pub is not None and self._rim_enabled
                and self.rim_debug_pub.get_subscription_count() > 0):
            rim_dbg['canvas'] = rgb.copy()
        for obj, mb in per_object_masks:
            if rim_dbg['canvas'] is not None:
                self._draw_det_overlay(rim_dbg['canvas'], obj, mb)
            mb_box = self._erode_mask(mb)
            if mb_box.sum() < 32:
                mb_box = mb
            obj_valid = mb_box & valid
            if obj_valid.sum() < 32:
                continue
            oy, ox = np.where(obj_valid)
            # Apply the configured pixel stride (this param was declared+read but
            # never actually used). Bounds points-per-frame so a large/close
            # mask can't blow up the per-window cost — every downstream pass
            # (vstack, MAD, least_squares fit, KDTree density, serialize) is ~O(N).
            if self.downsample > 1:
                oy = oy[::self.downsample]
                ox = ox[::self.downsample]
            oz = z[oy, ox]
            ocx_c = (ox.astype(np.float32) - self.intr.cx) * oz / self.intr.fx
            ocy_c = (oy.astype(np.float32) - self.intr.cy) * oz / self.intr.fy
            obj_cam = np.stack([ocx_c, ocy_c, oz], axis=1)
            obj_world = (R_wc @ obj_cam.T).T + t_wc
            if obj_world.shape[0] < 16:
                continue
            bgr = rgb[oy, ox]
            obj_rgb_packed = _pack_rgb(bgr[:, 2], bgr[:, 1], bgr[:, 0])

            inst_id = int(getattr(obj, 'instance_id', -1))
            if inst_id < 0:
                # ByteTrack hasn't promoted this detection yet — skip rather
                # than mint a synthetic id that would collide with future
                # tracker ids.
                continue
            class_name = (obj.class_name or '').lower()
            centroid_xy = np.median(obj_world[:, :2], axis=0)
            tid = self._resolve_track_id(inst_id, centroid_xy)
            track = self._tracks.get(tid)
            if track is None:
                track = {
                    'class_name': class_name,
                    'kf': None,                # PositionKF, created on first fit
                    'center_xy': np.asarray(centroid_xy, dtype=np.float64).copy(),
                    'reacquire_count': 0,      # consecutive gated-out windows
                    'settled': False,          # kf position 1σ ≤ kf_settled_std
                    'points_buf': [],
                    'colors_buf': [],
                    'miss': 0,
                    'last_state': None,
                    'last_score': 0.0,
                    'last_display_name': obj.class_name or class_name,
                    'last_residual': 0.0,
                    'color': None,             # argmax of color_votes
                    'color_votes': {},         # {color_name: vote_count}
                }
                self._tracks[tid] = track
            # Upright cups measured by rim need no cloud — skipping the
            # buffer here removes their whole per-window cost (MAD, KDTree,
            # serialize). obj_world above is still computed: rim needs the
            # rough z (level) and xy init from it.
            track['seen'] = True
            if not tf_stamped_ok and moving_at:
                # Mis-paired frame (latest-TF fallback while the camera
                # moved): its world points are smeared — keep them out of
                # the cloud buffer too, not just out of the rim/ztop paths.
                pass
            elif self._upright_clouds or class_name not in self.cup_class_names:
                track['points_buf'].append(obj_world)
                track['colors_buf'].append(obj_rgb_packed)
            track['last_score'] = float(obj.score)
            # Per-frame color vote from the masked pixels.
            if bgr.shape[0] >= self._color_min_pixels:
                color = _classify_color_bgr(bgr, self._color_classes)
                if color is not None:
                    track['color_votes'][color] = \
                        track['color_votes'].get(color, 0) + 1
                    track['color'] = max(
                        track['color_votes'],
                        key=track['color_votes'].get)
            track['last_display_name'] = obj.class_name

            if (self.cup_obs_pub is not None
                    and class_name in self.cup_class_names):
                # Rough top-z history per FRAME (not per fit attempt): the
                # 97th depth percentile of a single frame jitters cm-scale
                # and used to flip the level snap downstream — a running
                # median over ~0.5 s stabilises the level evidence.
                # Do not let a mis-paired (fallback-TF while moving) frame
                # contaminate the level-evidence median that the NEXT seven
                # good fits will consume.
                if tf_stamped_ok or not moving_at:
                    hist = track.setdefault('ztop_hist', [])
                    hist.append(float(np.percentile(obj_world[:, 2], 97.0)))
                    if len(hist) > 7:
                        hist.pop(0)
                ob = self._rim_observe(
                    obj, mb, oz, obj_world, R_wc, t_wc, track, class_name,
                    tid, rgb, rim_dbg, tf_stamped_ok, moving_at)
                if ob is not None:
                    rim_obs.append(ob)

        if rim_obs and self.cup_obs_pub is not None:
            self.cup_obs_pub.publish(CupObservationArray(
                header=Header(stamp=rgb_msg.header.stamp,
                              frame_id=self.world_frame),
                observations=rim_obs))
        if rim_dbg['canvas'] is not None:
            canvas = rim_dbg['canvas']
            if self._aruco_overlay:
                # ArUco marker + world(base) axes from the calibrated TF —
                # restored from the legacy box_debug stream so the panel 3D
                # pane shows WHERE the world origin sits in the image.
                self._draw_aruco_axes(canvas)
            # Draw every track's last overlay state IDENTICALLY whether the
            # fit ran this frame or is cached from up to 1 s ago — constant
            # brightness/labels, no strobing at the fit-throttle cadence.
            now_s = self.get_clock().now().nanoseconds * 1e-9
            for tr2 in self._tracks.values():
                ov = tr2.get('rim_overlay')
                if ov is None or now_s - ov['t'] > 1.0:
                    continue
                if ov['sil'] is not None:
                    cv2.polylines(canvas,
                                  [np.round(ov['sil']).astype(np.int32)],
                                  True, (255, 255, 0), 2)
                if ov['dot'] is not None:
                    cv2.circle(canvas, ov['dot'], 4, (0, 0, 255), -1)
                cv2.putText(canvas, ov['label'], ov['pos'],
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (255, 255, 0) if ov['ok'] else (0, 0, 255), 1)
            cv2.putText(canvas,
                        f'rim obs {len(rim_obs)} | dets '
                        f'{len(per_object_masks)}',
                        (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2)
            dbg_msg = self.bridge.cv2_to_imgmsg(canvas, encoding='bgr8')
            dbg_msg.header = rgb_msg.header
            self.rim_debug_pub.publish(dbg_msg)

        # Per-frame box debug overlay using the LAST window's fit (frozen
        # box / frustum between updates is expected — they refresh every
        # window_period_s).
        debug_img = rgb.copy()
        n_drawn = 0
        for tid, track in self._tracks.items():
            ls = track.get('last_state')
            if ls is None:
                continue
            colour = _palette(tid - 1)
            self._draw_box_overlay(
                debug_img, ls['center'], ls['R'], ls['size'], ls['top_world'],
                colour, ls['label'], R_wc, t_wc)
            if ls.get('frustum') is not None:
                self._draw_frustum_overlay(
                    debug_img, ls['frustum'], colour, R_wc, t_wc)
            n_drawn += 1
        self._annotate_status(debug_img, n_drawn)
        self._draw_aruco_axes(debug_img)
        self._publish_debug(debug_img, rgb_msg.header)

        # Window check — finalize after window_period_s elapsed.
        now = self.get_clock().now()
        if self._window_start_stamp is None:
            self._window_start_stamp = now
            return
        elapsed = (now - self._window_start_stamp).nanoseconds * 1e-9
        if elapsed >= self.window_period_s:
            self._finalize_window(rgb_msg.header.stamp)
            self._window_start_stamp = now

    # ------------------------------------------------------------------
    def _draw_det_overlay(self, canvas, obj, mb) -> None:
        """YOLO segmentation overlay (mask tint + bbox + label) for EVERY
        detection — same look as detection_debug, so the rim stream shows
        the raw segmentation evidence alongside the fits."""
        sel = canvas[mb]
        canvas[mb] = (sel.astype(np.float32) * 0.65
                      + np.array([0.0, 0.0, 255.0]) * 0.35).astype(np.uint8)
        x1, y1 = int(obj.x_min), int(obj.y_min)
        cv2.rectangle(canvas, (x1, y1),
                      (int(obj.x_max), int(obj.y_max)), (0, 255, 0), 1)
        iid = int(getattr(obj, 'instance_id', -1))
        tag = f'#{iid}' if iid >= 0 else '#?'
        cv2.putText(canvas, f'{tag} {obj.class_name} {obj.score:.2f}',
                    (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    # ------------------------------------------------------------------
    def _rim_observe(self, obj, mb, oz, obj_world, R_wc, t_wc, track,
                     class_name, tid, rgb, rim_dbg, tf_stamped_ok=True,
                     moving_at=False):
        """Throttled silhouette chamfer fit for one upright-cup detection.

        Returns a CupObservation or None. The fit aligns the KNOWN truncated
        cone (2 DOF: axis x, y at an assumed base elevation) to the raw mask
        contour — depth contributes only the rough top-z (level evidence) and
        the (x, y) initial guess, so depth bias does not propagate into the
        measurement beyond the slide-along-ray term the consumer corrects.

        Every ATTEMPTED fit (success or not) is drawn onto rim_dbg['canvas']
        for the /digital_twin/rim_debug stream: observed contour green,
        fitted silhouette cyan, depth init red, text with id/iou/rms or the
        failure reason.
        """
        # A latest-TF fallback frame on a camera that is MOVING pairs the
        # image with the wrong pose — exactly the image↔trajectory desync
        # seen as a ~30 mm exo↔hand residual. Produce no measurement then.
        # When the joints are STILL the latest TF equals the stamped TF
        # physically, so the fallback pairing is safe — without this
        # exemption a paused replay (stamped lookup unlucky every frame)
        # starves the fusion of hand observations entirely.
        if not tf_stamped_ok and self.camera_name != 'exo' and moving_at:
            self.get_logger().info(
                'rim: skipping fallback-TF frame while moving',
                throttle_duration_sec=5.0)
            return None
        now_s = self.get_clock().now().nanoseconds * 1e-9
        if 'rim_last_t' not in track:
            # De-phase first fits across tracks: with a shared epoch all N
            # cups fit on the SAME frame every period (a 6 ms x N callback
            # spike); a tid-keyed offset spreads them evenly.
            track['rim_last_t'] = now_s - self._rim_fit_period * (
                (tid * 0.37) % 1.0)
        if now_s - track['rim_last_t'] < self._rim_fit_period:
            return None
        track['rim_last_t'] = now_s

        canvas = rim_dbg['canvas']

        hist = track.get('ztop_hist') or []
        z_top_rough = (float(np.median(hist)) if hist
                       else float(np.percentile(obj_world[:, 2], 97.0)))
        z_base0 = z_top_rough - self.cup_h
        xy0 = (float(np.median(obj_world[:, 0])),
               float(np.median(obj_world[:, 1])))
        mask_u8 = mb.astype(np.uint8) * 255
        try:
            fit = fit_silhouette_xy(
                mask_u8, K=self.K, dist=self.intr.dist, R_wc=R_wc, t_wc=t_wc,
                r_top=self.cup_top_d * 0.5, r_bot=self.cup_bot_d * 0.5,
                height=self.cup_h, z_base=z_base0, xy0=xy0,
                fit_boundary_offset=self._rim_boundary_offset)
        except Exception as e:  # never let the obs path break frame ingest
            self.get_logger().warn(
                f'silhouette fit failed: {e}', throttle_duration_sec=5.0)
            fit = {'ok': False, 'fail': 'exception'}
        ok = fit['ok'] and fit['iou'] >= self._rim_min_iou

        # Edge-snap: replace the mask-derived boundary with the image
        # gradient where a strong edge exists. Falls back silently to the
        # chamfer result (edge_coverage = 0) when edges are too weak.
        edge_cov = 0.0
        if ok and self._rim_edge_snap:
            gray = rim_dbg.get('gray')
            if gray is None:
                gray = rim_dbg['gray'] = cv2.cvtColor(
                    rgb, cv2.COLOR_BGR2GRAY)
            mag = rim_dbg.get('mag')
            if mag is None:
                gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
                mag = rim_dbg['mag'] = cv2.magnitude(gx, gy)
            try:
                snap = edge_snap_fit(
                    gray, K=self.K, dist=self.intr.dist, R_wc=R_wc,
                    t_wc=t_wc, r_top=self.cup_top_d * 0.5,
                    r_bot=self.cup_bot_d * 0.5, height=self.cup_h,
                    z_base=z_base0, xy0=(fit['x'], fit['y']),
                    min_grad=self._rim_edge_min_grad, grad_mag=mag)
            except Exception:
                snap = {'ok': False}
            if snap.get('ok'):
                fit = {**fit, 'x': snap['x'], 'y': snap['y'],
                       'rms_px': snap['rms_px']}
                edge_cov = float(snap['edge_cov'])

        # Store the overlay STATE only — drawing happens uniformly for all
        # tracks in _process_frame's publish block. Drawing fresh fits in a
        # bright style here and cached ones dim there made the display pulse
        # at the fit-throttle rate (the reported flicker).
        if canvas is not None:
            tx, ty = int(obj.x_min), max(12, int(obj.y_min) - 4)
            if ok:
                sil = cone_silhouette_px(
                    fit['x'], fit['y'], z_base0,
                    r_top=self.cup_top_d * 0.5, r_bot=self.cup_bot_d * 0.5,
                    height=self.cup_h, K=self.K, dist=self.intr.dist,
                    R_wc=R_wc, t_wc=t_wc)
                dot = None
                p0c = R_wc.T @ (np.array([xy0[0], xy0[1], z_base0]) - t_wc)
                if p0c[2] > 1e-3:
                    dot = (int(round(self.intr.fx * p0c[0] / p0c[2]
                                     + self.intr.cx)),
                           int(round(self.intr.fy * p0c[1] / p0c[2]
                                     + self.intr.cy)))
                track['rim_overlay'] = {
                    'ok': True, 'sil': sil, 'dot': dot,
                    'label': (f"#{tid} iou{fit['iou']:.2f} "
                              f"rms{fit['rms_px']:.1f} "
                              f"b{fit.get('b_px', 0.0):+.1f} "
                              f"cov{edge_cov:.1f} z{z_base0:+.2f}"),
                    'pos': (tx, ty + 14), 't': now_s}
            else:
                reason = fit.get('fail') or f"low_iou {fit.get('iou', 0):.2f}"
                track['rim_overlay'] = {
                    'ok': False, 'sil': None, 'dot': None,
                    'label': f'#{tid} FIT-DROP {reason}',
                    'pos': (tx, ty + 14), 't': now_s}

        if not ok:
            return None

        sigma = self._rim_sigma_px
        if moving_at:
            sigma *= 3.0
        if fit['truncated']:
            sigma *= 2.0
        origin, ray_d = ray_through_point(
            np.array([fit['x'], fit['y'], z_base0]), t_wc)

        ob = CupObservation()
        ob.camera = self.camera_name
        # The resolved tid, NOT the raw YOLO id: in standalone role a
        # re-ID is absorbed into an existing track whose color/level
        # state this observation carries — the consumer's per-(cam,id)
        # cache must follow the same identity.
        ob.instance_id = int(tid)
        ob.class_name = class_name
        ob.score = float(obj.score)
        ob.color = str(track.get('color') or '')
        ob.x0 = float(fit['x'])
        ob.y0 = float(fit['y'])
        ob.z_base0 = float(z_base0)
        ob.ray_origin = MsgPoint(
            x=float(origin[0]), y=float(origin[1]), z=float(origin[2]))
        ob.ray_dir = Vector3(
            x=float(ray_d[0]), y=float(ray_d[1]), z=float(ray_d[2]))
        ob.focal_px = float(np.sqrt(self.intr.fx * self.intr.fy))
        ob.sigma_px = float(sigma)
        ob.chamfer_rms_px = float(fit['rms_px'])
        ob.mask_iou = float(fit['iou'])
        ob.contour_points = int(fit['n_contour'])
        ob.boundary_offset_px = float(fit.get('b_px', 0.0))
        ob.edge_coverage = edge_cov
        ob.median_depth_m = float(np.median(oz)) if oz.size else 0.0
        ob.z_top_rough_m = z_top_rough
        ob.moving = bool(moving_at)
        return ob

    # ------------------------------------------------------------------
    def _finalize_window(self, stamp) -> None:
        """End-of-window: aggregate each track's accumulated points, MAD-filter,
        re-fit the cup pose (or OBB fallback), then publish the union of all
        tracks' filtered points and a fresh marker set. This is the only path
        that publishes /points and /boxes.

        Per-track Kalman flow (replaces the old scan-and-lock freeze):
          1. predict() the constant-position filter once per window — even on a
             miss, so covariance grows and a re-appearing cup is not rejected.
          2. On a valid window, feed the fresh cup/OBB centre to the gated
             filter (see `_fit_and_render_state` → `_kf_update_centre`). The
             estimate keeps creeping toward good measurements, so a wrong fit
             self-corrects; transient spikes are gated out; a sustained run of
             rejections re-acquires (robot picked / relocated the cup).
          3. A track is evicted after `cup_track_keepalive_frames` empty windows.
        """
        alive_xyz: list[np.ndarray] = []
        alive_rgb: list[np.ndarray] = []
        world_objs: list = []   # producer role only: per-object world clouds

        for tid in list(self._tracks.keys()):
            track = self._tracks[tid]
            buf_pts = track['points_buf']
            buf_cols = track['colors_buf']
            track['points_buf'] = []
            track['colors_buf'] = []

            # Constant-position predict once per window (grows covariance even
            # on a miss so a re-appearing / relocated cup is not over-rejected).
            if track.get('kf') is not None:
                track['kf'].predict()
                track['settled'] = (
                    track['kf'].position_std() <= self._kf_settled_std)

            seen = track.pop('seen', False)
            if not buf_pts:
                # A cup that WAS detected this window (rim-only mode:
                # upright_clouds=false buffers no points) is not a miss —
                # event-counting an actively-seen track evicted it every
                # keepalive cycle, wiping ztop_hist/color/rim state.
                if not seen:
                    track['miss'] += 1
                    if track['miss'] > self.cup_keepalive:
                        self._tracks.pop(tid, None)
                else:
                    track['miss'] = 0
                # last_state persists → frozen marker until the cup reappears.
                continue

            all_pts = np.vstack(buf_pts)
            all_rgb = np.concatenate(buf_cols)
            keep = _mad_keep_indices(all_pts, self.outlier_mad_k)
            if keep is not None:
                all_pts = all_pts[keep]
                all_rgb = all_rgb[keep]
            if all_pts.shape[0] < 32:
                track['miss'] += 1
                if track['miss'] > self.cup_keepalive:
                    self._tracks.pop(tid, None)
                continue

            track['miss'] = 0

            # Cap points before the heavy passes (least_squares cup fit, KDTree
            # density filter, PointCloud2 serialize). All scale with N, so an
            # unbounded mask spike is what saturates the single-threaded loop.
            if self.max_points_per_track and \
                    all_pts.shape[0] > self.max_points_per_track:
                # Deterministic stride (not random) so the producer emits a
                # STABLE point set frame-to-frame — random subsampling made the
                # fused fit jitter and the /points cloud dance.
                k = int(np.ceil(all_pts.shape[0] / self.max_points_per_track))
                all_pts = all_pts[::k]
                all_rgb = all_rgb[::k]

            # Producer role: emit this object's world cloud and skip the local
            # fit/KF/markers — cup_fusion_node owns those. Density filtering is
            # intentionally NOT applied here (left to fusion so the side-arc
            # geometry survives the cross-view merge).
            if self.role == 'producer':
                world_objs.append(WorldObjectCloud(
                    camera=self.camera_name,
                    instance_id=int(tid),
                    class_name=track['class_name'],
                    score=float(track['last_score']),
                    moving=self._joints_moving,
                    centroid=MsgPoint(
                        x=float(all_pts[:, 0].mean()),
                        y=float(all_pts[:, 1].mean()),
                        z=float(all_pts[:, 2].mean())),
                    points=_make_pointcloud2(
                        Header(stamp=stamp, frame_id=self.world_frame),
                        all_pts.astype(np.float32),
                        all_rgb.astype(np.float32))))
                continue

            # Fit + Kalman-filter the centre (update happens inside).
            ls = self._fit_and_render_state(tid, track, all_pts)
            if ls is not None:
                track['last_state'] = ls

            # Top-rim filter for a clean rendered cloud: keep only points within
            # `top_rim_band_m` of the cluster's `top_rim_percentile` height.
            # band ≤ 0 disables it (keep the whole cluster). Applied to the
            # published cloud only — the fit above already used all points.
            if self._top_rim_band > 0.0 and all_pts.shape[0] > 16:
                z_top = float(np.percentile(all_pts[:, 2], self._top_rim_pct))
                top_mask = all_pts[:, 2] > (z_top - self._top_rim_band)
                if top_mask.sum() > 16:
                    all_pts = all_pts[top_mask]
                    all_rgb = all_rgb[top_mask]

            # Optional: Spatial density filter (remove isolated noise)
            if self._use_spatial_density and all_pts.shape[0] > self._spatial_density_min_neighbors:
                density_keep = _filter_spatial_density(
                    all_pts,
                    radius=self._spatial_density_radius,
                    min_neighbors=self._spatial_density_min_neighbors)
                all_pts = all_pts[density_keep]
                all_rgb = all_rgb[density_keep]

            alive_xyz.append(all_pts.astype(np.float32))
            alive_rgb.append(all_rgb)

        # Producer role: publish the per-object world clouds and stop here —
        # no /points, /boxes, or cups_on_table from this node in fusion mode.
        if self.role == 'producer':
            self.world_clouds_pub.publish(WorldObjectCloudArray(
                header=Header(stamp=stamp, frame_id=self.world_frame),
                objects=world_objs))
            return

        # Combined cloud — every (filtered) accumulated point is plotted.
        if alive_xyz:
            cloud_xyz = np.vstack(alive_xyz)
            cloud_rgb = np.concatenate(alive_rgb).astype(np.float32)
        else:
            cloud_xyz = np.zeros((0, 3), dtype=np.float32)
            cloud_rgb = np.zeros((0,), dtype=np.float32)
        self.points_pub.publish(_make_pointcloud2(
            header=Header(stamp=stamp, frame_id=self.world_frame),
            xyz=cloud_xyz, rgb=cloud_rgb))

        # Marker emission — one update per window. DELETE for evicted tracks.
        markers = MarkerArray()
        alive_ids: set[int] = set()
        for tid, track in sorted(self._tracks.items()):
            ls = track.get('last_state')
            if ls is None:
                continue
            colour = _palette(tid - 1)
            # '[L]' marks a settled (low-covariance) estimate — the stable-pose
            # signal pick_ui_node parses as `locked`. Kept as '[L]' so that
            # downstream contract is unchanged; it now means KF-settled.
            label = ('[L]_' + ls['label']
                     if track.get('settled') else ls['label'])
            self._append_box_markers(
                markers, tid, ls['center'], ls['R'], ls['size'],
                ls['top_world'], colour, label, stamp)
            if ls.get('frustum') is not None:
                self._append_cup_frustum_markers(
                    markers, tid, ls['frustum'], colour, stamp)
            alive_ids.add(tid)
        stale = self._last_published_ids - alive_ids
        for tid in stale:
            self._append_delete_markers(markers, tid, stamp)
        self._last_published_ids = alive_ids
        self.boxes_pub.publish(markers)
        self._publish_cups_on_table()

    # ------------------------------------------------------------------
    def _publish_cups_on_table(self) -> None:
        """JSON {color: count} of currently rendered tracks (has last_state)
        whose id is NOT in /stack_track_ids.  Schema is always the configured
        color_classes plus 'unknown' so downstream sees a stable shape."""
        counts: dict[str, int] = {c: 0 for c in self._color_classes}
        counts['unknown'] = 0
        for tid, track in self._tracks.items():
            if track.get('last_state') is None:
                continue
            if tid in self._stacked_ids:
                continue
            colour = track.get('color') or 'unknown'
            counts[colour] = counts.get(colour, 0) + 1
        self.cups_on_table_pub.publish(
            String(data=json.dumps(counts, ensure_ascii=False)))

    def _on_stack_track_ids(self, msg: Int32MultiArray) -> None:
        self._stacked_ids = {int(x) for x in msg.data}

    # ------------------------------------------------------------------
    def _kf_update_centre(self, tid: int, track: dict,
                          z: np.ndarray, residual: float) -> np.ndarray:
        """Run the per-track Kalman measurement update for the 3D centre `z`
        and return the filtered centre. `predict()` is done once per window in
        `_finalize_window`; this only does the update / re-acquire bookkeeping.

        - First sight: create the filter anchored at `z`.
        - In-gate measurement: standard update; reset the relocation counter.
        - Gated-out measurement: hold the estimate (transient spike). After
          `kf_reacquire_windows` consecutive rejections, treat it as a real
          relocation and re-acquire at `z`.
        Measurement noise is inflated by the cup-fit residual so a poor fit
        cannot yank the estimate.
        """
        z = np.asarray(z, dtype=np.float64)
        if track.get('kf') is None:
            track['kf'] = PositionKF(z, self._kf_p0, self._kf_q)
            track['reacquire_count'] = 0
            self._sync_track_centre(track)
            return track['kf'].x.copy()

        kf = track['kf']
        infl = 1.0
        if self._kf_resid_infl > 0.0 and self.cup_resid_max > 0.0:
            infl = 1.0 + self._kf_resid_infl * (residual / self.cup_resid_max)
        accepted, _d2 = kf.update(z, self._kf_r * infl, self._kf_gate)
        if accepted:
            track['reacquire_count'] = 0
        else:
            track['reacquire_count'] = track.get('reacquire_count', 0) + 1
            if track['reacquire_count'] >= self._kf_reacquire_windows:
                kf.reset(z, self._kf_p0)
                track['reacquire_count'] = 0
                self.get_logger().info(
                    f'[kf] #{tid} re-acquired (relocation) at '
                    f'({z[0]:+.3f},{z[1]:+.3f},{z[2]:+.3f})m')
        self._sync_track_centre(track)
        track['settled'] = kf.position_std() <= self._kf_settled_std
        return kf.x.copy()

    def _sync_track_centre(self, track: dict) -> None:
        """Cache the filtered XY so `_resolve_track_id` re-association sees the
        smoothed centre rather than a raw per-window centroid."""
        track['center_xy'] = track['kf'].x[:2].astype(np.float64).copy()

    # ------------------------------------------------------------------
    def _fit_and_render_state(self, tid: int, track: dict,
                              all_pts: np.ndarray):
        """Fit the cup pose (with OBB fallback), Kalman-filter the centre, and
        return the geometry + label dict for marker / overlay emission, or
        None if neither cup nor OBB fit succeeds.

        The raw per-window fit is the KF *measurement*; the rendered pose comes
        from the filtered estimate, so the box tracks continuously and a
        slightly-wrong fit is pulled back toward truth instead of being frozen.

        Labels use underscore separators so each line is a single
        whitespace-free token — TEXT_VIEW_FACING markers spread spaces too
        wide and the label drifts horizontally across the screen otherwise.
        """
        class_name = track['class_name']
        cup_kind = (class_name in self.cup_class_names)

        # ── Measurement: cup-cone fit (preferred) ─────────────────────────
        if cup_kind:
            fit = _fit_cup_axis_xy(
                all_pts, top_d=self.cup_top_d, bot_d=self.cup_bot_d,
                height=self.cup_h)
            if fit is not None and fit[3] <= self.cup_resid_max:
                cx_new, cy_new, z_base_new, residual = fit
                track['last_residual'] = residual
                # KF state is the cup *centre*; z_base = centre_z - h/2.
                z_meas = np.array(
                    [cx_new, cy_new, z_base_new + 0.5 * self.cup_h],
                    dtype=np.float64)
                filt = self._kf_update_centre(tid, track, z_meas, residual)
                cx_s, cy_s = float(filt[0]), float(filt[1])
                z_base_s = float(filt[2]) - 0.5 * self.cup_h

                center = np.array([cx_s, cy_s, float(filt[2])], dtype=np.float64)
                R_box = np.eye(3)
                d_max = max(self.cup_top_d, self.cup_bot_d)
                size = np.array([d_max, d_max, self.cup_h], dtype=np.float64)
                top_world = np.array(
                    [cx_s, cy_s, z_base_s + self.cup_h], dtype=np.float64)
                frustum = _cup_frustum_geometry(
                    cx_s, cy_s, top_d=self.cup_top_d, bot_d=self.cup_bot_d,
                    height=self.cup_h, floor_z=z_base_s, n_seg=self.cup_n_seg)
                r_mm = residual * 1000.0
                color_tok = track.get('color') or 'unknown'
                line1 = (f"#{tid}_c={color_tok}_{track['last_display_name']}_"
                         f"{track['last_score']:.2f}")
                line2 = (f"r={r_mm:.0f}mm_"
                         f"({cx_s:.2f},{cy_s:.2f},{top_world[2]:.2f})")
                label = (line1.replace(' ', '_') + '\n'
                         + line2.replace(' ', '_'))
                return {
                    'center': center, 'R': R_box, 'size': size,
                    'top_world': top_world, 'frustum': frustum, 'label': label,
                }

        # ── Fallback: OBB / AABB (non-cup or cone fit failed) ─────────────
        box = _compute_box_world(
            all_pts,
            standing_ratio=self.standing_ratio,
            min_elongation=self.min_elongation,
            force_aabb=self.force_aabb)
        if box is None:
            return None
        center_meas, R_box, size, pose_label = box
        # KF-filter only the centre; orientation / size stay per-window fresh.
        filt = self._kf_update_centre(
            tid, track, center_meas.astype(np.float64), 0.0)
        center = filt.copy()
        top_world = center + R_box @ np.array(
            [0.0, 0.0, float(size[2]) * 0.5], dtype=np.float64)
        color_tok = track.get('color') or 'unknown'
        line1 = (f"#{tid}_c={color_tok}_{track['last_display_name']}_"
                 f"{track['last_score']:.2f}_[{pose_label}]")
        line2 = (f"({top_world[0]:.2f},{top_world[1]:.2f},"
                 f"{top_world[2]:.2f})")
        label = line1.replace(' ', '_') + '\n' + line2.replace(' ', '_')
        return {
            'center': center, 'R': R_box, 'size': size,
            'top_world': top_world, 'frustum': None, 'label': label,
        }

    # ------------------------------------------------------------------
    def _append_box_markers(self, markers: MarkerArray, idx: int,
                            center: np.ndarray, R_box: np.ndarray,
                            size: np.ndarray, top_world: np.ndarray,
                            colour: tuple[float, float, float],
                            label: str, stamp) -> None:
        qx, qy, qz, qw = _rot_to_quat(R_box)

        cube = Marker()
        cube.header.frame_id = self.world_frame
        cube.header.stamp = stamp
        cube.ns = 'boxes'
        cube.id = idx
        cube.type = Marker.CUBE
        cube.action = Marker.ADD
        cube.pose.position = MsgPoint(
            x=float(center[0]), y=float(center[1]), z=float(center[2]))
        cube.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
        cube.scale.x = float(max(size[0], 1e-3))
        cube.scale.y = float(max(size[1], 1e-3))
        cube.scale.z = float(max(size[2], 1e-3))
        cube.color = ColorRGBA(
            r=colour[0], g=colour[1], b=colour[2], a=float(self.box_alpha))
        markers.markers.append(cube)

        outline = Marker()
        outline.header = cube.header
        outline.ns = 'box_outline'
        outline.id = idx
        outline.type = Marker.LINE_LIST
        outline.action = Marker.ADD
        outline.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        outline.scale.x = self.box_line_w
        outline.color = ColorRGBA(r=colour[0], g=colour[1], b=colour[2], a=1.0)
        corners = _box_corners(center, R_box, size)
        for a, b in _BOX_EDGES:
            outline.points.append(MsgPoint(
                x=float(corners[a, 0]), y=float(corners[a, 1]), z=float(corners[a, 2])))
            outline.points.append(MsgPoint(
                x=float(corners[b, 0]), y=float(corners[b, 1]), z=float(corners[b, 2])))
        markers.markers.append(outline)

        # Sphere marker at the box top-centre (pivot (0,0,+1) in box-local).
        top = Marker()
        top.header = cube.header
        top.ns = 'box_top'
        top.id = idx
        top.type = Marker.SPHERE
        top.action = Marker.ADD
        top.pose.position = MsgPoint(
            x=float(top_world[0]), y=float(top_world[1]), z=float(top_world[2]))
        top.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        top.scale.x = top.scale.y = top.scale.z = 0.02
        top.color = ColorRGBA(r=1.0, g=0.95, b=0.0, a=1.0)
        markers.markers.append(top)

        text = Marker()
        text.header = cube.header
        text.ns = 'box_labels'
        text.id = idx
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        # Stagger label Z by track id so coplanar cups don't overlap. Two
        # rows alternate (0.04, 0.09 m above the top centre).
        z_offset = 0.04 + 0.05 * ((idx - 1) % 2)
        text.pose.position = MsgPoint(
            x=float(top_world[0]),
            y=float(top_world[1]),
            z=float(top_world[2] + z_offset))
        text.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        text.scale.z = 0.025
        text.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        text.text = label
        markers.markers.append(text)

    # ------------------------------------------------------------------
    def _draw_box_overlay(self, img: np.ndarray, center: np.ndarray,
                          R_box: np.ndarray, size: np.ndarray,
                          top_world: np.ndarray,
                          colour: tuple[float, float, float], label: str,
                          R_wc: np.ndarray, t_wc: np.ndarray) -> None:
        corners_world = _box_corners(center, R_box, size)
        # world -> camera: p_c = R_wc^T @ (p_w - t_wc)
        corners_cam = (R_wc.T @ (corners_world - t_wc).T).T
        in_front = corners_cam[:, 2] > 0.05
        if not np.any(in_front):
            return
        z_safe = np.clip(corners_cam[:, 2], 1e-6, None)
        pix = (self.K @ corners_cam.T).T
        pix = pix[:, :2] / z_safe[:, None]
        pts = pix.astype(int)
        bgr_colour = (int(colour[2] * 255), int(colour[1] * 255), int(colour[0] * 255))
        for a, b in _BOX_EDGES:
            if not (in_front[a] and in_front[b]):
                continue
            cv2.line(img, tuple(pts[a]), tuple(pts[b]), bgr_colour, 2)

        # Project the top-centre point and draw a marker on the image.
        top_cam = R_wc.T @ (top_world - t_wc)
        top_label_anchor = None
        if top_cam[2] > 0.05:
            tp = (self.K @ top_cam) / max(float(top_cam[2]), 1e-6)
            tx, ty = int(round(tp[0])), int(round(tp[1]))
            cv2.circle(img, (tx, ty), 6, (0, 240, 255), -1)
            cv2.circle(img, (tx, ty), 7, (0, 0, 0), 1)
            top_label_anchor = (tx, ty)

        if top_label_anchor is not None:
            tx, ty = top_label_anchor
            cv2.putText(img, label, (tx + 8, max(0, ty - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 240, 255), 1,
                        lineType=cv2.LINE_AA)
        elif np.any(in_front):
            anchor = pts[in_front][np.argmin(pix[in_front, 1])]
            ax, ay = int(anchor[0]), max(0, int(anchor[1]) - 6)
            cv2.putText(img, label, (ax, ay),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, bgr_colour, 2)

    # ------------------------------------------------------------------
    def _append_cup_frustum_markers(self, markers: MarkerArray, idx: int,
                                    frustum: dict,
                                    colour: tuple[float, float, float],
                                    stamp) -> None:
        col = ColorRGBA(r=colour[0], g=colour[1], b=colour[2], a=1.0)

        def _loop_marker(ns: str, loop: np.ndarray) -> Marker:
            m = Marker()
            m.header.frame_id = self.world_frame
            m.header.stamp = stamp
            m.ns = ns
            m.id = idx
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            m.scale.x = self.box_line_w
            m.color = col
            for p in loop:
                m.points.append(MsgPoint(
                    x=float(p[0]), y=float(p[1]), z=float(p[2])))
            return m

        markers.markers.append(_loop_marker('cup_top_loop', frustum['top_loop']))
        markers.markers.append(_loop_marker('cup_bot_loop', frustum['bot_loop']))

        gen = Marker()
        gen.header.frame_id = self.world_frame
        gen.header.stamp = stamp
        gen.ns = 'cup_generatrix'
        gen.id = idx
        gen.type = Marker.LINE_LIST
        gen.action = Marker.ADD
        gen.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        gen.scale.x = self.box_line_w
        gen.color = col
        for p_top, p_bot in frustum['generatrix']:
            gen.points.append(MsgPoint(
                x=float(p_top[0]), y=float(p_top[1]), z=float(p_top[2])))
            gen.points.append(MsgPoint(
                x=float(p_bot[0]), y=float(p_bot[1]), z=float(p_bot[2])))
        markers.markers.append(gen)

    def _draw_frustum_overlay(self, img: np.ndarray, frustum: dict,
                              colour: tuple[float, float, float],
                              R_wc: np.ndarray, t_wc: np.ndarray) -> None:
        bgr_colour = (int(colour[2] * 255), int(colour[1] * 255), int(colour[0] * 255))

        def _project(loop_world: np.ndarray):
            cam = (R_wc.T @ (loop_world - t_wc).T).T
            ok = cam[:, 2] > 0.05
            zs = np.clip(cam[:, 2], 1e-6, None)
            pix = (self.K @ cam.T).T
            pix = pix[:, :2] / zs[:, None]
            return pix.astype(int), ok

        for loop in (frustum['top_loop'], frustum['bot_loop']):
            pts, ok = _project(loop)
            for k in range(len(pts) - 1):
                if ok[k] and ok[k + 1]:
                    cv2.line(img, tuple(pts[k]), tuple(pts[k + 1]),
                             bgr_colour, 1, lineType=cv2.LINE_AA)
        for p_top, p_bot in frustum['generatrix']:
            pair = np.stack([p_top, p_bot], axis=0)
            pts, ok = _project(pair)
            if ok[0] and ok[1]:
                cv2.line(img, tuple(pts[0]), tuple(pts[1]),
                         bgr_colour, 1, lineType=cv2.LINE_AA)

    def _append_delete_markers(self, markers: MarkerArray, idx: int,
                               stamp) -> None:
        for ns in ('boxes', 'box_outline', 'box_top', 'box_labels',
                   'cup_top_loop', 'cup_bot_loop', 'cup_generatrix'):
            d = Marker()
            d.header.frame_id = self.world_frame
            d.header.stamp = stamp
            d.ns = ns
            d.id = idx
            d.action = Marker.DELETE
            markers.markers.append(d)

    # ------------------------------------------------------------------
    def _on_trigger_scan(self, request, response):
        """Clear all tracks (drops every Kalman filter so each cup re-acquires
        from scratch). Called via ~/trigger_scan."""
        n = len(self._tracks)
        self._tracks.clear()
        self._last_published_ids.clear()
        self._window_start_stamp = None
        if self.role != 'producer':
            # producer shares /digital_twin/boxes' default name with the
            # fusion node — a latched DELETEALL from here would wipe the
            # fused display that this node does not own.
            self._publish_clear_markers(self.get_clock().now().to_msg())
        response.success = True
        response.message = f'scan reset: {n} track(s) cleared'
        self.get_logger().info(f'[kf] trigger_scan → cleared {n} tracks')
        return response

    def _resolve_track_id(self, inst_id: int, centroid_xy: np.ndarray) -> int:
        """Map a ByteTrack instance_id to a world-frame track id.

        If inst_id is already a known track, return it directly.
        If inst_id is new but its world centroid is within `_track_merge_dist`
        of an existing track, return that track's id instead. This absorbs
        re-IDs that ByteTrack issues when the camera moves fast, preventing
        the old track (still within keepalive) and the new id from coexisting
        as duplicate ghost boxes.
        """
        if inst_id in self._tracks:
            return inst_id
        if self.role == 'producer':
            # In fusion mode the downstream cup_fusion_node does the geometric
            # association ACROSS cameras. The producer must NOT pre-merge here:
            # at far range (exo ~4.2 m) the noisy world centroids of distinct
            # YOLO masks collapse within _track_merge_dist, so several cups would
            # fold into ONE big contaminated cloud → one huge OBB. Keep one
            # cloud (one OBB) per YOLO instance id.
            return inst_id
        best_id = inst_id
        best_dist = self._track_merge_dist
        for tid, track in self._tracks.items():
            dist = float(np.linalg.norm(centroid_xy - track['center_xy']))
            if dist < best_dist:
                best_dist = dist
                best_id = tid
        return best_id

    def _erode_mask(self, mb: np.ndarray) -> np.ndarray:
        """Shrink the YOLO mask by `mask_erode_px` to drop edge pixels whose
        depth is unreliable (mixed foreground/background). No-op if disabled."""
        if self.mask_erode_px <= 0:
            return mb
        k = self.mask_erode_px
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        eroded = cv2.erode(mb.astype(np.uint8), kernel, iterations=1)
        return eroded > 0

    def _draw_aruco_axes(self, img: np.ndarray) -> None:
        """Project calibrated world (base) and ArUco marker frames onto the debug image.

        Both transforms are published once by world_origin_node at calibration
        time and looked up from the TF tree — no real-time marker detection needed.
        """
        if not self._aruco_overlay:
            return
        dist = getattr(self.intr, 'dist', None)

        def _project_label(tvec_3: np.ndarray, text: str, colour) -> None:
            z = float(tvec_3[2])
            if z < 0.01:
                return
            px = self.intr.K @ tvec_3 / z
            cv2.putText(img, text, (int(px[0]) + 8, int(px[1]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2, cv2.LINE_AA)

        # ── world frame = robot base origin ──────────────────────────────
        try:
            tf_w = self.tf_buffer.lookup_transform(
                self.camera_frame, self.world_frame, rclpy.time.Time())
            tw = tf_w.transform.translation
            qw = tf_w.transform.rotation
            tvec_w = np.array([[tw.x], [tw.y], [tw.z]], dtype=np.float64)
            if float(tw.z) > 0.01:
                R_cw = _quat_to_rot(qw.x, qw.y, qw.z, qw.w)
                rvec_w, _ = cv2.Rodrigues(R_cw)
                cv2.drawFrameAxes(img, self.intr.K, dist, rvec_w, tvec_w,
                                  self._aruco_axis_len * 1.5, thickness=3)
                _project_label(np.array([tw.x, tw.y, tw.z]),
                               'base', (255, 255, 255))
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            pass

        # ── aruco marker frame ────────────────────────────────────────────
        try:
            tf_a = self.tf_buffer.lookup_transform(
                self.camera_frame, 'aruco', rclpy.time.Time())
            ta = tf_a.transform.translation
            qa = tf_a.transform.rotation
            tvec_a = np.array([[ta.x], [ta.y], [ta.z]], dtype=np.float64)
            if float(ta.z) > 0.01:
                R_ca = _quat_to_rot(qa.x, qa.y, qa.z, qa.w)
                rvec_a, _ = cv2.Rodrigues(R_ca)
                cv2.drawFrameAxes(img, self.intr.K, dist, rvec_a, tvec_a,
                                  self._aruco_axis_len, thickness=2)
                _project_label(np.array([ta.x, ta.y, ta.z]),
                               'aruco', (0, 255, 255))
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            pass

    def _publish_clear_markers(self, stamp) -> None:
        clear = MarkerArray()
        d = Marker()
        d.header.frame_id = self.world_frame
        d.header.stamp = stamp
        d.action = Marker.DELETEALL
        clear.markers.append(d)
        self.boxes_pub.publish(clear)

    def _publish_debug(self, img: np.ndarray, src_header) -> None:
        msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        msg.header = src_header
        self.box_debug_pub.publish(msg)

    def _publish_depth_debug(self, depth_msg: Image, z_m: np.ndarray,
                             valid: np.ndarray, union_mask: np.ndarray) -> None:
        """JET-colormapped depth view for live debugging. Pixels outside
        [z_min, z_max] (or zero depth from the sensor) are blacked out;
        detection mask outlines are drawn in white so it's obvious whether
        depth is dropping out *inside* the object silhouette."""
        norm = np.zeros_like(z_m, dtype=np.uint8)
        if bool(valid.any()):
            zspan = max(self.z_max - self.z_min, 1e-6)
            scaled = np.clip((z_m - self.z_min) / zspan, 0.0, 1.0)
            norm[valid] = (scaled[valid] * 255.0).astype(np.uint8)
        color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        # Black-out invalids — applyColorMap on 0 gives dark blue otherwise,
        # which is hard to distinguish from "near the camera".
        color[~valid] = (0, 0, 0)
        if union_mask is not None and bool(union_mask.any()):
            edges = cv2.Canny((union_mask.astype(np.uint8) * 255), 50, 150)
            color[edges > 0] = (255, 255, 255)
        # Floor-patch overlay: rectangle + centre crosshair + median depth so
        # we can immediately see whether world_origin_node is sampling depth
        # over actual floor (and not, say, a hand or a chair leg).
        h, w = z_m.shape[:2]
        cx = w // 2 if self.patch_cx_px < 0 else self.patch_cx_px
        cy = h // 2 if self.patch_cy_px < 0 else self.patch_cy_px
        cx = int(np.clip(cx, 0, w - 1))
        cy = int(np.clip(cy, 0, h - 1))
        r = self.patch_radius
        x0, y0 = max(0, cx - r), max(0, cy - r)
        x1, y1 = min(w - 1, cx + r), min(h - 1, cy + r)
        cv2.rectangle(color, (x0, y0), (x1, y1), (0, 255, 255), 1)
        cv2.drawMarker(color, (cx, cy), (0, 255, 255),
                       markerType=cv2.MARKER_CROSS, markerSize=12, thickness=1)
        patch_z = z_m[y0:y1 + 1, x0:x1 + 1]
        patch_valid = valid[y0:y1 + 1, x0:x1 + 1]
        if bool(patch_valid.any()):
            med = float(np.median(patch_z[patch_valid]))
            patch_label = (f'patch {2 * r + 1}x{2 * r + 1}px @({cx},{cy}) '
                           f'med={med:.3f}m  valid={int(patch_valid.sum())}')
        else:
            patch_label = (f'patch {2 * r + 1}x{2 * r + 1}px @({cx},{cy}) '
                           f'NO VALID DEPTH')
        cv2.putText(color, patch_label, (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1,
                    cv2.LINE_AA)

        n_valid = int(valid.sum())
        n_total = int(valid.size)
        cv2.putText(
            color,
            f'depth valid {n_valid}/{n_total} ({100.0 * n_valid / max(n_total, 1):.0f}%) '
            f'  range [{self.z_min:.2f},{self.z_max:.2f}] m',
            (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            cv2.LINE_AA)
        msg = self.bridge.cv2_to_imgmsg(color, encoding='bgr8')
        msg.header = depth_msg.header
        self.depth_debug_pub.publish(msg)

    @staticmethod
    def _annotate_status(img: np.ndarray, n: int) -> None:
        colour = (0, 255, 0) if n else (0, 200, 255)
        cv2.putText(img, f'3d boxes={n}', (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)


# ----------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------

def _filter_spatial_density(xyz: np.ndarray, radius: float, min_neighbors: int) -> np.ndarray:
    """Remove isolated points via local density.
    
    For each point, count neighbors within `radius`. Remove if count < min_neighbors.
    (Catches stray noise islands on depth discontinuities.)
    
    Args:
        xyz: (N, 3) point cloud
        radius: neighborhood radius (meters)
        min_neighbors: minimum neighbor count to keep a point
    
    Returns:
        keep: (N,) boolean mask.
    """
    if xyz.shape[0] < min_neighbors:
        return np.ones(xyz.shape[0], dtype=bool)
    
    # KDTree for fast neighbor search. return_length counts neighbors in C
    # (avoids building an O(N*K) Python list-of-lists + per-point loop that
    # spikes memory and time on dense clusters); workers=-1 uses all cores.
    tree = cKDTree(xyz)
    counts = tree.query_ball_point(
        xyz, radius, workers=-1, return_length=True)
    return counts >= min_neighbors


# def _fit_cup_axis_xy(points: np.ndarray, *, top_d: float, bot_d: float,
#                      height: float):
#     """Algebraic LS fit of the cup axis (cx, cy) and base elevation z_base
#     given a vertical truncated-cone prior.

#     The cup may stand on any horizontal surface — table, shelf, floor — so
#     we don't assume a global floor height. The cluster's robust 5th-percentile
#     Z is treated as the cup base; r(z) interpolates between r_bot at z_base
#     and r_top at z_base+height. Every surface point satisfies
#         (x - cx)^2 + (y - cy)^2 = r(z)^2
#     so expanding gives a linear system in (cx, cy, C=cx^2+cy^2):
#         -2*cx*x - 2*cy*y + C = r(z)^2 - x^2 - y^2

#     Returns (cx, cy, z_base, rmse_residual_m) or None if degenerate. The
#     visible side alone is enough — radius variation along z constrains the
#     centre even from a one-sided arc.
#     """
def _fit_cup_axis_xy(points: np.ndarray, *, top_d: float, bot_d: float,
                     height: float):
    """Robust Non-linear LS fit of the cup axis (cx, cy) and base elevation z_base
    using the entire visible surface of the truncated cone.

    Unlike naive algebraic fits or top-rim extraction, this uses scipy's least_squares
    with a robust loss function (soft_l1) to completely ignore outliers caused by YOLO
    mask bleeding into stacked cups. It naturally handles occluded top rims.
    """
    if points.shape[0] < 16 or height <= 1e-6:
        return None
        
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # 1. 아랫면 Z (5th Percentile) 추정 - 컵이 쌓이거나 마스크가 번져도 바닥은 가장 정확함
    z_base = float(np.percentile(z, 5.0))
    
    # 2. 각 점들의 높이(z)에 맞는 '정상적인 반지름' 보간
    r_bot = bot_d * 0.5
    r_top = top_d * 0.5
    z_rel = np.clip(z - z_base, 0.0, height)
    r_z = r_bot + (r_top - r_bot) * (z_rel / height)
    
    # 3. 최적화 (Scipy least_squares with soft_l1 loss for outlier rejection)
    from scipy.optimize import least_squares
    
    def residuals(center):
        cx, cy = center
        rho = np.sqrt((x - cx)**2 + (y - cy)**2)
        return rho - r_z
        
    # 초기값: 2D 중심의 평균
    cx0, cy0 = float(np.mean(x)), float(np.mean(y))
    
    # soft_l1 loss: YOLO 마스크가 윗단 컵을 침범한 아웃라이어 데이터를 무시하게 만듦
    res = least_squares(residuals, x0=[cx0, cy0], loss='soft_l1')
    
    if not res.success:
        return None
        
    cx, cy = float(res.x[0]), float(res.x[1])
    if not (np.isfinite(cx) and np.isfinite(cy)):
        return None

    rho = np.sqrt((x - cx)**2 + (y - cy)**2)
    # Recover z_base from the KNOWN truncated-cone profile instead of assuming
    # the lowest visible point IS the base. Each point's radius tells WHERE on
    # the cone it sits (narrow top r_top ↔ wide bottom r_bot), so the base is
    # placed correctly even from a TOP-ONLY (hand looking down) view — which the
    # 5th-percentile assumption mis-placed to the rim. (r_top != r_bot required.)
    dr = r_top - r_bot
    if abs(dr) > 1e-4:
        z_rel_implied = np.clip((rho - r_bot) / dr * height, 0.0, height)
        z_base = float(np.median(z - z_rel_implied))

    # Model error (RMSE) at the recovered base.
    z_rel = np.clip(z - z_base, 0.0, height)
    r_z = r_bot + (r_top - r_bot) * (z_rel / height)
    rmse = float(np.sqrt(np.mean((rho - r_z)**2)))

    return cx, cy, z_base, rmse

def _cup_frustum_geometry(cx: float, cy: float, *, top_d: float, bot_d: float,
                          height: float, floor_z: float, n_seg: int) -> dict:
    """Pre-compute the world-frame vertices of the cup frustum wireframe used
    by the markers: closed top/bottom loops + a few vertical generatrix lines.
    """
    angles = np.linspace(0.0, 2.0 * np.pi, n_seg + 1)
    r_top = top_d * 0.5
    r_bot = bot_d * 0.5
    z_top = floor_z + height
    z_bot = floor_z
    top_loop = np.stack([
        cx + r_top * np.cos(angles),
        cy + r_top * np.sin(angles),
        np.full_like(angles, z_top),
    ], axis=1)
    bot_loop = np.stack([
        cx + r_bot * np.cos(angles),
        cy + r_bot * np.sin(angles),
        np.full_like(angles, z_bot),
    ], axis=1)
    n_gen = min(8, n_seg)
    gen_idx = np.linspace(0, n_seg, n_gen, endpoint=False).astype(int)
    pairs = [(top_loop[i].copy(), bot_loop[i].copy()) for i in gen_idx]
    return {'top_loop': top_loop, 'bot_loop': bot_loop, 'generatrix': pairs}


def _mad_keep_indices(points: np.ndarray, mad_k: float):
    """Boolean mask of points within `mad_k * 1.4826 * MAD` per axis. Returns
    None when the threshold is disabled (mad_k <= 0), the cluster is too
    small to compute a robust median, or filtering would discard so much
    data that downstream fitting becomes unstable."""
    if mad_k <= 0.0 or points.shape[0] < 16:
        return None
    med = np.median(points, axis=0)
    abs_dev = np.abs(points - med)
    mad = np.median(abs_dev, axis=0)
    threshold = mad_k * 1.4826 * np.maximum(mad, 1e-6)
    keep = np.all(abs_dev <= threshold, axis=1)
    if int(keep.sum()) < 16:
        return None
    return keep


def _filter_outliers(points: np.ndarray, mad_k: float) -> np.ndarray:
    """Drop points whose per-axis absolute deviation from the median exceeds
    `mad_k * 1.4826 * MAD` (≈ k·σ under Gaussian noise). Returns the filtered
    subset; falls back to the original cluster if filtering would leave too
    few points to fit a box."""
    if mad_k <= 0.0 or points.shape[0] < 16:
        return points
    med = np.median(points, axis=0)
    abs_dev = np.abs(points - med)
    mad = np.median(abs_dev, axis=0)
    # Avoid divide-by-zero on a perfectly flat axis.
    threshold = mad_k * 1.4826 * np.maximum(mad, 1e-6)
    keep = np.all(abs_dev <= threshold, axis=1)
    if int(keep.sum()) < 16:
        return points
    return points[keep]


def _compute_box_world(points: np.ndarray, *,
                       standing_ratio: float, min_elongation: float,
                       force_aabb: bool):
    """Estimate a 3D position box for a cluster of world-frame points.

    Returns (center(3,), R(3,3), size(3,), pose_label) or None.

    For cup-like targets: treat tall clusters (z extent dominates) as
    standing → axis-aligned box (yaw=0). Otherwise project to the XY plane
    and inspect PCA elongation; only commit to a yaw rotation when the
    principal axis is clearly longer than the secondary one.
    """
    if points.shape[0] < 32:
        return None

    pmin = points.min(axis=0)
    pmax = points.max(axis=0)
    extent = pmax - pmin
    z_ext = float(extent[2])
    h_ext = float(max(extent[0], extent[1]))
    aabb_center = (pmin + pmax) * 0.5

    if force_aabb or h_ext < 1e-6:
        return aabb_center, np.eye(3), extent, 'standing' if z_ext >= h_ext else 'unknown'

    if z_ext / max(h_ext, 1e-6) > standing_ratio:
        return aabb_center, np.eye(3), extent, 'standing'

    xy = points[:, :2]
    xy_centered = xy - xy.mean(axis=0)
    cov = np.cov(xy_centered.T)
    if not np.all(np.isfinite(cov)):
        return aabb_center, np.eye(3), extent, 'unknown'
    eigvals, eigvecs = np.linalg.eigh(cov)
    lam_major = float(eigvals[1])
    lam_minor = float(max(eigvals[0], 1e-12))
    elongation = (lam_major / lam_minor) ** 0.5

    if elongation < min_elongation:
        return aabb_center, np.eye(3), extent, 'unknown'

    principal = eigvecs[:, -1]
    yaw = float(np.arctan2(principal[1], principal[0]))
    cy, sy = np.cos(yaw), np.sin(yaw)
    R = np.array([[cy, -sy, 0.0],
                  [sy,  cy, 0.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)

    centroid = points.mean(axis=0)
    local = (R.T @ (points - centroid).T).T
    lmin = local.min(axis=0)
    lmax = local.max(axis=0)
    size = lmax - lmin
    center_world = centroid + R @ ((lmin + lmax) * 0.5)
    return center_world, R, size, 'fallen'


def _box_corners(center: np.ndarray, R: np.ndarray, size: np.ndarray) -> np.ndarray:
    h = np.asarray(size, dtype=np.float64) * 0.5
    s = np.array([
        [-1, -1, -1], [+1, -1, -1], [+1, +1, -1], [-1, +1, -1],
        [-1, -1, +1], [+1, -1, +1], [+1, +1, +1], [-1, +1, +1],
    ], dtype=np.float64) * h
    return (R @ s.T).T + np.asarray(center, dtype=np.float64)


def _quat_to_rot(x: float, y: float, z: float, w: float) -> np.ndarray:
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xx = x * x * s; yy = y * y * s; zz = z * z * s
    xy = x * y * s; xz = x * z * s; yz = y * z * s
    wx = w * x * s; wy = w * y * s; wz = w * z * s
    return np.array([
        [1 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1 - (xx + yy)],
    ], dtype=np.float64)


def _rot_to_quat(R: np.ndarray) -> tuple[float, float, float, float]:
    """Return quaternion (x, y, z, w) from a 3x3 rotation matrix."""
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    tr = m00 + m11 + m22
    if tr > 0:
        s = (tr + 1.0) ** 0.5 * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif (m00 > m11) and (m00 > m22):
        s = ((1.0 + m00 - m11 - m22) ** 0.5) * 2.0
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = ((1.0 + m11 - m00 - m22) ** 0.5) * 2.0
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = ((1.0 + m22 - m00 - m11) ** 0.5) * 2.0
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    return float(x), float(y), float(z), float(w)


def _make_pointcloud2(header: Header, xyz: np.ndarray, rgb: np.ndarray) -> PointCloud2:
    n = xyz.shape[0]
    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    point_step = 16
    buf = np.empty(n, dtype=np.dtype([
        ('x', np.float32), ('y', np.float32), ('z', np.float32), ('rgb', np.float32)
    ]))
    buf['x'] = xyz[:, 0]
    buf['y'] = xyz[:, 1]
    buf['z'] = xyz[:, 2]
    buf['rgb'] = rgb
    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = n
    msg.fields = fields
    msg.is_bigendian = False
    msg.point_step = point_step
    msg.row_step = point_step * n
    msg.is_dense = True
    msg.data = buf.tobytes()
    return msg


def main(args: Iterable[str] | None = None) -> None:
    rclpy.init(args=args)
    node = PointCloudNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
