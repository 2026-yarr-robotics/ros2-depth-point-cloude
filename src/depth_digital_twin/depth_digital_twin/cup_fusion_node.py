"""cup_fusion_node — fuse per-camera world-frame object clouds into one
per-cup estimate.

The per-camera `point_cloud_node`s run in `producer` role and publish
`WorldObjectCloudArray` (per-object world points). This node owns what used to
live in point_cloud_node's `_finalize_window`: it ASSOCIATES detections across
cameras by world geometry (NOT by ByteTrack id — those are per-camera), fits a
cup (or OBB for a fallen cup), Kalman-filters the centre per *physical* cup, and
emits the box MarkerArray.

Cadence: a fixed timer drives the KF `predict` (so process-noise Q is consistent
regardless of how the per-camera windows arrive); a cup's KF `update` only fires
when a fresh measurement is available, otherwise the estimate coasts. This is
the answer to the "accumulate-then-fit breaks the KF" concern.

Single vs dual view: identical path. With one camera only that camera's array is
non-empty, so clustering still de-duplicates intra-camera duplicate ids; the
cross-view merge simply has one member. Phase A merges members unweighted;
Phase C adds the hand/exo adaptive view weighting + voxel normalisation.
"""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
                       QoSReliabilityPolicy)
from geometry_msgs.msg import Point as MsgPoint, Quaternion, Vector3
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import JointState, PointCloud2
from std_msgs.msg import ColorRGBA, Header
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray
from scipy.spatial import cKDTree

from depth_digital_twin_msgs.msg import WorldObjectCloudArray

# Single source of truth for the estimation maths — reuse the pure helpers from
# point_cloud_node rather than duplicating them. (A later refactor may move
# these into a dedicated cup_estimation module; importing the node module is
# side-effect-free because main() is guarded by __main__.)
from depth_digital_twin.point_cloud_node import (
    PositionKF, _fit_cup_axis_xy, _compute_box_world, _make_pointcloud2,
    _cup_frustum_geometry, _palette, _rot_to_quat)


def _pc2_xyzrgb(cloud: PointCloud2):
    """Unpack (xyz, packed-rgb) from a producer-emitted PointCloud2
    (x,y,z,rgb FLOAT32, point_step 16). Matches _make_pointcloud2's layout."""
    n = cloud.width * cloud.height
    if n == 0:
        return np.zeros((0, 3)), np.zeros((0,), dtype=np.float32)
    arr = np.frombuffer(bytes(cloud.data), dtype=np.float32).reshape(n, -1)
    xyz = arr[:, :3].astype(np.float64)
    rgb = arr[:, 3].copy() if arr.shape[1] >= 4 else np.zeros((n,), np.float32)
    return xyz, rgb


def _voxel_idx(pts: np.ndarray, voxel: float) -> np.ndarray:
    """Indices of one representative point per voxel cell (deterministic)."""
    if pts.shape[0] == 0 or voxel <= 0:
        return np.arange(pts.shape[0])
    keys = np.floor(pts / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return idx


def _voxel_downsample(pts: np.ndarray, voxel: float) -> np.ndarray:
    """Keep one point per `voxel`-sized cell — equalises density so a near,
    high-density hand view doesn't swamp a far, sparse exo view in the fit.
    Deterministic: same input → same output (no per-frame jitter)."""
    return pts[_voxel_idx(pts, voxel)]


def _aabb_robust(pts: np.ndarray, clip: float = 1.0):
    """Outlier-robust per-axis AABB via [clip, 100-clip] percentiles — a single
    stray point must not blow up the box and inflate IoU."""
    if pts.shape[0] == 0:
        return np.zeros(3), np.zeros(3)
    lo = np.percentile(pts, clip, axis=0)
    hi = np.percentile(pts, 100.0 - clip, axis=0)
    return lo, hi


def _aabb_iou(a, b) -> float:
    (alo, ahi), (blo, bhi) = a, b
    lo = np.maximum(alo, blo)
    hi = np.minimum(ahi, bhi)
    inter = float(np.prod(np.clip(hi - lo, 0.0, None)))
    va = float(np.prod(np.clip(ahi - alo, 1e-9, None)))
    vb = float(np.prod(np.clip(bhi - blo, 1e-9, None)))
    u = va + vb - inter
    return inter / u if u > 0.0 else 0.0


def _union_find_groups(n: int, edges) -> list[list[int]]:
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in edges:
        parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _cluster_indices(centroids: list[np.ndarray], dist: float) -> list[list[int]]:
    """Union-find clustering of centroids within `dist` (xy+z). One cluster =
    one physical cup (absorbs intra-camera duplicate ids AND cross-camera
    detections of the same cup)."""
    n = len(centroids)
    if n == 0:
        return []
    if n == 1:
        return [[0]]
    pts = np.asarray(centroids, dtype=np.float64)
    tree = cKDTree(pts)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in tree.query_pairs(dist):
        parent[find(a)] = find(b)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


class CupFusionNode(Node):
    def __init__(self) -> None:
        super().__init__('cup_fusion_node')
        gp = self.declare_parameter

        gp('exo_clouds_topic', '/digital_twin/cups_exo')
        gp('hand_clouds_topic', '/digital_twin/cups_hand')
        gp('boxes_topic', '/digital_twin/boxes')
        gp('points_topic', '/digital_twin/points')
        gp('world_frame', 'world')
        # Debug-visualisation toggles (driven live by the Tk panel checkboxes).
        # Default: only the final stage-2 fit is shown.
        gp('dbg_detections', False)   # per-camera raw detection boxes
        gp('dbg_premerge', False)     # stage-1 merged-group boxes (pre-fit)
        gp('dbg_fit1', False)         # stage-1 (first) fit box + cup shape
        gp('dbg_fit2', True)          # stage-2 final fit box + cup shape (=/boxes)
        gp('dbg_detections_topic', '/digital_twin/dbg_detections')
        gp('dbg_premerge_topic', '/digital_twin/dbg_premerge')
        gp('dbg_fit1_topic', '/digital_twin/dbg_fit1')

        gp('fusion_period_s', 0.1)     # KF predict cadence (Q is per tick)
        # Keep a camera's LATEST cloud valid between its (possibly slow) updates
        # so the other view's points don't vanish/return every event. Must be
        # >= the slowest camera's publish period.
        gp('max_age_s', 1.5)
        gp('merge_dist_m', 0.035)      # STAGE-0 coarse centroid cluster (strict)
        gp('fusion_voxel_m', 0.004)    # voxel size to equalise per-view density
        gp('max_merge_points', 4000)   # cap merged points per cup (deterministic)
        # Reject a box bigger than this footprint — it's an over-merge of
        # several cups (or noise), not one cup. Stops giant boxes.
        gp('max_cup_footprint_m', 0.11)

        # STAGE-1 pre-fit point-cloud merge (AABB-IoU, STRICT AND of 3 gates).
        gp('premerge_iou', 0.62)
        gp('premerge_dxy_m', 0.018)
        gp('premerge_dz_m', 0.022)     # blocks telescoped/vertically-stacked cups
        gp('premerge_radius_m', 0.05)  # candidate-pair prefilter (centroid KDTree)
        # STAGE-2b post-fit re-merge (two fits pointing at the same cup).
        gp('postmerge_dxy_m', 0.015)
        gp('postmerge_dz_m', 0.015)
        # STAGE-3 association: 3D ELLIPSOIDAL gate (xy and z scaled separately so
        # a pyramid keeps distinct tracks per layer — XY-only collapsed them).
        gp('assoc_gate_xy_m', 0.035)
        gp('assoc_gate_z_m', 0.018)
        gp('min_hits', 3)              # consecutive matches before a track renders
        gp('points_voxel_m', 0.004)    # deterministic /points downsample

        # Cup geometry — MUST match params.yaml point_cloud_node's Speed Stack
        # model (top 0.054 < bottom 0.078 ⇒ wide circle at BOTTOM, not upside
        # down). The fusion launch also passes these from params.yaml so the two
        # nodes never diverge.
        gp('cup_top_diameter_m', 0.054)
        gp('cup_bottom_diameter_m', 0.078)
        gp('cup_height_m', 0.095)
        gp('cup_fit_residual_max', 0.02)
        gp('cup_polygon_segments', 24)
        gp('cup_class_names', ['upright-cup'])
        gp('box_standing_ratio', 0.8)
        gp('box_min_elongation', 1.5)

        # View weighting (Phase C; skeleton merges unweighted but reads them).
        gp('w_hand_base', 0.6)
        gp('w_exo_base', 0.4)
        gp('hand_motion_gating', True)

        # Kalman filter (per physical cup; mirror point_cloud_node defaults).
        gp('kf_process_std_xy_m', 0.005)
        gp('kf_process_std_z_m', 0.01)
        gp('kf_meas_std_xy_m', 0.01)
        gp('kf_meas_std_z_m', 0.02)
        gp('kf_init_std_m', 0.1)
        gp('kf_gate_mahalanobis', 9.0)
        gp('kf_settled_std_m', 0.01)
        gp('keepalive_ticks', 8)

        # ── scan & lock mode (multi-view accumulate-at-waypoints) ───────────
        # The arm parks at 2 scan waypoints; while it dwells we accumulate that
        # view's world clouds per cup, fit once, and publish a stable latched
        # lock. Arrival is detected by matching /joint_states to the waypoints
        # (mapped BY NAME — the M0609 wire order is 1,2,4,5,3,6).
        gp('scan_lock_active', False)
        gp('scan_joint_states_topic', '/joint_states')
        gp('scan_joint_names',
           ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6'])
        gp('scan_wp1_deg',
           [57.4585, 10.3361, 75.0595, -31.1839, 127.2512, 44.6653])
        gp('scan_wp2_deg',
           [-42.1911, 12.8098, 66.9024, 25.0818, 107.6920, -18.8562])
        gp('scan_arrival_tol_rad', 0.02)     # per-joint |q-wp| arrival gate
        gp('scan_settle_vel_rad_s', 0.05)    # max joint speed to be "settled"
        gp('scan_wait_s', 1.0)               # dwell-in wait before capturing
        gp('scan_capture_s', 1.0)            # capture/accumulation duration
        gp('scan_js_timeout_s', 0.5)         # joints older than this → UNKNOWN
        # Looser merge/fit gates for the COMPUTE pass: a multi-view union of one
        # cup is more spread (top disc + side slab) than a single view.
        gp('scan_merge_dist_m', 0.06)
        gp('scan_premerge_dxy_m', 0.045)
        gp('scan_premerge_dz_m', 0.05)
        gp('scan_premerge_iou', 0.20)
        gp('scan_premerge_radius_m', 0.07)
        gp('scan_postmerge_dxy_m', 0.04)
        gp('scan_postmerge_dz_m', 0.04)
        gp('scan_max_cup_footprint_m', 0.14)
        gp('scan_fit_voxel_m', 0.004)        # consolidation voxel (bound growth)
        gp('scan_max_points_per_cup', 6000)

        def P(n):
            return self.get_parameter(n).value

        self.world_frame = str(P('world_frame'))
        self.max_age = float(P('max_age_s'))
        self.merge_dist = float(P('merge_dist_m'))
        self.voxel_m = float(P('fusion_voxel_m'))
        self.max_merge_points = int(P('max_merge_points'))
        self.max_cup_footprint = float(P('max_cup_footprint_m'))
        self.premerge_iou = float(P('premerge_iou'))
        self.premerge_dxy = float(P('premerge_dxy_m'))
        self.premerge_dz = float(P('premerge_dz_m'))
        self.premerge_radius = float(P('premerge_radius_m'))
        self.postmerge_dxy = float(P('postmerge_dxy_m'))
        self.postmerge_dz = float(P('postmerge_dz_m'))
        self.assoc_gate_xy = float(P('assoc_gate_xy_m'))
        self.assoc_gate_z = float(P('assoc_gate_z_m'))
        self.min_hits = int(P('min_hits'))
        self.points_voxel = float(P('points_voxel_m'))
        self.cup_top_d = float(P('cup_top_diameter_m'))
        self.cup_bot_d = float(P('cup_bottom_diameter_m'))
        self.cup_h = float(P('cup_height_m'))
        self.cup_resid_max = float(P('cup_fit_residual_max'))
        self.cup_n_seg = int(P('cup_polygon_segments'))
        self.cup_class_names = set(P('cup_class_names'))
        self.standing_ratio = float(P('box_standing_ratio'))
        self.min_elongation = float(P('box_min_elongation'))
        self.w_hand_base = float(P('w_hand_base'))
        self.w_exo_base = float(P('w_exo_base'))
        self.hand_gating = bool(P('hand_motion_gating'))
        self.kf_gate = float(P('kf_gate_mahalanobis'))
        self.kf_settled_std = float(P('kf_settled_std_m'))
        self.keepalive = int(P('keepalive_ticks'))

        self.kf_proc_xy = float(P('kf_process_std_xy_m'))
        self.kf_proc_z = float(P('kf_process_std_z_m'))
        self.kf_meas_xy = float(P('kf_meas_std_xy_m'))
        self.kf_meas_z = float(P('kf_meas_std_z_m'))
        self._recompute_kf()
        i0 = float(P('kf_init_std_m')) ** 2
        self.p0_diag = np.array([i0, i0, i0], dtype=np.float64)

        # ── scan & lock: read params, precompute waypoints, init FSM state ──
        self.scan_lock_active = bool(P('scan_lock_active'))
        self.scan_joint_states_topic = str(P('scan_joint_states_topic'))
        self.scan_joint_names = list(P('scan_joint_names'))
        self._wp_rad = {
            1: np.deg2rad(np.asarray(P('scan_wp1_deg'), dtype=float)),
            2: np.deg2rad(np.asarray(P('scan_wp2_deg'), dtype=float)),
        }
        self.scan_tol = float(P('scan_arrival_tol_rad'))
        self.scan_settle_vel = float(P('scan_settle_vel_rad_s'))
        self.scan_wait_s = float(P('scan_wait_s'))
        self.scan_capture_s = float(P('scan_capture_s'))
        self.scan_js_timeout = float(P('scan_js_timeout_s'))
        self.scan_merge_dist = float(P('scan_merge_dist_m'))
        self.scan_premerge_dxy = float(P('scan_premerge_dxy_m'))
        self.scan_premerge_dz = float(P('scan_premerge_dz_m'))
        self.scan_premerge_iou = float(P('scan_premerge_iou'))
        self.scan_premerge_radius = float(P('scan_premerge_radius_m'))
        self.scan_postmerge_dxy = float(P('scan_postmerge_dxy_m'))
        self.scan_postmerge_dz = float(P('scan_postmerge_dz_m'))
        self.scan_max_cup_footprint = float(P('scan_max_cup_footprint_m'))
        self.scan_fit_voxel = float(P('scan_fit_voxel_m'))
        self.scan_max_points = int(P('scan_max_points_per_cup'))
        for k in (1, 2):
            if self._wp_rad[k].shape[0] != 6:
                self.get_logger().error(
                    f'scan_wp{k}_deg must have 6 values, '
                    f'got {self._wp_rad[k].shape[0]}')
        if not (0.005 <= self.scan_tol <= 0.1):
            self.get_logger().warn(
                f'scan_arrival_tol_rad={self.scan_tol} outside [0.005, 0.1]')
        wp_sep = float(np.max(np.abs(self._wp_rad[1] - self._wp_rad[2])))
        if wp_sep < 2.0 * self.scan_tol:
            self.get_logger().warn(
                f'scan waypoints close (inf-norm {wp_sep:.3f} rad < 2*tol) — '
                f'captures may be mis-attributed')
        # SINGLE-THREADED executor (rclpy.spin): the joint_states callback, the
        # 2 cloud callbacks, the timer, ~/clear_scan and the param callback are
        # all serialized on ONE thread → no locks needed for the state below.
        self._mode = 'OFF'             # OFF(live) | ACTIVE(capturing) | PAUSED
        self._scan_state = 'IDLE'      # IDLE | WAIT | CAPTURE
        self._cur_wp = None
        self._t_arrive = None
        self._t_cap = None
        self._captured = {1: False, 2: False}
        self._scan_visited: set = set()   # waypoints captured this pass
        self._scan_done = False           # both captured → stop diag logging
        self._have_lock = False
        self._pending_clear = False
        self._accum: list[dict] = []
        self._scan_proc_stamp = {'exo': -1.0, 'hand': -1.0}
        self._locked_tracks: dict[int, dict] = {}
        self._locked_ids: set[int] = set()
        self._locked_points_xyz = None
        self._locked_points_rgb = None
        self._scan_last_js = None      # (t_stamp_s, q_rad) for velocity calc
        self._cur_q = None
        self._cur_vmax = float('inf')
        self._js_t = None              # wall-clock arrival of last good js

        latched = QoSProfile(
            depth=1, reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)

        self._latest: dict[str, tuple] = {'exo': (None, None), 'hand': (None, None)}
        # Last-processed cloud stamp per camera — a KF update only fires when a
        # camera's stamp advances, so a stale cloud is not counted repeatedly
        # (which would make the filter over-confident and the box flicker).
        self._proc_stamp: dict[str, float] = {'exo': -1.0, 'hand': -1.0}
        self._tracks: dict[int, dict] = {}
        self._next_gid = 1
        self._last_ids: set[int] = set()

        self.create_subscription(
            WorldObjectCloudArray, str(P('exo_clouds_topic')),
            lambda m: self._on_clouds(m, 'exo'), 10)
        self.create_subscription(
            WorldObjectCloudArray, str(P('hand_clouds_topic')),
            lambda m: self._on_clouds(m, 'hand'), 10)
        self.create_subscription(
            JointState, self.scan_joint_states_topic, self._on_joint_states, 10)
        self._clear_scan_srv = self.create_service(
            Trigger, '~/clear_scan', self._on_clear_scan)
        self.dbg_detections = bool(P('dbg_detections'))
        self.dbg_premerge = bool(P('dbg_premerge'))
        self.dbg_fit1 = bool(P('dbg_fit1'))
        self.dbg_fit2 = bool(P('dbg_fit2'))

        self.boxes_pub = self.create_publisher(
            MarkerArray, str(P('boxes_topic')), latched)
        self.points_pub = self.create_publisher(
            PointCloud2, str(P('points_topic')), 5)
        self.dbg_det_pub = self.create_publisher(
            MarkerArray, str(P('dbg_detections_topic')), latched)
        self.dbg_premerge_pub = self.create_publisher(
            MarkerArray, str(P('dbg_premerge_topic')), latched)
        self.dbg_fit1_pub = self.create_publisher(
            MarkerArray, str(P('dbg_fit1_topic')), latched)

        self.create_timer(float(P('fusion_period_s')), self._tick)
        # Live-tunable thresholds — `ros2 param set /cup_fusion_node <p> <v>`
        # takes effect immediately (no relaunch needed while tuning).
        self._tunable = {
            'max_age_s': ('max_age', float),
            'merge_dist_m': ('merge_dist', float),
            'max_cup_footprint_m': ('max_cup_footprint', float),
            'premerge_iou': ('premerge_iou', float),
            'premerge_dxy_m': ('premerge_dxy', float),
            'premerge_dz_m': ('premerge_dz', float),
            'postmerge_dxy_m': ('postmerge_dxy', float),
            'postmerge_dz_m': ('postmerge_dz', float),
            'assoc_gate_xy_m': ('assoc_gate_xy', float),
            'assoc_gate_z_m': ('assoc_gate_z', float),
            'min_hits': ('min_hits', int),
            'keepalive_ticks': ('keepalive', int),
            'dbg_detections': ('dbg_detections', bool),
            'dbg_premerge': ('dbg_premerge', bool),
            'dbg_fit1': ('dbg_fit1', bool),
            'dbg_fit2': ('dbg_fit2', bool),
            # KF smoothing / view weighting / fit tolerance (issue tuning).
            'kf_gate_mahalanobis': ('kf_gate', float),
            'kf_meas_std_xy_m': ('kf_meas_xy', float),
            'kf_meas_std_z_m': ('kf_meas_z', float),
            'kf_process_std_xy_m': ('kf_proc_xy', float),
            'kf_process_std_z_m': ('kf_proc_z', float),
            'w_exo_base': ('w_exo_base', float),
            'w_hand_base': ('w_hand_base', float),
            'cup_fit_residual_max': ('cup_resid_max', float),
            # scan & lock live-tunables (scan_lock_active drives the mode)
            'scan_lock_active': ('scan_lock_active', bool),
            'scan_arrival_tol_rad': ('scan_tol', float),
            'scan_settle_vel_rad_s': ('scan_settle_vel', float),
            'scan_wait_s': ('scan_wait_s', float),
            'scan_capture_s': ('scan_capture_s', float),
            'scan_js_timeout_s': ('scan_js_timeout', float),
            'scan_merge_dist_m': ('scan_merge_dist', float),
            'scan_premerge_dxy_m': ('scan_premerge_dxy', float),
            'scan_premerge_dz_m': ('scan_premerge_dz', float),
            'scan_premerge_iou': ('scan_premerge_iou', float),
            'scan_premerge_radius_m': ('scan_premerge_radius', float),
            'scan_postmerge_dxy_m': ('scan_postmerge_dxy', float),
            'scan_postmerge_dz_m': ('scan_postmerge_dz', float),
            'scan_max_cup_footprint_m': ('scan_max_cup_footprint', float),
            'scan_fit_voxel_m': ('scan_fit_voxel', float),
            'scan_max_points_per_cup': ('scan_max_points', int),
        }
        self.add_on_set_parameters_callback(self._on_set_params)
        self.get_logger().info(
            f'cup_fusion_node ready (period={float(P("fusion_period_s")):.3f}s, '
            f'merge_dist={self.merge_dist:.3f}m, max_age={self.max_age:.2f}s, '
            f'hand_gating={self.hand_gating})')

    def _recompute_kf(self) -> None:
        self.q_diag = np.array(
            [self.kf_proc_xy ** 2, self.kf_proc_xy ** 2, self.kf_proc_z ** 2])
        self.r_diag = np.array(
            [self.kf_meas_xy ** 2, self.kf_meas_xy ** 2, self.kf_meas_z ** 2])

    def _on_set_params(self, params):
        for p in params:
            spec = self._tunable.get(p.name)
            if spec is not None:
                attr, cast = spec
                setattr(self, attr, cast(p.value))
        if any(p.name.startswith(('kf_meas_std', 'kf_process_std'))
               for p in params):
            self._recompute_kf()
        return SetParametersResult(successful=True)

    # ------------------------------------------------------------------
    def _on_clouds(self, msg: WorldObjectCloudArray, cam: str) -> None:
        self._latest[cam] = (msg, self.get_clock().now())

    def _gather(self):
        """Collect objects for THIS measurement event. Returns [] (→ coast,
        no KF update) unless at least one camera produced a NEW cloud since the
        last event; when fresh, uses the latest cloud from BOTH cameras (within
        max_age) so a cup seen by both is fused in one fit."""
        now = self.get_clock().now()
        snaps = {}
        fresh = False
        for cam in ('exo', 'hand'):
            msg, t = self._latest[cam]
            if msg is None or (now - t).nanoseconds * 1e-9 > self.max_age:
                continue
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            snaps[cam] = (msg, stamp)
            if stamp > self._proc_stamp[cam]:
                fresh = True
        if not fresh:
            return None      # not a fresh event → caller skips, markers persist
        objs = []
        for cam, (msg, stamp) in snaps.items():
            self._proc_stamp[cam] = stamp
            for o in msg.objects:
                if cam == 'hand' and o.moving and self.hand_gating:
                    continue
                xyz, rgb = _pc2_xyzrgb(o.points)
                if xyz.shape[0] < 32:
                    continue
                w_base = self.w_hand_base if cam == 'hand' else self.w_exo_base
                objs.append({
                    'cam': cam, 'xyz': xyz, 'rgb': rgb,
                    'centroid': np.array([o.centroid.x, o.centroid.y,
                                          o.centroid.z], dtype=np.float64),
                    'score': float(o.score), 'class_name': o.class_name,
                    'moving': bool(o.moving), 'w': w_base,
                })
        return objs

    def _merge(self, members: list[dict]) -> np.ndarray:
        """View-weighted, density-equalised merge of a cluster's points.
        Each view is voxel-downsampled (kills the hand's near-field over-density)
        then subsampled to a count proportional to its weight (base hand 0.6 /
        exo 0.4), so the cup fit isn't dominated by whichever view has more
        points. A hand view flagged `moving` is already dropped in _gather."""
        total_w = sum(m['w'] for m in members) or 1.0
        mean_w = total_w / max(len(members), 1)
        chunks = []
        for m in members:
            # Per-view voxel size scaled by weight (heavier view → denser),
            # DETERMINISTIC — no per-frame random subsample (that made the fit
            # jitter, the association flap, and the /points dance).
            vsz = self.voxel_m / np.sqrt(max(m['w'] / mean_w, 1e-6))
            chunks.append(_voxel_downsample(m['xyz'], vsz))
        out = np.vstack(chunks) if chunks else np.zeros((0, 3))
        if out.shape[0] > self.max_merge_points:
            k = int(np.ceil(out.shape[0] / self.max_merge_points))
            out = out[::k]                            # deterministic stride
        return out

    def _fit(self, pts: np.ndarray, members: list[dict]):
        """Return (center3, R3x3, size3, pose_kind, frustum|None, residual) or
        None. Mirrors point_cloud_node._fit_and_render_state's measurement."""
        cup_kind = any(m['class_name'] in self.cup_class_names for m in members)
        if cup_kind:
            fit = _fit_cup_axis_xy(pts, top_d=self.cup_top_d,
                                   bot_d=self.cup_bot_d, height=self.cup_h)
            if fit is not None and fit[3] <= self.cup_resid_max:
                cx, cy, z_base, residual = fit
                center = np.array([cx, cy, z_base + 0.5 * self.cup_h])
                d_max = max(self.cup_top_d, self.cup_bot_d)
                size = np.array([d_max, d_max, self.cup_h])
                frustum = _cup_frustum_geometry(
                    cx, cy, top_d=self.cup_top_d, bot_d=self.cup_bot_d,
                    height=self.cup_h, floor_z=z_base, n_seg=self.cup_n_seg)
                return center, np.eye(3), size, 'standing', frustum, residual
        box = _compute_box_world(pts, standing_ratio=self.standing_ratio,
                                 min_elongation=self.min_elongation,
                                 force_aabb=False)
        if box is None:
            return None
        center, R, size, kind = box
        if max(float(size[0]), float(size[1])) > self.max_cup_footprint:
            return None     # over-merged cluster (several cups) or noise — not
            #                 one cup; dropping it avoids a giant box.
        return center, R, size, kind, None, 0.0

    def _resolve_gid(self, center: np.ndarray) -> int:
        """Associate a fitted centre to an existing track by a 3D ELLIPSOIDAL
        gate (XY and Z scaled independently). XY-only distance collapsed a
        pyramid's vertical column into one track; an ellipsoid keeps layers
        apart (Z gate < layer spacing) while merging cross-view jitter (XY)."""
        best, best_cost = None, 1.0
        for gid, tr in self._tracks.items():
            d = center - tr['kf'].x
            cost = ((d[0] ** 2 + d[1] ** 2) / (self.assoc_gate_xy ** 2)
                    + d[2] ** 2 / (self.assoc_gate_z ** 2))
            if cost < best_cost:
                best_cost, best = cost, gid
        if best is not None:           # cost < 1.0 ⇒ inside the ellipsoid
            return best
        gid = self._next_gid
        self._next_gid += 1
        return gid

    # ------------------------------------------------------------------
    def _premerge(self, objs):
        """STAGE-0/1: coarse centroid cluster, then within each cluster merge
        ONLY detections that are the SAME physical cup (AABB-IoU & dXY & dZ, all
        strict). Returns groups of obj indices = one group per physical cup."""
        clusters = _cluster_indices([o['centroid'] for o in objs], self.merge_dist)
        groups = []
        for cl in clusters:
            if len(cl) == 1:
                groups.append(cl)
                continue
            edges = []
            for a in range(len(cl)):
                for b in range(a + 1, len(cl)):
                    i, j = cl[a], cl[b]
                    dc = objs[i]['centroid'] - objs[j]['centroid']
                    if (abs(dc[2]) <= self.premerge_dz
                            and float(np.hypot(dc[0], dc[1])) <= self.premerge_dxy
                            and _aabb_iou(objs[i]['aabb'], objs[j]['aabb'])
                            >= self.premerge_iou):
                        edges.append((a, b))
            for grp in _union_find_groups(len(cl), edges):
                groups.append([cl[k] for k in grp])
        return groups

    # ================= scan & lock mode =================================
    def _on_joint_states(self, msg: JointState) -> None:
        """Dumb cache (no FSM here): map joints BY NAME (M0609 wire order is
        1,2,4,5,3,6, so index access would be wrong), store the vector + the
        max joint speed. _tick reads this once per tick to run the FSM."""
        jmap = dict(zip(msg.name, msg.position))
        try:
            q = np.array([jmap[n] for n in self.scan_joint_names], dtype=float)
        except KeyError:
            # A name in scan_joint_names is absent from /joint_states. Common
            # cause: a publisher using non-underscored names (e.g. the
            # robot_pose_bridge idle fallback 'joint1'..'joint6'). Without this
            # the arm stays UNKNOWN forever and scan-lock silently never fires.
            self.get_logger().warn(
                f'/joint_states names {list(msg.name)} do not contain '
                f'{self.scan_joint_names} — scan-lock cannot detect arrival',
                throttle_duration_sec=10.0)
            return
        if not np.all(np.isfinite(q)):
            return
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        vmax = float('inf')                     # until a 2nd sample gives dt
        if self._scan_last_js is not None:
            t0, q0 = self._scan_last_js
            dt = t - t0
            if dt > 1e-3:
                vmax = float(np.max(np.abs(q - q0)) / dt)
            else:
                vmax = self._cur_vmax           # reuse last good (don't /noise)
        self._scan_last_js = (t, q)
        self._cur_q, self._cur_vmax = q, vmax
        self._js_t = self.get_clock().now()

    def _on_clear_scan(self, request, response):
        """std_srvs/Trigger ~/clear_scan: defer the wipe to the next _tick so
        all state mutation happens on the timer thread, in order."""
        self._pending_clear = True
        response.success = True
        response.message = 'scan lock cleared; back to live detection'
        return response

    def _scan_snapshot(self, now):
        """(settled, near_enter{k}, near_leave{k}, unknown) from the cached
        joint state. UNKNOWN if joints are stale / unseen / first sample."""
        unknown = (self._cur_q is None or self._js_t is None
                   or not np.isfinite(self._cur_vmax)
                   or (now - self._js_t).nanoseconds * 1e-9 > self.scan_js_timeout)
        if unknown:
            return False, {1: False, 2: False}, {1: False, 2: False}, True
        settled = self._cur_vmax < self.scan_settle_vel
        ne, nl = {}, {}
        for k in (1, 2):
            d = float(np.max(np.abs(self._cur_q - self._wp_rad[k])))
            ne[k] = d <= self.scan_tol
            nl[k] = d <= 1.5 * self.scan_tol    # hysteresis band for "leaving"
        return settled, ne, nl, False

    def _scan_fsm(self, now) -> None:
        settled, near, near_leave, unknown = self._scan_snapshot(now)

        # Leave-latch: re-arm a waypoint once the arm has clearly left it, so a
        # 2nd scan pass re-captures (and accumulates) at both waypoints. Runs
        # every tick, including PAUSED.
        for k in (1, 2):
            if self._captured[k] and not near_leave[k]:
                self._captured[k] = False

        # Abort an in-flight visit if the arm left the waypoint, went UNKNOWN,
        # or scanning was paused. Any partial accumulation is KEPT (valid world
        # observations); no COMPUTE on a truncated dwell; captured stays False.
        if self._scan_state in ('WAIT', 'CAPTURE'):
            if (unknown or self._cur_wp is None
                    or not near_leave[self._cur_wp]
                    or self._mode != 'ACTIVE'):
                self._scan_state = 'IDLE'
                self._cur_wp = None
                return

        if self._mode != 'ACTIVE':
            return

        if self._scan_state == 'IDLE':
            if unknown:
                self.get_logger().warn(
                    'scan: no fresh /joint_states (arm state UNKNOWN) — check '
                    'the joint source/bridge; cannot detect waypoint arrival',
                    throttle_duration_sec=3.0)
                return
            tol_deg = np.degrees(self.scan_tol)
            cand = [k for k in (1, 2)
                    if near[k] and settled and not self._captured[k]]
            if cand:                            # nearest wins (loose-tol tie)
                k = min(cand, key=lambda j: float(
                    np.max(np.abs(self._cur_q - self._wp_rad[j]))))
                err = np.degrees(np.max(np.abs(self._cur_q - self._wp_rad[k])))
                self.get_logger().info(
                    f'scan: ▶ ARRIVED at pos{k} (max joint err {err:.2f}° ≤ '
                    f'tol {tol_deg:.2f}°, settled) — wait {self.scan_wait_s:.1f}s '
                    f'then capture {self.scan_capture_s:.1f}s')
                self._cur_wp = k
                self._t_arrive = now
                self._scan_state = 'WAIT'
            elif not self._scan_done:
                # Approaching diagnostic — ONCE PER SECOND, and ONLY until the
                # pass is done (both waypoints captured). Shows how far the arm
                # is from each waypoint so the motion + arrival detection can be
                # verified live. Goes silent after the scan finishes.
                d1 = np.degrees(np.max(np.abs(self._cur_q - self._wp_rad[1])))
                d2 = np.degrees(np.max(np.abs(self._cur_q - self._wp_rad[2])))
                self.get_logger().info(
                    f'scan: waiting for a waypoint — pos1 err {d1:.1f}°, pos2 '
                    f'err {d2:.1f}° (tol {tol_deg:.1f}°), settled={settled}',
                    throttle_duration_sec=1.0)
        elif self._scan_state == 'WAIT':
            if (now - self._t_arrive).nanoseconds * 1e-9 >= self.scan_wait_s:
                self._t_cap = now
                self._cap_n0 = len(self._accum)
                self.get_logger().info(
                    f'scan: ● CAPTURING pos{self._cur_wp} for '
                    f'{self.scan_capture_s:.1f}s (accumulating world clouds)')
                # NOTE: do NOT reset _scan_proc_stamp here. It is reset once per
                # SESSION (in _enter_active) and kept monotonic, so a stale
                # producer cloud (e.g. exo not republished between wp1→wp2) is
                # consumed at most once total — never re-appended at the 2nd wp.
                self._scan_state = 'CAPTURE'
        elif self._scan_state == 'CAPTURE':
            self._capture_into_accum()
            if (now - self._t_cap).nanoseconds * 1e-9 >= self.scan_capture_s:
                k = self._cur_wp
                added = len(self._accum) - getattr(self, '_cap_n0', 0)
                self.get_logger().info(
                    f'scan: ✓ pos{k} capture done (+{added} observations) → '
                    f'computing lock')
                self._compute_locked()
                self._captured[k] = True
                self._scan_visited.add(k)
                self._cur_wp = None
                self._scan_state = 'IDLE'
                if self._scan_visited >= {1, 2} and not self._scan_done:
                    self._scan_done = True
                    self.get_logger().info(
                        'scan: ✔ pass complete — pos1 & pos2 captured, lock '
                        'published; diagnostic logging stops (clear or '
                        're-activate to scan again)')

    def _capture_into_accum(self) -> None:
        """Append fresh per-camera detection-objs (already world-frame) to the
        accumulation buffer. Dedups by per-camera stamp so a slow producer's
        repeated latest-cloud isn't appended every tick."""
        now = self.get_clock().now()
        for cam in ('exo', 'hand'):
            msg, t = self._latest[cam]
            if msg is None or (now - t).nanoseconds * 1e-9 > self.max_age:
                continue
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            if stamp <= self._scan_proc_stamp[cam]:
                continue                        # same cloud already captured
            self._scan_proc_stamp[cam] = stamp
            for o in msg.objects:
                if cam == 'hand' and o.moving and self.hand_gating:
                    continue
                xyz, rgb = _pc2_xyzrgb(o.points)
                if xyz.shape[0] < 32:
                    continue
                self._accum.append({
                    'cam': cam, 'xyz': xyz, 'rgb': rgb,
                    'centroid': np.array([o.centroid.x, o.centroid.y,
                                          o.centroid.z], dtype=np.float64),
                    'aabb': _aabb_robust(xyz),
                    'score': float(o.score), 'class_name': o.class_name,
                    'moving': bool(o.moving),
                    'w': self.w_hand_base if cam == 'hand' else self.w_exo_base,
                })

    @contextmanager
    def _scan_params(self):
        """Temporarily swap the live merge/fit gates to the looser scan_* values
        for the COMPUTE pass (a multi-view union of one cup is more spread than a
        single view). Single-threaded → safe to mutate instance attrs."""
        attrs = ('merge_dist', 'premerge_dxy', 'premerge_dz', 'premerge_iou',
                 'postmerge_dxy', 'postmerge_dz', 'max_cup_footprint')
        saved = {a: getattr(self, a) for a in attrs}
        self.merge_dist = self.scan_merge_dist
        self.premerge_dxy = self.scan_premerge_dxy
        self.premerge_dz = self.scan_premerge_dz
        self.premerge_iou = self.scan_premerge_iou
        self.postmerge_dxy = self.scan_postmerge_dxy
        self.postmerge_dz = self.scan_postmerge_dz
        self.max_cup_footprint = self.scan_max_cup_footprint
        try:
            yield
        finally:
            for a, v in saved.items():
                setattr(self, a, v)

    def _run_fit_pipeline(self, objs):
        """premerge → fit each group → STAGE-2b post-merge + refit. Returns the
        list of final fit dicts. Same maths as _tick_live, on accumulated objs."""
        fits = []
        for grp in self._premerge(objs):
            members = [objs[i] for i in grp]
            meas = self._fit(self._merge(members), members)
            if meas is not None:
                c, R, size, kind, frustum, residual = meas
                fits.append({'center': c, 'R': R, 'size': size, 'kind': kind,
                             'frustum': frustum, 'residual': residual,
                             'members': members})
        fedges = []
        for a in range(len(fits)):
            for b in range(a + 1, len(fits)):
                dc = fits[a]['center'] - fits[b]['center']
                if (abs(dc[2]) <= self.postmerge_dz
                        and float(np.hypot(dc[0], dc[1])) <= self.postmerge_dxy):
                    fedges.append((a, b))
        final_fits = []
        for grp in _union_find_groups(len(fits), fedges):
            if len(grp) == 1:
                final_fits.append(fits[grp[0]])
                continue
            members = [m for k in grp for m in fits[k]['members']]
            meas = self._fit(self._merge(members), members)
            if meas is None:
                final_fits.append(fits[grp[0]])
                continue
            c, R, size, kind, frustum, residual = meas
            final_fits.append({'center': c, 'R': R, 'size': size, 'kind': kind,
                               'frustum': frustum, 'residual': residual,
                               'members': members})
        return final_fits

    def _compute_locked(self) -> None:
        """COMPUTE: fit the accumulated multi-view clouds, publish a stable
        latched lock, then consolidate the buffer to bound growth."""
        if not self._accum:
            return
        with self._scan_params():
            final_fits = self._run_fit_pipeline(self._accum)
        if not final_fits:                      # keep the previous lock, no wipe
            self.get_logger().warn(
                'scan: waypoint captured 0 cups (keeping previous lock)',
                throttle_duration_sec=5.0)
            return
        stamp = self.get_clock().now().to_msg()
        new_tracks, new_ids = {}, set()
        for i, f in enumerate(final_fits, start=1):
            cams = {m['cam'] for m in f['members']}
            new_tracks[i] = self._render_state(
                i, f['center'], f['R'], f['size'], f['kind'],
                f['frustum'], f['residual'], cams)
            new_ids.add(i)
        self._locked_tracks = new_tracks
        self._publish_lock_markers(stamp, new_ids)
        self._wipe_dbg(stamp)                   # stop stale live dbg overlays
        self._accum = self._consolidate_per_cup(final_fits)
        self._rebuild_locked_points()
        self._have_lock = True
        n_pts = sum(o['xyz'].shape[0] for o in self._accum)
        self.get_logger().info(
            f'scan: locked {len(final_fits)} cup(s) from {n_pts} consolidated '
            f'points')

    def _consolidate_per_cup(self, final_fits):
        """Replace the raw accumulation with ONE synthetic obj per locked cup
        (voxel-downsampled union, capped) so repeated scans don't grow without
        bound. Next pass's single-view detections append and re-cluster against
        these consolidated unions (hence the loosened scan_* gates)."""
        out = []
        for f in final_fits:
            xyz = np.vstack([m['xyz'] for m in f['members']])
            rgb = np.concatenate([m['rgb'] for m in f['members']])
            idx = _voxel_idx(xyz, self.scan_fit_voxel)
            xyz, rgb = xyz[idx], rgb[idx]
            if xyz.shape[0] > self.scan_max_points:
                k = int(np.ceil(xyz.shape[0] / self.scan_max_points))
                xyz, rgb = xyz[::k], rgb[::k]
            cup = any(m['class_name'] in self.cup_class_names
                      for m in f['members'])
            cls = (next(m['class_name'] for m in f['members']
                        if m['class_name'] in self.cup_class_names) if cup
                   else f['members'][0]['class_name'])
            out.append({
                'cam': 'scan', 'xyz': xyz, 'rgb': rgb,
                'centroid': xyz.mean(axis=0), 'aabb': _aabb_robust(xyz),
                'score': max(m['score'] for m in f['members']),
                'class_name': cls, 'moving': False, 'w': self.w_hand_base,
            })
        return out

    def _rebuild_locked_points(self) -> None:
        """Cache the consolidated union for per-tick /points republish."""
        if self._accum:
            self._locked_points_xyz = np.vstack(
                [o['xyz'] for o in self._accum]).astype(np.float32)
            self._locked_points_rgb = np.concatenate(
                [o['rgb'] for o in self._accum])
        else:
            self._locked_points_xyz = np.zeros((0, 3), np.float32)
            self._locked_points_rgb = np.zeros((0,), np.float32)

    def _publish_lock_markers(self, stamp, new_ids) -> None:
        """Latched: ONE MarkerArray that DELETEALLs (clears any live fusion_*
        AND prior lock_* boxes atomically) then ADDs every locked cup."""
        markers = MarkerArray()
        clr = Marker()
        clr.header.frame_id = self.world_frame
        clr.header.stamp = stamp
        clr.action = Marker.DELETEALL
        markers.markers.append(clr)
        for gid in sorted(new_ids):
            ls = self._locked_tracks[gid]
            self._box_markers(markers, gid, ls, _palette(gid - 1), stamp,
                              settled=True, ns_prefix='lock_')
        self.boxes_pub.publish(markers)
        self._locked_ids = set(new_ids)
        self._last_ids = set()                  # the live render set is now void

    def _wipe_dbg(self, stamp) -> None:
        for pub in (self.dbg_det_pub, self.dbg_premerge_pub, self.dbg_fit1_pub):
            ma = MarkerArray()
            m = Marker()
            m.header.frame_id = self.world_frame
            m.header.stamp = stamp
            m.action = Marker.DELETEALL
            ma.markers.append(m)
            pub.publish(ma)

    def _enter_active(self) -> None:
        """OFF → ACTIVE: start a fresh scan session. Live boxes keep rendering
        until the first lock exists (then _compute_locked wipes them)."""
        self._accum = []
        self._captured = {1: False, 2: False}
        self._scan_visited = set()
        self._scan_done = False
        self._scan_state = 'IDLE'
        self._cur_wp = None
        # Monotonic across the whole session (NOT per-visit) so each producer
        # cloud is consumed at most once total — a stale wp1 cloud can't be
        # re-appended at wp2.
        self._scan_proc_stamp = {'exo': -1.0, 'hand': -1.0}
        self.get_logger().info('scan-lock ACTIVE — capturing at waypoints')

    def _do_clear(self, stamp) -> None:
        """Full reset → live detection. Wipes lock + live markers + buffers."""
        clr = MarkerArray()
        m = Marker()
        m.header.frame_id = self.world_frame
        m.header.stamp = stamp
        m.action = Marker.DELETEALL
        clr.markers.append(m)
        self.boxes_pub.publish(clr)
        self._wipe_dbg(stamp)
        self.points_pub.publish(_make_pointcloud2(
            Header(stamp=stamp, frame_id=self.world_frame),
            np.zeros((0, 3), np.float32), np.zeros((0,), np.float32)))
        self._locked_tracks = {}
        self._locked_ids = set()
        self._locked_points_xyz = None
        self._locked_points_rgb = None
        self._accum = []
        self._captured = {1: False, 2: False}
        self._scan_visited = set()
        self._scan_done = False
        self._scan_state = 'IDLE'
        self._cur_wp = None
        self._have_lock = False
        self._mode = 'OFF'
        self._scan_proc_stamp = {'exo': -1.0, 'hand': -1.0}
        # Drive the ROS PARAM (source of truth), not just the attribute, so the
        # param, the cached attr and any re-sync/save stay consistent — and a
        # later re-check (False→True) actually re-fires _on_set_params. (Routes
        # through _on_set_params: idempotent setattr, no kf recompute.)
        self.set_parameters([rclpy.parameter.Parameter(
            'scan_lock_active', rclpy.parameter.Parameter.Type.BOOL, False)])
        # Reset live-pipeline state so it resumes clean.
        self._tracks = {}
        self._next_gid = 1
        self._last_ids = set()
        self._proc_stamp = {'exo': -1.0, 'hand': -1.0}
        self.get_logger().info('scan-lock CLEARED — back to live detection')

    # ================= tick dispatcher + live pipeline ==================
    def _tick(self) -> None:
        """Dispatcher: live KF pipeline (OFF, or ACTIVE before the first lock)
        vs scan-lock (advance FSM, bypass live KF once a lock exists). All scan
        FSM transitions run HERE, never in the joint_states callback."""
        stamp = self.get_clock().now().to_msg()
        now = self.get_clock().now()

        if self._pending_clear:                 # priority 1: deferred clear
            self._do_clear(stamp)
            self._pending_clear = False

        # "In a session" = a lock exists, points are accumulated, or a visit is
        # in flight. While in a session, unchecking only PAUSES (keeps accum +
        # lock, and lets _scan_fsm abort any in-flight WAIT/CAPTURE); only
        # clear_scan exits to OFF. _enter_active (which WIPES) fires ONLY on a
        # true cold start (OFF→ACTIVE with nothing accumulated) — so the same
        # uncheck gesture never silently discards a partial first pass.
        in_session = (self._have_lock or bool(self._accum)
                      or self._scan_state != 'IDLE')
        if not self.scan_lock_active:
            self._mode = 'PAUSED' if in_session else 'OFF'
        else:
            if self._mode == 'OFF':
                self._enter_active()
            self._mode = 'ACTIVE'

        if self._mode == 'OFF':
            return self._tick_live()

        # ACTIVE/PAUSED: advance the waypoint FSM (capture only when ACTIVE; the
        # leave-latch reset and WAIT/CAPTURE abort run even when PAUSED).
        self._scan_fsm(now)

        if not self._have_lock:
            return self._tick_live()            # no lock yet → keep live alive

        # Locked: republish the cached union on the VOLATILE /points each tick
        # (a late RViz subscriber can't get a latched volatile cloud). Lock
        # boxes are latched from _compute_locked, so no per-tick marker publish.
        # Republish even a 0-point cloud so a stale live frame can't linger.
        if self._locked_points_xyz is not None:
            self.points_pub.publish(_make_pointcloud2(
                Header(stamp=stamp, frame_id=self.world_frame),
                self._locked_points_xyz, self._locked_points_rgb))

    def _tick_live(self) -> None:
        objs = self._gather()
        if objs is None:
            # Not a fresh measurement event → don't touch the markers; the last
            # published set persists in RViz. (No flicker between measurements.)
            return
        for o in objs:
            o['aabb'] = _aabb_robust(o['xyz'])

        # Predict once per measurement event (Q is tuned for this cadence).
        for tr in self._tracks.values():
            tr['kf'].predict()
            tr['settled'] = tr['kf'].position_std() <= self.kf_settled_std

        # STAGE-1: merge duplicate detections of the same cup → fit each group.
        premerge_groups = self._premerge(objs)
        fits = []
        for grp in premerge_groups:
            members = [objs[i] for i in grp]
            meas = self._fit(self._merge(members), members)
            if meas is not None:
                c, R, size, kind, frustum, residual = meas
                fits.append({'center': c, 'R': R, 'size': size, 'kind': kind,
                             'frustum': frustum, 'residual': residual,
                             'members': members})

        # STAGE-2b: re-merge fits that point at the SAME cup (strict 3D) + refit.
        fedges = []
        for a in range(len(fits)):
            for b in range(a + 1, len(fits)):
                dc = fits[a]['center'] - fits[b]['center']
                if (abs(dc[2]) <= self.postmerge_dz
                        and float(np.hypot(dc[0], dc[1])) <= self.postmerge_dxy):
                    fedges.append((a, b))
        final_fits = []
        for grp in _union_find_groups(len(fits), fedges):
            if len(grp) == 1:
                final_fits.append(fits[grp[0]])
                continue
            members = [m for k in grp for m in fits[k]['members']]
            meas = self._fit(self._merge(members), members)
            if meas is None:
                final_fits.append(fits[grp[0]])
                continue
            c, R, size, kind, frustum, residual = meas
            final_fits.append({'center': c, 'R': R, 'size': size, 'kind': kind,
                               'frustum': frustum, 'residual': residual,
                               'members': members})

        # STAGE-3: 3D-ellipsoidal association + KF + hit count.
        alive: set[int] = set()
        for f in final_fits:
            gid = self._resolve_gid(f['center'])
            tr = self._tracks.get(gid)
            if tr is None:
                tr = {'kf': PositionKF(f['center'], self.p0_diag, self.q_diag),
                      'last_state': None, 'miss': 0, 'hits': 1,
                      'settled': False, 'cams': set()}
                self._tracks[gid] = tr
            else:
                tr['kf'].update(f['center'], self.r_diag, self.kf_gate)
                tr['hits'] += 1
            tr['cams'] = {m['cam'] for m in f['members']}
            tr['miss'] = 0
            tr['last_state'] = self._render_state(
                gid, tr['kf'].x.copy(), f['R'], f['size'], f['kind'],
                f['frustum'], f['residual'], tr['cams'])
            alive.add(gid)

        # STAGE-4: coast/evict after keepalive measurement events with no match.
        for gid in list(self._tracks.keys()):
            if gid not in alive:
                self._tracks[gid]['miss'] += 1
                if self._tracks[gid]['miss'] > self.keepalive:
                    self._tracks.pop(gid, None)

        # STAGE-5: render final (stage-2) tracks — only if the dbg_fit2 toggle
        # is on (it is by default). /points = the FULL raw union (every point).
        stamp = self.get_clock().now().to_msg()
        self._publish_markers(stamp, enabled=self.dbg_fit2)
        if objs:
            xyz = np.vstack([o['xyz'] for o in objs]).astype(np.float32)
            rgb = np.concatenate([o['rgb'] for o in objs])
        else:
            xyz = np.zeros((0, 3), np.float32)
            rgb = np.zeros((0,), np.float32)
        self.points_pub.publish(_make_pointcloud2(
            Header(stamp=stamp, frame_id=self.world_frame), xyz, rgb))

        # ── Debug visualisations (each toggled live from the Tk panel) ──
        EXO_C, HAND_C, GRN, CYAN = ((0.4, 0.6, 1.0), (1.0, 0.6, 0.1),
                                    (0.3, 0.9, 0.3), (0.2, 0.9, 0.9))
        # (1) per-detection OBB (from each YOLO mask's cloud), coloured by camera.
        det_boxes = []
        for o in objs:
            box = _compute_box_world(o['xyz'], standing_ratio=self.standing_ratio,
                                     min_elongation=self.min_elongation,
                                     force_aabb=False)
            if box is None:
                continue
            center, R, size, _kind = box
            det_boxes.append({
                'center': center, 'R': R, 'size': size,
                'color': HAND_C if o['cam'] == 'hand' else EXO_C,
                'label': o['cam'], 'frustum': None})
        self._emit_dbg(self.dbg_det_pub, self.dbg_detections, det_boxes, stamp)
        # (2) stage-1 merged-group AABB boxes (pre-fit).
        pm_boxes = []
        for grp in premerge_groups:
            lo, hi = _aabb_robust(np.vstack([objs[i]['xyz'] for i in grp]))
            pm_boxes.append({
                'center': (lo + hi) * 0.5, 'R': np.eye(3), 'size': hi - lo,
                'color': GRN, 'label': f'n{len(grp)}', 'frustum': None})
        self._emit_dbg(self.dbg_premerge_pub, self.dbg_premerge, pm_boxes, stamp)
        # (3) stage-1 (first) fit boxes + cup shape.
        f1_boxes = [{'center': f['center'], 'R': f['R'], 'size': f['size'],
                     'color': CYAN, 'label': f['kind'], 'frustum': f['frustum']}
                    for f in fits]
        self._emit_dbg(self.dbg_fit1_pub, self.dbg_fit1, f1_boxes, stamp)

    def _render_state(self, gid, center, R, size, kind, frustum, residual, cams):
        """Box geometry for marker emission, using the KF-filtered centre."""
        if kind == 'standing' and frustum is not None:
            cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
            z_base = cz - 0.5 * self.cup_h
            frustum = _cup_frustum_geometry(
                cx, cy, top_d=self.cup_top_d, bot_d=self.cup_bot_d,
                height=self.cup_h, floor_z=z_base, n_seg=self.cup_n_seg)
            top_world = np.array([cx, cy, z_base + self.cup_h])
        else:
            top_world = center + R @ np.array([0.0, 0.0, float(size[2]) * 0.5])
        views = '+'.join(sorted(cams))
        label = f"#{gid}_{kind}_{views}_r={residual * 1000:.0f}mm"
        return {'center': np.asarray(center), 'R': R, 'size': np.asarray(size),
                'top_world': top_world, 'frustum': frustum, 'label': label}

    # ------------------------------------------------------------------
    def _emit_dbg(self, pub, enabled, boxes, stamp) -> None:
        """Publish a debug MarkerArray: CUBE + label + optional cup frustum per
        box. Always DELETEALL first, so toggling the checkbox off clears it."""
        ma = MarkerArray()
        clr = Marker()
        clr.header.frame_id = self.world_frame
        clr.header.stamp = stamp
        clr.action = Marker.DELETEALL
        ma.markers.append(clr)
        if enabled:
            for i, bx in enumerate(boxes):
                c, R, size, col = bx['center'], bx['R'], bx['size'], bx['color']
                qx, qy, qz, qw = _rot_to_quat(R)
                cube = Marker()
                cube.header.frame_id = self.world_frame
                cube.header.stamp = stamp
                cube.ns = 'dbg_box'
                cube.id = i
                cube.type = Marker.CUBE
                cube.action = Marker.ADD
                cube.pose.position = MsgPoint(x=float(c[0]), y=float(c[1]),
                                              z=float(c[2]))
                cube.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
                cube.scale = Vector3(x=max(float(size[0]), 1e-3),
                                     y=max(float(size[1]), 1e-3),
                                     z=max(float(size[2]), 1e-3))
                cube.color = ColorRGBA(r=col[0], g=col[1], b=col[2], a=0.25)
                ma.markers.append(cube)
                if bx.get('label'):
                    txt = Marker()
                    txt.header.frame_id = self.world_frame
                    txt.header.stamp = stamp
                    txt.ns = 'dbg_label'
                    txt.id = i
                    txt.type = Marker.TEXT_VIEW_FACING
                    txt.action = Marker.ADD
                    txt.pose.position = MsgPoint(
                        x=float(c[0]), y=float(c[1]),
                        z=float(c[2] + size[2] * 0.5 + 0.02))
                    txt.scale.z = 0.022
                    txt.color = ColorRGBA(r=col[0], g=col[1], b=col[2], a=1.0)
                    txt.text = str(bx['label'])
                    ma.markers.append(txt)
                fr = bx.get('frustum')
                if fr is not None:
                    for ns, loop in (('dbg_top', fr['top_loop']),
                                     ('dbg_bot', fr['bot_loop'])):
                        ln = Marker()
                        ln.header.frame_id = self.world_frame
                        ln.header.stamp = stamp
                        ln.ns = ns
                        ln.id = i
                        ln.type = Marker.LINE_STRIP
                        ln.action = Marker.ADD
                        ln.scale.x = 0.003
                        ln.color = ColorRGBA(r=col[0], g=col[1], b=col[2], a=0.9)
                        ln.pose.orientation.w = 1.0
                        for p in loop:
                            ln.points.append(MsgPoint(x=float(p[0]), y=float(p[1]),
                                                      z=float(p[2])))
                        ma.markers.append(ln)
        pub.publish(ma)

    def _publish_markers(self, stamp, enabled=True) -> None:
        markers = MarkerArray()
        if not enabled:
            clr = Marker()
            clr.header.frame_id = self.world_frame
            clr.header.stamp = stamp
            clr.action = Marker.DELETEALL
            markers.markers.append(clr)
            self._last_ids = set()
            self.boxes_pub.publish(markers)
            return
        current: set[int] = set()
        for gid, tr in sorted(self._tracks.items()):
            ls = tr.get('last_state')
            # min_hits: don't render a track until it has been matched a few
            # consecutive events — kills 1-2 frame spurious fits (anti-flicker).
            if ls is None or tr.get('hits', 0) < self.min_hits:
                continue
            colour = _palette(gid - 1)
            self._box_markers(markers, gid, ls, colour, stamp,
                              settled=tr.get('settled', False))
            current.add(gid)
        # DELETE only tracks that were rendered before and are now GONE
        # (evicted). Coasting tracks keep their markers → no flicker between
        # measurement events.
        for gid in (self._last_ids - current):
            self._delete_markers(markers, gid, stamp)
        self._last_ids = current
        self.boxes_pub.publish(markers)

    def _box_markers(self, markers, gid, ls, colour, stamp, settled,
                     ns_prefix='fusion_'):
        center, R, size = ls['center'], ls['R'], ls['size']
        qx, qy, qz, qw = _rot_to_quat(R)
        cube = Marker()
        cube.header.frame_id = self.world_frame
        cube.header.stamp = stamp
        # CONSUMER CONTRACT ns: skill_manager._on_boxes / boxes_to_detections
        # parse by 'boxes'/'box_top'/'box_labels' (same as the original
        # point_cloud_node). Live and lock share these — they never publish at
        # the same time (mode-gated), and each transition leads with DELETEALL.
        cube.ns = 'boxes'
        cube.id = gid
        cube.type = Marker.CUBE
        cube.action = Marker.ADD
        cube.pose.position = MsgPoint(x=float(center[0]), y=float(center[1]),
                                      z=float(center[2]))
        cube.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
        cube.scale = Vector3(x=float(size[0]), y=float(size[1]),
                             z=float(size[2]))
        cube.color = ColorRGBA(r=colour[0], g=colour[1], b=colour[2], a=0.35)
        markers.markers.append(cube)

        top = ls['top_world']
        sph = Marker()
        sph.header.frame_id = self.world_frame
        sph.header.stamp = stamp
        sph.ns = 'box_top'
        sph.id = gid
        sph.type = Marker.SPHERE
        sph.action = Marker.ADD
        sph.pose.position = MsgPoint(x=float(top[0]), y=float(top[1]),
                                     z=float(top[2]))
        sph.scale = Vector3(x=0.02, y=0.02, z=0.02)
        sph.color = ColorRGBA(r=colour[0], g=colour[1], b=colour[2], a=1.0)
        markers.markers.append(sph)

        txt = Marker()
        txt.header.frame_id = self.world_frame
        txt.header.stamp = stamp
        txt.ns = 'box_labels'
        txt.id = gid
        txt.type = Marker.TEXT_VIEW_FACING
        txt.action = Marker.ADD
        txt.pose.position = MsgPoint(x=float(top[0]), y=float(top[1]),
                                     z=float(top[2]) + 0.04)
        txt.scale.z = 0.03
        txt.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        prefix = '[L]_' if settled else ''
        txt.text = prefix + ls['label']
        markers.markers.append(txt)

        # Cup-shape wireframe (truncated cone): top/bottom loops + generatrix.
        fr = ls.get('frustum')
        if fr is not None:
            for ns, loop in ((ns_prefix + 'cup_top', fr['top_loop']),
                             (ns_prefix + 'cup_bot', fr['bot_loop'])):
                line = Marker()
                line.header.frame_id = self.world_frame
                line.header.stamp = stamp
                line.ns = ns
                line.id = gid
                line.type = Marker.LINE_STRIP
                line.action = Marker.ADD
                line.scale.x = 0.003
                line.color = ColorRGBA(r=colour[0], g=colour[1],
                                       b=colour[2], a=0.9)
                line.pose.orientation.w = 1.0
                for p in loop:
                    line.points.append(MsgPoint(x=float(p[0]), y=float(p[1]),
                                                z=float(p[2])))
                markers.markers.append(line)
            gen = Marker()
            gen.header.frame_id = self.world_frame
            gen.header.stamp = stamp
            gen.ns = ns_prefix + 'cup_gen'
            gen.id = gid
            gen.type = Marker.LINE_LIST
            gen.action = Marker.ADD
            gen.scale.x = 0.003
            gen.color = ColorRGBA(r=colour[0], g=colour[1], b=colour[2], a=0.9)
            gen.pose.orientation.w = 1.0
            for p_top, p_bot in fr['generatrix']:
                gen.points.append(MsgPoint(x=float(p_top[0]), y=float(p_top[1]),
                                           z=float(p_top[2])))
                gen.points.append(MsgPoint(x=float(p_bot[0]), y=float(p_bot[1]),
                                           z=float(p_bot[2])))
            markers.markers.append(gen)

    def _delete_markers(self, markers, gid, stamp, ns_prefix='fusion_'):
        for ns in ('boxes', 'box_top', 'box_labels',
                   ns_prefix + 'cup_top', ns_prefix + 'cup_bot',
                   ns_prefix + 'cup_gen'):
            m = Marker()
            m.header.frame_id = self.world_frame
            m.header.stamp = stamp
            m.ns = ns
            m.id = gid
            m.action = Marker.DELETE
            markers.markers.append(m)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CupFusionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
