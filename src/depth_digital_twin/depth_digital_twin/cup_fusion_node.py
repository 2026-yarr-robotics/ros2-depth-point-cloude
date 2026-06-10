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

import json
from contextlib import contextmanager

import numpy as np
from builtin_interfaces.msg import Duration as DurationMsg
import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
                       QoSReliabilityPolicy)
from geometry_msgs.msg import Point as MsgPoint, Quaternion, Vector3
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import JointState, PointCloud2
from std_msgs.msg import ColorRGBA, Header, Int32MultiArray, String
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray
from scipy.spatial import cKDTree

from depth_digital_twin_msgs.msg import (CupObservationArray,
                                         WorldObjectCloudArray)
from depth_digital_twin.cup_geometry import (fuse_xy, slide_xy_to_z,
                                             snap_level, xy_cov_from_px)

# Single source of truth for the estimation maths — reuse the pure helpers from
# point_cloud_node rather than duplicating them. (A later refactor may move
# these into a dedicated cup_estimation module; importing the node module is
# side-effect-free because main() is guarded by __main__.)
from depth_digital_twin.point_cloud_node import (
    PositionKF, _fit_cup_axis_xy, _compute_box_world, _make_pointcloud2,
    _cup_frustum_geometry, _palette, _rot_to_quat, _classify_color_bgr,
    _filter_spatial_density)


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


def _color_from_packed_rgb(rgb_packed, allowed):
    """Median-HSV color name from a producer's packed-float32 rgb array
    (PointCloud2 'rgb' convention r<<16|g<<8|b), or None. Reuses the standalone
    point_cloud_node bucketing so fused cups carry the SAME color meaning."""
    if rgb_packed is None or len(rgb_packed) == 0:
        return None
    u = np.ascontiguousarray(rgb_packed, dtype=np.float32).view(np.uint32)
    r = ((u >> 16) & 255).astype(np.uint8)
    g = ((u >> 8) & 255).astype(np.uint8)
    b = (u & 255).astype(np.uint8)
    bgr = np.stack([b, g, r], axis=1)
    return _classify_color_bgr(bgr, allowed)


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
        gp('cups_on_table_topic', '/vision/cups_on_table')
        gp('stack_track_ids_topic', '/stack_track_ids')
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
        # (A) Temporal frame ACCUMULATION: union each camera's clouds over the
        # last `fusion_accum_window_s` before fitting, so the fit averages out
        # the raw per-frame depth noise (esp. far exo ±40-80mm, zero-mean → it
        # cancels). 0 = use only the latest cloud (old behaviour). Keep short so
        # a moving cup doesn't smear; static cups on a table benefit most.
        gp('fusion_accum_window_s', 0.3)
        # (B) Spatial-density filter: drop isolated noise islands (a point with
        # < min_neighbors others within radius) from the FIT input — catches the
        # exo flying-pixels the producer MAD can't (in-distribution but lonely).
        gp('use_fusion_density_filter', True)
        gp('fusion_density_radius_m', 0.03)
        gp('fusion_density_min_neighbors', 4)
        # Reject a box bigger than this footprint — it's an over-merge of
        # several cups (or noise), not one cup. Stops giant boxes.
        gp('max_cup_footprint_m', 0.11)

        # STAGE-1 pre-fit point-cloud merge. PRIMARY = cup-cylinder containment
        # (cup_axis_xy + cup_span_margin below). The IoU/dxy/dz triple is now the
        # FALLBACK for fallen (non-vertical) cups only.
        gp('premerge_iou', 0.62)
        gp('premerge_dxy_m', 0.018)
        gp('premerge_dz_m', 0.022)     # blocks telescoped/vertically-stacked cups
        gp('premerge_radius_m', 0.05)  # candidate-pair prefilter (centroid KDTree)
        # Cup-cylinder premerge (vertical cups): merge iff XY box-centers within
        # cup_axis_xy_m AND union z-span <= cup_h*(1+cup_span_margin). The margin
        # MUST sit between one cup (0.095) and cup+min-layer-spacing (~0.134):
        # 0.095 < 0.095*1.25=0.119 < 0.134 ✓ (merges partial views, splits stacks).
        gp('cup_axis_xy_m', 0.05)      # ~cup radius (0.039) + slack
        gp('cup_span_margin', 0.25)
        gp('scan_cup_axis_xy_m', 0.06)     # looser twins for the scan COMPUTE pass
        gp('scan_cup_span_margin', 0.40)
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
        # Color contract: fused cups must carry the SAME color identity the
        # standalone pipeline emitted (verifier/plan_executor parse `c=<color>`).
        gp('cup_colors', ['red', 'orange', 'yellow', 'green', 'blue',
                          'purple', 'white', 'black', 'gray'])
        # Per-view vote weight (hand cam down-weighted until calib verified).
        gp('color_exo_weight', 1.0)
        gp('color_hand_weight', 0.7)
        # Bounded, correctable color vote: gate tiny views, decay stale
        # votes so an early mis-classification self-corrects.
        gp('color_min_points', 64)
        gp('color_vote_decay', 0.9)
        gp('box_standing_ratio', 0.8)
        gp('box_min_elongation', 1.5)

        # View weighting (Phase C; skeleton merges unweighted but reads them).
        gp('w_hand_base', 0.6)
        gp('w_exo_base', 0.4)
        gp('hand_motion_gating', True)
        # LIVE-mode hand-view toggle (Tk checkbox). True (default) = hand cloud
        # contributes to the live fit; False = exclude it (exo only). Scan
        # capture/lock ALWAYS use the hand view regardless of this.
        gp('live_use_hand', True)

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
        # false = HAND frozen + exo LIVE (re-fuse the frozen hand scan with the
        # latest exo cloud every tick → new cups exo sees appear; a picked cup
        # stays from the hand lock). true = freeze BOTH into one static lock.
        gp('scan_lock_exo', False)
        gp('scan_joint_states_topic', '/joint_states')
        gp('scan_joint_names',
           ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6'])
        # Scan waypoints: a FLAT list of 6 joint angles (deg) per waypoint,
        # concatenated. The arm visits each in order; arrival at any is
        # auto-detected by joint match (order-agnostic). Default = a SINGLE
        # waypoint (pos1). Append 6 more values to add pos2, etc.
        gp('scan_waypoints_deg',
           [103.4671, -4.3731, 100.2539, -29.3674, 115.5776, -139.6331])
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

        # ── Measurement source for the LIVE pipeline ─────────────────────────
        # rim   (default): upright cups are measured from the producers'
        #         silhouette observations (mask contour + calibration; depth
        #         only classifies the level). The noisy point-cloud cone-fit
        #         path is bypassed for upright cups; clouds remain in use for
        #         fallen cups (OBB needs 3D points) and for /points display.
        # cloud : legacy full point-cloud path (rollback switch).
        gp('fit_source', 'rim')
        # Drop hand observations captured while the arm moves (image↔FK sync
        # error lands directly in the measurement; the KF coasts instead).
        gp('rim_drop_moving', True)

        # ── Rim/silhouette observation path (Phase 1: PARALLEL instrumentation)
        # Consumes the producers' CupObservationArray (mask-silhouette chamfer
        # fits — see cup_geometry.py), snaps z to the nesting lattice, fuses
        # cameras by inverse covariance, and publishes a SEPARATE debug box
        # set + health JSON + optional CSV for A/B comparison against the
        # cloud path. Never touches /digital_twin/boxes or the KF tracks.
        gp('rim_enabled', True)
        gp('exo_obs_topic', '/digital_twin/cup_obs_exo')
        gp('hand_obs_topic', '/digital_twin/cup_obs_hand')
        gp('rim_boxes_dbg_topic', '/digital_twin/boxes_rim_dbg')
        gp('fusion_health_topic', '/digital_twin/fusion_health')
        gp('table_z_m', 0.0)           # table surface in world (= robot base) z
        # Height ONE nested cup adds to a column (measure: stack 2, subtract).
        gp('nesting_offset_m', 0.020)
        # Pyramid slot lattice (MUST mirror the FastAPI server's
        # PYRAMID_LAYER_HEIGHT). A stacked cup's base sits on k*layer, not
        # k*nest — snapping picks whichever lattice fits the rough z better.
        # A wrong z slides the estimate along the view ray: at the exo
        # elevation (~20°) a 7 mm z error becomes ~19 mm in XY.
        gp('rim_layer_height_m', 0.093)
        gp('rim_obs_max_age_s', 3.0)   # forget observations older than this
        #                                (hand publishes slower than exo —
        #                                 1.0s aged its obs out between ticks)
        gp('rim_cluster_xy_m', 0.05)   # cross-camera same-cup XY gate
        gp('rim_cluster_z_m', 0.04)    # z gate (must be < layer spacing 0.093
        #                                so pyramid layers stay separate)
        gp('rim_min_iou', 0.5)         # consumer-side quality gates
        gp('rim_max_rms_px', 6.0)
        gp('rim_log_dir', '')          # non-empty → append A/B CSV rows there

        # ── Cross-camera extrinsic self-calibration (Phase 3) ───────────────
        # When BOTH cameras see the same cup, the per-camera (x, y) delta is
        # the extrinsic disagreement (observed: static ~30 mm — exo's distant
        # ArUco solvePnP vs the hand FK+hand-eye chain). An EMA of that delta
        # estimates the non-reference camera's world-frame bias; with apply
        # on, its observations are corrected before fusion, so cups only that
        # camera sees inherit the reference frame's accuracy. EMA steps are
        # <= alpha*delta per event (~1.5 mm) — gradual, so tracks follow
        # without KF gate rejections. A rotation error is position-dependent,
        # but across this 0.3 m workspace the variation is ~4 mm — a single
        # translation bias captures the bulk of it.
        gp('rim_bias_apply', False)    # estimation always runs; this gates use
        gp('rim_bias_ref_cam', 'hand')
        gp('rim_bias_alpha', 0.05)     # EMA gain per shared-cup event
        gp('rim_bias_max_m', 0.05)     # clamp |bias|
        gp('rim_bias_delta_max_m', 0.08)  # ignore absurd per-cup deltas

        def P(n):
            return self.get_parameter(n).value

        self.world_frame = str(P('world_frame'))
        self.max_age = float(P('max_age_s'))
        self.merge_dist = float(P('merge_dist_m'))
        self.voxel_m = float(P('fusion_voxel_m'))
        self.max_merge_points = int(P('max_merge_points'))
        self.accum_window = float(P('fusion_accum_window_s'))
        self.use_density = bool(P('use_fusion_density_filter'))
        self.density_radius = float(P('fusion_density_radius_m'))
        self.density_min_nb = int(P('fusion_density_min_neighbors'))
        self.max_cup_footprint = float(P('max_cup_footprint_m'))
        self.premerge_iou = float(P('premerge_iou'))
        self.cup_axis_xy = float(P('cup_axis_xy_m'))
        self.cup_span_margin = float(P('cup_span_margin'))
        self.scan_cup_axis_xy = float(P('scan_cup_axis_xy_m'))
        self.scan_cup_span_margin = float(P('scan_cup_span_margin'))
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
        self._cup_colors = list(P('cup_colors'))
        self._color_w = {'exo': float(P('color_exo_weight')),
                         'hand': float(P('color_hand_weight'))}
        self._color_min_points = int(P('color_min_points'))
        self._color_vote_decay = float(P('color_vote_decay'))
        self.standing_ratio = float(P('box_standing_ratio'))
        self.min_elongation = float(P('box_min_elongation'))
        self.w_hand_base = float(P('w_hand_base'))
        self.w_exo_base = float(P('w_exo_base'))
        self.hand_gating = bool(P('hand_motion_gating'))
        self.live_use_hand = bool(P('live_use_hand'))
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
        self.scan_lock_exo = bool(P('scan_lock_exo'))
        self.scan_joint_states_topic = str(P('scan_joint_states_topic'))
        self.scan_joint_names = list(P('scan_joint_names'))
        # Parse N waypoints from the flat (N*6) list → {1: rad6, 2: rad6, ...}.
        _wps = np.asarray(P('scan_waypoints_deg'), dtype=float)
        if _wps.size == 0 or _wps.size % 6 != 0:
            self.get_logger().error(
                f'scan_waypoints_deg length {_wps.size} is not a positive '
                f'multiple of 6 — using one zero waypoint')
            _wps = np.zeros(6)
        _wps = _wps.reshape(-1, 6)
        self._wp_rad = {i + 1: np.deg2rad(_wps[i]) for i in range(len(_wps))}
        self._wp_keys = tuple(self._wp_rad)     # (1,) single | (1,2,...) multi
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
        if not (0.005 <= self.scan_tol <= 0.1):
            self.get_logger().warn(
                f'scan_arrival_tol_rad={self.scan_tol} outside [0.005, 0.1]')
        if len(self._wp_keys) >= 2:     # only multi-waypoint can mis-attribute
            seps = [float(np.max(np.abs(self._wp_rad[a] - self._wp_rad[b])))
                    for a in self._wp_keys for b in self._wp_keys if a < b]
            if min(seps) < 2.0 * self.scan_tol:
                self.get_logger().warn(
                    f'two scan waypoints are within 2*tol (min inf-norm '
                    f'{min(seps):.3f} rad) — captures may be mis-attributed')
        self.get_logger().info(
            f'scan: {len(self._wp_keys)} waypoint(s) configured '
            f'(pos{list(self._wp_keys)})')
        # SINGLE-THREADED executor (rclpy.spin): the joint_states callback, the
        # 2 cloud callbacks, the timer, ~/clear_scan and the param callback are
        # all serialized on ONE thread → no locks needed for the state below.
        self._mode = 'OFF'             # OFF(live) | ACTIVE(capturing) | PAUSED
        self._scan_state = 'IDLE'      # IDLE | WAIT | CAPTURE
        self._cur_wp = None
        self._t_arrive = None
        self._t_cap = None
        self._captured = {k: False for k in self._wp_keys}
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
        # Rolling per-camera cloud history for temporal frame accumulation (A).
        self._cloud_hist: dict[str, list] = {'exo': [], 'hand': []}
        # Last-processed cloud stamp per camera — a KF update only fires when a
        # camera's stamp advances, so a stale cloud is not counted repeatedly
        # (which would make the filter over-confident and the box flicker).
        self._proc_stamp: dict[str, float] = {'exo': -1.0, 'hand': -1.0}
        self._tracks: dict[int, dict] = {}
        self._stacked_ids: set = set()
        self._next_gid = 1
        self._last_ids: set[int] = set()
        self._last_points_xyz = None     # last non-empty /points union (coast)
        self._last_points_rgb = None

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
        self.cups_on_table_pub = self.create_publisher(
            String, str(P('cups_on_table_topic')), latched)
        self.create_subscription(
            Int32MultiArray, str(P('stack_track_ids_topic')),
            self._on_stack_ids, 10)
        self.points_pub = self.create_publisher(
            PointCloud2, str(P('points_topic')), 5)
        self.dbg_det_pub = self.create_publisher(
            MarkerArray, str(P('dbg_detections_topic')), latched)
        self.dbg_premerge_pub = self.create_publisher(
            MarkerArray, str(P('dbg_premerge_topic')), latched)
        self.dbg_fit1_pub = self.create_publisher(
            MarkerArray, str(P('dbg_fit1_topic')), latched)
        # ── Rim observation path: state + subs/pubs ─────────────────────────
        self.fit_source = str(P('fit_source')).strip().lower()
        self.rim_drop_moving = bool(P('rim_drop_moving'))
        self._rim_fresh = False        # new obs since last live tick
        self.rim_enabled = bool(P('rim_enabled'))
        self.table_z = float(P('table_z_m'))
        self.nest_off = float(P('nesting_offset_m'))
        self.rim_layer_h = float(P('rim_layer_height_m'))
        self.rim_obs_max_age = float(P('rim_obs_max_age_s'))
        self.rim_cluster_xy = float(P('rim_cluster_xy_m'))
        self.rim_cluster_z = float(P('rim_cluster_z_m'))
        self.rim_min_iou = float(P('rim_min_iou'))
        self.rim_max_rms = float(P('rim_max_rms_px'))
        self.rim_log_dir = str(P('rim_log_dir'))
        self._rim_latest: dict[tuple, tuple] = {}   # (cam, iid) → (obs, t_s)
        self.rim_bias_apply = bool(P('rim_bias_apply'))
        self.rim_bias_ref = str(P('rim_bias_ref_cam'))
        self.rim_bias_alpha = float(P('rim_bias_alpha'))
        self.rim_bias_max = float(P('rim_bias_max_m'))
        self.rim_bias_delta_max = float(P('rim_bias_delta_max_m'))
        self._cam_bias: dict[str, np.ndarray] = {}  # cam → world-XY bias (m)
        self._cam_bias_n: dict[str, int] = {}       # cam → update count
        self._rim_csv = None                        # lazy-opened file handle
        self.rim_dbg_pub = None
        self.health_pub = None
        if self.rim_enabled:
            self.create_subscription(
                CupObservationArray, str(P('exo_obs_topic')),
                lambda m: self._on_cup_obs(m, 'exo'), 10)
            self.create_subscription(
                CupObservationArray, str(P('hand_obs_topic')),
                lambda m: self._on_cup_obs(m, 'hand'), 10)
            self.rim_dbg_pub = self.create_publisher(
                MarkerArray, str(P('rim_boxes_dbg_topic')), latched)
            self.health_pub = self.create_publisher(
                String, str(P('fusion_health_topic')), 10)
            self.get_logger().info(
                f"rim path ON: obs={P('exo_obs_topic')},{P('hand_obs_topic')} "
                f"→ {P('rim_boxes_dbg_topic')} + {P('fusion_health_topic')}"
                + (f' + CSV {self.rim_log_dir}' if self.rim_log_dir else ''))

        # Debug markers age out via lifetime instead of a per-event DELETEALL,
        # so a 1-2 event detection dropout doesn't blink the overlay in RViz.
        # 4 measurement events of slack; refreshed well before expiry.
        _life = max(0.3, 4.0 * float(P('fusion_period_s')))
        self._dbg_lifetime = DurationMsg(
            sec=int(_life), nanosec=int((_life % 1.0) * 1e9))
        self._dbg_enabled_prev = {}     # pub → last enabled state (for wipe)

        self.create_timer(float(P('fusion_period_s')), self._tick)
        # Live-tunable thresholds — `ros2 param set /cup_fusion_node <p> <v>`
        # takes effect immediately (no relaunch needed while tuning).
        self._tunable = {
            'max_age_s': ('max_age', float),
            'fusion_accum_window_s': ('accum_window', float),
            'use_fusion_density_filter': ('use_density', bool),
            'fusion_density_radius_m': ('density_radius', float),
            'fusion_density_min_neighbors': ('density_min_nb', int),
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
            'cup_axis_xy_m': ('cup_axis_xy', float),
            'cup_span_margin': ('cup_span_margin', float),
            'scan_cup_axis_xy_m': ('scan_cup_axis_xy', float),
            'scan_cup_span_margin': ('scan_cup_span_margin', float),
            # scan & lock live-tunables (scan_lock_active drives the mode)
            'scan_lock_active': ('scan_lock_active', bool),
            'scan_lock_exo': ('scan_lock_exo', bool),
            'live_use_hand': ('live_use_hand', bool),
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
            # rim path live-tunables
            'rim_enabled': ('rim_enabled', bool),
            'table_z_m': ('table_z', float),
            'nesting_offset_m': ('nest_off', float),
            'rim_layer_height_m': ('rim_layer_h', float),
            'rim_obs_max_age_s': ('rim_obs_max_age', float),
            'rim_cluster_xy_m': ('rim_cluster_xy', float),
            'rim_cluster_z_m': ('rim_cluster_z', float),
            'rim_min_iou': ('rim_min_iou', float),
            'rim_max_rms_px': ('rim_max_rms', float),
            'rim_log_dir': ('rim_log_dir', str),
            'fit_source': ('fit_source', str),
            'rim_drop_moving': ('rim_drop_moving', bool),
            'rim_bias_apply': ('rim_bias_apply', bool),
            'rim_bias_alpha': ('rim_bias_alpha', float),
            'rim_bias_max_m': ('rim_bias_max', float),
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
        now = self.get_clock().now()
        self._latest[cam] = (msg, now)
        # Append to the rolling history and prune frames older than the
        # accumulation window (a little slack so a brief rate dip keeps frames).
        hist = self._cloud_hist[cam]
        hist.append((now, msg))
        keep_s = max(self.accum_window, self.max_age)
        while hist and (now - hist[0][0]).nanoseconds * 1e-9 > keep_s:
            hist.pop(0)

    def _gather(self):
        """Collect objects for THIS measurement event. Returns None (→ coast, no
        KF update) unless at least one camera produced a NEW cloud since the last
        event; when fresh, ACCUMULATES each active camera's clouds over the last
        fusion_accum_window_s (temporal denoise) so a cup's fit averages out the
        raw per-frame depth noise."""
        now = self.get_clock().now()
        active = []
        fresh = False
        for cam in ('exo', 'hand'):
            if cam == 'hand' and not self.live_use_hand:
                continue                 # live: exclude hand view (Tk toggle)
            msg, t = self._latest[cam]
            if msg is None or (now - t).nanoseconds * 1e-9 > self.max_age:
                continue
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            active.append((cam, stamp))
            if stamp > self._proc_stamp[cam]:
                fresh = True
        if not fresh:
            return None      # not a fresh event → caller skips, markers persist
        objs = []
        for cam, stamp in active:
            self._proc_stamp[cam] = stamp
            objs.extend(self._accumulated_objs(cam, now))
        return objs

    def _accumulated_objs(self, cam, now):
        """Union one camera's detection-objs over the last `accum_window` of
        frames (temporal frame accumulation, A). _premerge then clusters the SAME
        cup across frames → _merge unions their points → a denser cloud whose fit
        averages out per-frame depth noise. FALLBACK: if no frame falls inside
        the window (a slow producer whose cloud INTERVAL > accum_window — which
        made the slower camera vanish every event and flicker /points), use the
        latest cloud (the caller already gated it by max_age)."""
        hist = self._cloud_hist[cam]
        if self.accum_window > 0.0 and hist:
            objs = []
            for recv_t, msg in hist:
                if (now - recv_t).nanoseconds * 1e-9 <= self.accum_window:
                    objs.extend(self._build_cam_objs(cam, msg))
            if objs:
                return objs
        msg, _ = self._latest[cam]      # fallback: latest cloud (within max_age)
        return self._build_cam_objs(cam, msg) if msg is not None else []

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
        # (B) Spatial-density filter: drop isolated noise islands (esp. exo
        # flying-pixels) before the fit. Skip if it would gut the cluster.
        if self.use_density and out.shape[0] > self.density_min_nb:
            keep = _filter_spatial_density(
                out, self.density_radius, self.density_min_nb)
            if int(keep.sum()) >= 16:
                out = out[keep]
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
        ONLY detections that are the SAME physical cup. Returns groups of obj
        indices = one group per physical cup.

        Primary metric = CUP-CYLINDER containment (a standing cup is a vertical
        cylinder): two upright detections merge iff their robust XY box-centers
        (axis estimate) are within one cup radius AND the UNION of their robust
        z-extents fits inside one cup height·(1+margin). This merges a hand
        top-rim disc with an exo side slab (they share the axis; union-z ≈ one
        cup) where 3D AABB-IoU failed (thin disc vs tall slab barely overlap),
        while a vertically-STACKED pair (union-z > one cup) stays separate.
        Pairs where EITHER detection is not an upright-cup (YOLO class, e.g.
        fallen-cup) fall back to the legacy AABB-IoU gate."""
        # Cluster by XY (the vertical column), NOT 3D centroid: a cup's hand
        # top-disc and exo side-slab share XY but differ in centroid-Z, so 3D
        # clustering would split them before the pair-test ever runs. Stacked
        # cups share XY too → they land in one cluster and the cup-cylinder
        # pair-test below separates them by z-span.
        clusters = _cluster_indices(
            [np.array([o['xyc'][0], o['xyc'][1], 0.0]) for o in objs],
            self.merge_dist)
        groups = []
        z_ceil = self.cup_h * (1.0 + self.cup_span_margin)
        for cl in clusters:
            if len(cl) == 1:
                groups.append(cl)
                continue
            edges = []
            for a in range(len(cl)):
                for b in range(a + 1, len(cl)):
                    oi, oj = objs[cl[a]], objs[cl[b]]
                    # Upright (by YOLO class) → vertical-cylinder metric; any
                    # non-upright (fallen-cup) in the pair → legacy IoU gate.
                    up = (oi['class_name'] in self.cup_class_names
                          and oj['class_name'] in self.cup_class_names)
                    if up:
                        dxy = float(np.hypot(oi['xyc'][0] - oj['xyc'][0],
                                             oi['xyc'][1] - oj['xyc'][1]))
                        z_span = (max(oi['zhi'], oj['zhi'])
                                  - min(oi['zlo'], oj['zlo']))
                        same = dxy <= self.cup_axis_xy and z_span <= z_ceil
                    else:               # fallen cup → legacy AABB-IoU gate
                        dc = oi['centroid'] - oj['centroid']
                        same = (abs(dc[2]) <= self.premerge_dz
                                and float(np.hypot(dc[0], dc[1]))
                                <= self.premerge_dxy
                                and _aabb_iou(oi['aabb'], oj['aabb'])
                                >= self.premerge_iou)
                    if same:
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
            return (False, {k: False for k in self._wp_keys},
                    {k: False for k in self._wp_keys}, True)
        settled = self._cur_vmax < self.scan_settle_vel
        ne, nl = {}, {}
        for k in self._wp_keys:
            d = float(np.max(np.abs(self._cur_q - self._wp_rad[k])))
            ne[k] = d <= self.scan_tol
            nl[k] = d <= 1.5 * self.scan_tol    # hysteresis band for "leaving"
        return settled, ne, nl, False

    def _scan_fsm(self, now) -> None:
        settled, near, near_leave, unknown = self._scan_snapshot(now)

        # Leave-latch: re-arm a waypoint once the arm has clearly left it, so a
        # 2nd scan pass re-captures (and accumulates) at each waypoint. Runs
        # every tick, including PAUSED.
        for k in self._wp_keys:
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
            cand = [k for k in self._wp_keys
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
                # pass is done (all waypoints captured). Shows how far the arm
                # is from EACH waypoint so the motion + arrival detection can be
                # verified live. Goes silent after the scan finishes.
                errs = ', '.join(
                    f'pos{k} '
                    f'{np.degrees(np.max(np.abs(self._cur_q - self._wp_rad[k]))):.1f}°'
                    for k in self._wp_keys)
                self.get_logger().info(
                    f'scan: waiting for a waypoint — {errs} '
                    f'(tol {tol_deg:.1f}°), settled={settled}',
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
                if (self._scan_visited >= set(self._wp_keys)
                        and not self._scan_done):
                    self._scan_done = True
                    self.get_logger().info(
                        f'scan: ✔ pass complete — all {len(self._wp_keys)} '
                        f'waypoint(s) captured, lock published; diagnostic '
                        f'logging stops (clear or re-activate to scan again)')

    def _cup_geom_fields(self, xyz):
        """Premerge geometry from ONE detection's points (single robust AABB):
        the box, the robust XY box-center (a standing-cup AXIS estimate), and the
        z-extent. Whether the cup-cylinder metric applies (vs the legacy IoU
        gate) is decided in _premerge from the YOLO class — upright-cup vs
        fallen-cup — NOT re-derived from geometry."""
        lo, hi = _aabb_robust(xyz)
        return {
            'aabb': (lo, hi),
            'xyc': np.array([0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])]),
            'zlo': float(lo[2]), 'zhi': float(hi[2]),
        }

    def _build_cam_objs(self, cam, msg):
        """Detection-objs (world-frame already) for one camera's cloud message
        — same shape _gather builds, incl. the premerge geom fields."""
        out = []
        for o in msg.objects:
            if cam == 'hand' and o.moving and self.hand_gating:
                continue
            xyz, rgb = _pc2_xyzrgb(o.points)
            if xyz.shape[0] < 32:
                continue
            out.append({
                'cam': cam, 'xyz': xyz, 'rgb': rgb,
                'centroid': np.array([o.centroid.x, o.centroid.y,
                                      o.centroid.z], dtype=np.float64),
                **self._cup_geom_fields(xyz),
                'score': float(o.score), 'class_name': o.class_name,
                'moving': bool(o.moving),
                'w': self.w_hand_base if cam == 'hand' else self.w_exo_base,
            })
        return out

    def _latest_cam_objs(self, cam):
        """Accumulated objs of `cam` over fusion_accum_window_s (within max_age)
        — overlays LIVE, temporally-denoised exo on the frozen hand lock each
        tick, so the exo re-fuse doesn't jitter from single-frame noise."""
        msg, t = self._latest[cam]
        now = self.get_clock().now()
        if msg is None or (now - t).nanoseconds * 1e-9 > self.max_age:
            return []
        return self._accumulated_objs(cam, now)

    def _capture_into_accum(self) -> None:
        """Append fresh per-camera detection-objs into the accumulation buffer
        (dedup by per-camera stamp). When scan_lock_exo is False, accumulate the
        HAND camera ONLY — exo stays LIVE and is re-fused each tick in the
        locked branch; freezing it here would double-count it."""
        now = self.get_clock().now()
        cams = ('exo', 'hand') if self.scan_lock_exo else ('hand',)
        for cam in cams:
            msg, t = self._latest[cam]
            if msg is None or (now - t).nanoseconds * 1e-9 > self.max_age:
                continue
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            if stamp <= self._scan_proc_stamp[cam]:
                continue                        # same cloud already captured
            self._scan_proc_stamp[cam] = stamp
            self._accum.extend(self._build_cam_objs(cam, msg))

    @contextmanager
    def _scan_params(self):
        """Temporarily swap the live merge/fit gates to the looser scan_* values
        for the COMPUTE pass (a multi-view union of one cup is more spread than a
        single view). Single-threaded → safe to mutate instance attrs."""
        attrs = ('merge_dist', 'premerge_dxy', 'premerge_dz', 'premerge_iou',
                 'postmerge_dxy', 'postmerge_dz', 'max_cup_footprint',
                 'cup_axis_xy', 'cup_span_margin')
        saved = {a: getattr(self, a) for a in attrs}
        self.merge_dist = self.scan_merge_dist
        self.premerge_dxy = self.scan_premerge_dxy
        self.premerge_dz = self.scan_premerge_dz
        self.premerge_iou = self.scan_premerge_iou
        self.postmerge_dxy = self.scan_postmerge_dxy
        self.postmerge_dz = self.scan_postmerge_dz
        self.max_cup_footprint = self.scan_max_cup_footprint
        self.cup_axis_xy = self.scan_cup_axis_xy
        self.cup_span_margin = self.scan_cup_span_margin
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
                'centroid': xyz.mean(axis=0),
                **self._cup_geom_fields(xyz),
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
        self._captured = {k: False for k in self._wp_keys}
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
        self._captured = {k: False for k in self._wp_keys}
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
        self._last_points_xyz = None
        self._last_points_rgb = None
        self._proc_stamp = {'exo': -1.0, 'hand': -1.0}
        self.get_logger().info('scan-lock CLEARED — back to live detection')

    # ================= tick dispatcher + live pipeline ==================
    # ── Rim/silhouette observation path (Phase 1: parallel A/B) ───────────
    def _on_cup_obs(self, msg: CupObservationArray, cam: str) -> None:
        now_s = self.get_clock().now().nanoseconds * 1e-9
        for ob in msg.observations:
            self._rim_latest[(cam, int(ob.instance_id))] = (ob, now_s)
        if msg.observations:
            self._rim_fresh = True

    def _rim_collect(self) -> list:
        """Quality-gated, age-pruned snapshot of the cached observations —
        shared by the live rim estimator and the debug tick."""
        now_s = self.get_clock().now().nanoseconds * 1e-9
        for k in [k for k, (_, t) in self._rim_latest.items()
                  if now_s - t > self.rim_obs_max_age]:
            del self._rim_latest[k]
        return [ob for ob, _ in self._rim_latest.values()
                if ob.mask_iou >= self.rim_min_iou
                and ob.chamfer_rms_px <= self.rim_max_rms
                and not (self.rim_drop_moving and ob.moving)]

    def _rim_tick(self, stamp) -> None:
        """Fuse the cached silhouette observations into per-cup estimates and
        publish debug markers + health JSON (+ optional CSV). Read-only with
        respect to the cloud pipeline."""
        now_s = self.get_clock().now().nanoseconds * 1e-9
        ests = self._rim_estimates(self._rim_collect())
        self._publish_rim_markers(ests, stamp)
        self._publish_fusion_health(ests, stamp)
        if self.rim_log_dir:
            self._rim_log(ests, now_s)

    def _rim_estimates(self, obs: list) -> list[dict]:
        """Cluster observations across cameras (XY), snap each cluster's base
        to the nesting lattice, slide each camera's (x, y) along its view ray
        to the snapped level, then fuse by inverse covariance. The per-camera
        residuals are the cross-view consistency signal (extrinsic health)."""
        if not obs:
            return []
        # 3D ellipsoidal clustering: XY and Z gated separately. XY-only
        # collapsed a pyramid — cups on adjacent layers sit at HALF the cup
        # spacing in XY (0.039 < 0.05 gate) and only differ in z.
        xy_g = max(self.rim_cluster_xy, 1e-6)
        z_g = max(self.rim_cluster_z, 1e-6)
        scaled = [np.array([o.x0 / xy_g, o.y0 / xy_g,
                            o.z_top_rough_m / z_g]) for o in obs]
        ests = []
        for grp in _cluster_indices(scaled, 1.0):
            members = [obs[i] for i in grp]
            by_cam: dict[str, object] = {}
            for o in members:           # best (highest-IoU) obs per camera
                cur = by_cam.get(o.camera)
                if cur is None or o.mask_iou > cur.mask_iou:
                    by_cam[o.camera] = o
            z_top = float(np.median(
                [o.z_top_rough_m for o in by_cam.values()]))
            # Snap to whichever lattice (nested column vs pyramid slot)
            # explains the rough z better.
            k, z_base, lvl_err = snap_level(
                z_top, table_z=self.table_z, cup_h=self.cup_h,
                nest_off=self.nest_off)
            kl, zl, el = snap_level(
                z_top, table_z=self.table_z, cup_h=self.cup_h,
                nest_off=self.rim_layer_h)
            if abs(el) < abs(lvl_err):
                k, z_base, lvl_err = kl, zl, el
            slids: dict[str, tuple] = {}    # cam → (raw xy, cov, obs)
            for o in by_cam.values():
                d = np.array([o.ray_dir.x, o.ray_dir.y, o.ray_dir.z])
                slid = slide_xy_to_z(o.x0, o.y0, o.z_base0, d, z_base)
                if slid is None:
                    continue
                org = np.array([o.ray_origin.x, o.ray_origin.y,
                                o.ray_origin.z])
                rng = float(np.linalg.norm(
                    np.array([slid[0], slid[1], z_base]) - org))
                cov = xy_cov_from_px(
                    o.sigma_px, rng, float(o.focal_px) or 600.0, d)
                slids[o.camera] = (np.asarray(slid, dtype=np.float64),
                                   cov, o)
            # Extrinsic bias estimation from RAW (uncorrected) deltas — a
            # shared cup is a calibration target. Update before applying so
            # the EMA tracks the true disagreement, not the residual of its
            # own correction.
            if self.rim_bias_ref in slids:
                ref_xy = slids[self.rim_bias_ref][0]
                for cam, (xy, _, o) in slids.items():
                    if cam == self.rim_bias_ref or o.moving:
                        continue
                    delta = xy - ref_xy
                    if float(np.linalg.norm(delta)) > self.rim_bias_delta_max:
                        continue
                    b = self._cam_bias.get(cam, np.zeros(2))
                    b = (1.0 - self.rim_bias_alpha) * b \
                        + self.rim_bias_alpha * delta
                    nb = float(np.linalg.norm(b))
                    if nb > self.rim_bias_max:
                        b *= self.rim_bias_max / nb
                    self._cam_bias[cam] = b
                    self._cam_bias_n[cam] = self._cam_bias_n.get(cam, 0) + 1
            meas, used = [], []
            for cam, (xy, cov, o) in slids.items():
                if self.rim_bias_apply and cam != self.rim_bias_ref:
                    xy = xy - self._cam_bias.get(cam, np.zeros(2))
                meas.append((xy, cov))
                used.append(o)
            fused = fuse_xy(meas)
            if fused is None:
                continue
            xy_f, cov_f, resid = fused
            ests.append({
                'x': float(xy_f[0]), 'y': float(xy_f[1]),
                'z_base': float(z_base), 'level': int(k),
                'level_err_mm': round(1e3 * float(lvl_err), 1),
                'sigma_mm': round(
                    1e3 * float(np.sqrt(np.max(np.linalg.eigvalsh(cov_f)))), 1),
                'cov': cov_f,
                'cams': {o.camera: {
                    'resid_mm': round(1e3 * float(np.linalg.norm(r)), 1),
                    'iou': round(float(o.mask_iou), 3),
                    'rms_px': round(float(o.chamfer_rms_px), 2),
                    'score': float(o.score),
                    'color': str(o.color),
                    'moving': bool(o.moving)}
                    for o, r in zip(used, resid)},
            })
        return ests

    def _rim_final_fits(self) -> list[dict]:
        """LIVE measurement set from the silhouette observations, shaped like
        the cloud path's `final_fits` so STAGE-3/4/5 (association, KF, color,
        markers, cups_on_table) run unchanged. Per-cup measurement noise comes
        from the fused covariance instead of the fixed kf_meas_* params."""
        fits = []
        for e in self._rim_estimates(self._rim_collect()):
            center = np.array([e['x'], e['y'],
                               e['z_base'] + 0.5 * self.cup_h])
            frustum = _cup_frustum_geometry(
                e['x'], e['y'], top_d=self.cup_top_d, bot_d=self.cup_bot_d,
                height=self.cup_h, floor_z=e['z_base'], n_seg=self.cup_n_seg)
            members = [{
                'cam': cam,
                'class_name': 'upright-cup',
                'score': c['score'],
                'rgb': np.zeros(0, np.float32),   # color via color_name
                'color_name': c['color'] or None,
            } for cam, c in e['cams'].items()]
            cov = np.asarray(e['cov'])
            r_diag = np.array([
                max(cov[0, 0], 1e-6), max(cov[1, 1], 1e-6),
                # snapped z: trust it well below the level spacing
                (0.25 * max(self.nest_off, 1e-3)) ** 2])
            fits.append({
                'center': center, 'R': np.eye(3),
                'size': np.array([self.cup_bot_d, self.cup_bot_d,
                                  self.cup_h]),
                'kind': 'cup', 'frustum': frustum,
                'residual': 1e-3 * e['sigma_mm'],
                'members': members, 'r_diag': r_diag,
            })
        return fits

    def _publish_rim_markers(self, ests: list[dict], stamp) -> None:
        arr = MarkerArray()
        for i, e in enumerate(ests):
            fr = _cup_frustum_geometry(
                e['x'], e['y'], top_d=self.cup_top_d, bot_d=self.cup_bot_d,
                height=self.cup_h, floor_z=e['z_base'], n_seg=self.cup_n_seg)
            m = Marker()
            m.header = Header(stamp=stamp, frame_id=self.world_frame)
            m.ns = 'rim_dbg'
            m.id = i
            m.type = Marker.LINE_LIST
            m.action = Marker.ADD
            m.scale = Vector3(x=0.002, y=0.0, z=0.0)
            m.color = ColorRGBA(r=0.0, g=0.95, b=0.95, a=0.9)
            m.lifetime = self._dbg_lifetime
            pts = []
            for loop in (fr['top_loop'], fr['bot_loop']):
                for a, b in zip(loop[:-1], loop[1:]):
                    pts.extend((a, b))
            for a, b in fr['generatrix']:
                pts.extend((a, b))
            m.points = [MsgPoint(x=float(p[0]), y=float(p[1]), z=float(p[2]))
                        for p in pts]
            arr.markers.append(m)

            t = Marker()
            t.header = m.header
            t.ns = 'rim_dbg_txt'
            t.id = i
            t.type = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose.position = MsgPoint(
                x=e['x'], y=e['y'], z=e['z_base'] + self.cup_h + 0.04)
            t.scale = Vector3(x=0.0, y=0.0, z=0.02)
            t.color = ColorRGBA(r=0.0, g=0.95, b=0.95, a=0.9)
            t.lifetime = self._dbg_lifetime
            cams = '+'.join(sorted(e['cams']))
            t.text = (f"R{i} L{e['level']} ±{e['sigma_mm']:.0f}mm "
                      f"[{cams}]")
            arr.markers.append(t)
        if arr.markers:
            self.rim_dbg_pub.publish(arr)

    def _publish_fusion_health(self, ests: list[dict], stamp) -> None:
        # 'cov' is a numpy array (KF consumption) — not JSON-serializable.
        cups = [{k: v for k, v in e.items() if k != 'cov'} for e in ests]
        self.health_pub.publish(String(data=json.dumps({
            't': float(stamp.sec) + float(stamp.nanosec) * 1e-9,
            'n_obs_cached': len(self._rim_latest),
            'extrinsic_bias_mm': {
                cam: {'xy': [round(1e3 * float(v), 1) for v in b],
                      'n': self._cam_bias_n.get(cam, 0),
                      'applied': bool(self.rim_bias_apply)}
                for cam, b in self._cam_bias.items()},
            'cups': cups,
        })))

    def _rim_log(self, ests: list[dict], now_s: float) -> None:
        """A/B CSV: one `rim` row per rim estimate and one `cloud` row per
        live KF track, same timestamp — offline analysis aligns them by t."""
        if getattr(self, '_rim_csv_dir', None) != self.rim_log_dir:
            # rim_log_dir changed live (ros2 param set) → reopen in new dir.
            if self._rim_csv is not None:
                self._rim_csv.close()
                self._rim_csv = None
            self._rim_csv_dir = self.rim_log_dir
        if self._rim_csv is None:
            import os
            from pathlib import Path
            d = Path(self.rim_log_dir).expanduser()
            d.mkdir(parents=True, exist_ok=True)
            path = d / f'rim_ab_{os.getpid()}.csv'
            self._rim_csv = path.open('a', buffering=1)
            if path.stat().st_size == 0:
                self._rim_csv.write(
                    't,src,idx,x,y,z,ncams,resid_max_mm,sigma_mm,level\n')
            self.get_logger().info(f'rim A/B CSV → {path}')
        for i, e in enumerate(ests):
            rmax = max((c['resid_mm'] for c in e['cams'].values()),
                       default=0.0)
            self._rim_csv.write(
                f"{now_s:.3f},rim,{i},{e['x']:.4f},{e['y']:.4f},"
                f"{e['z_base']:.4f},{len(e['cams'])},{rmax:.1f},"
                f"{e['sigma_mm']:.1f},{e['level']}\n")
        for gid, tr in self._tracks.items():
            kf = tr.get('kf')
            if kf is None:
                continue
            self._rim_csv.write(
                f"{now_s:.3f},cloud,{gid},{kf.x[0]:.4f},{kf.x[1]:.4f},"
                f"{kf.x[2]:.4f},0,0.0,{1e3 * kf.position_std():.1f},-1\n")

    def _tick(self) -> None:
        """Dispatcher: live KF pipeline (OFF, or ACTIVE before the first lock)
        vs scan-lock (advance FSM, bypass live KF once a lock exists). All scan
        FSM transitions run HERE, never in the joint_states callback."""
        stamp = self.get_clock().now().to_msg()
        now = self.get_clock().now()

        if self._pending_clear:                 # priority 1: deferred clear
            self._do_clear(stamp)
            self._pending_clear = False

        # Rim/silhouette estimator — strictly parallel instrumentation; an
        # exception here must never take down the cloud pipeline.
        if self.rim_enabled and self.rim_dbg_pub is not None:
            try:
                self._rim_tick(stamp)
            except Exception as e:
                self.get_logger().warn(
                    f'rim estimator error: {e}', throttle_duration_sec=5.0)

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

        if not self.scan_lock_exo:
            # HAND frozen + exo LIVE: re-fuse the frozen hand scan with the
            # latest exo cloud every tick, so cups exo newly sees appear and a
            # picked cup stays from the hand lock.
            return self._publish_hand_lock_live_exo(stamp)

        # BOTH frozen: republish the cached union on the VOLATILE /points each
        # tick (a late RViz subscriber can't get a latched volatile cloud). Lock
        # boxes are latched from _compute_locked, so no per-tick marker publish.
        # Republish even a 0-point cloud so a stale live frame can't linger.
        if self._locked_points_xyz is not None:
            self.points_pub.publish(_make_pointcloud2(
                Header(stamp=stamp, frame_id=self.world_frame),
                self._locked_points_xyz, self._locked_points_rgb))

    def _publish_hand_lock_live_exo(self, stamp) -> None:
        """Hand-only lock: fuse the frozen (consolidated) hand accumulation with
        the LATEST exo objs, re-fit, and publish boxes + /points every tick. The
        dense frozen hand points dominate each shared cup's fit (stable), while
        exo adds live coverage and any cup it newly sees."""
        combined = list(self._accum) + self._latest_cam_objs('exo')
        with self._scan_params():
            fits = self._run_fit_pipeline(combined)
        tracks, ids = {}, set()
        for i, f in enumerate(fits, start=1):
            cams = {m['cam'] for m in f['members']}
            tracks[i] = self._render_state(
                i, f['center'], f['R'], f['size'], f['kind'],
                f['frustum'], f['residual'], cams)
            ids.add(i)
        self._locked_tracks = tracks
        self._publish_lock_markers(stamp, ids)
        if combined:
            xyz = np.vstack([o['xyz'] for o in combined]).astype(np.float32)
            rgb = np.concatenate([o['rgb'] for o in combined])
        else:
            xyz = np.zeros((0, 3), np.float32)
            rgb = np.zeros((0,), np.float32)
        self.points_pub.publish(_make_pointcloud2(
            Header(stamp=stamp, frame_id=self.world_frame), xyz, rgb))

    def _tick_live(self) -> None:
        rim_mode = self.fit_source == 'rim'
        objs = self._gather()
        if rim_mode:
            # Measurement event = fresh silhouette obs OR fresh clouds
            # (clouds still serve fallen cups + the /points display).
            if objs is None and not self._rim_fresh:
                return
            self._rim_fresh = False
            objs = objs or []
        elif objs is None:
            # Not a fresh measurement event → don't touch the markers; the last
            # published set persists in RViz. (No flicker between measurements.)
            return
        # objs already carry aabb + cup-geom fields (_build_cam_objs).

        # Predict once per measurement event (Q is tuned for this cadence).
        for tr in self._tracks.values():
            tr['kf'].predict()
            tr['settled'] = tr['kf'].position_std() <= self.kf_settled_std

        # STAGE-1: merge duplicate detections of the same cup → fit each group.
        # rim mode: upright cups are measured from silhouette observations
        # (no cone fit on noisy depth); the cloud path below only handles
        # what rim cannot — fallen cups, whose OBB needs 3D points.
        cloud_objs = ([o for o in objs
                       if o['class_name'] not in self.cup_class_names]
                      if rim_mode else objs)
        premerge_groups = self._premerge(cloud_objs)
        fits = []
        for grp in premerge_groups:
            members = [cloud_objs[i] for i in grp]
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

        if rim_mode:
            final_fits = self._rim_final_fits() + final_fits

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
                tr['kf'].update(f['center'],
                                f.get('r_diag', self.r_diag), self.kf_gate)
                tr['hits'] += 1
            tr['cams'] = {m['cam'] for m in f['members']}
            tr['miss'] = 0
            # ── Color: per-view HSV vote with decay (recency-weighted, ──
            # bounded ~= weight/(1-decay)). Each measurement event decays prior
            # votes then adds a FIXED per-view weight (not point-count), so an
            # early mis-classification self-corrects; tiny views are gated by
            # color_min_points. argmax of known colors = track color.
            votes = tr.setdefault('color_votes', {})
            for k in votes:
                votes[k] *= self._color_vote_decay
            by_cam: dict = {}
            named: dict = {}      # rim members carry the color by NAME
            for m in f['members']:
                if m.get('color_name'):
                    named.setdefault(m['cam'], m['color_name'])
                else:
                    by_cam.setdefault(m['cam'], []).append(m['rgb'])
            for cam, col in named.items():
                if col in self._cup_colors:
                    votes[col] = votes.get(col, 0.0) \
                        + self._color_w.get(cam, 1.0)
            for cam, rgbs in by_cam.items():
                packed = np.concatenate(rgbs) if rgbs else None
                if packed is None or len(packed) < self._color_min_points:
                    continue
                col = _color_from_packed_rgb(packed, self._cup_colors)
                if col is None:
                    continue
                votes[col] = votes.get(col, 0.0) + self._color_w.get(cam, 1.0)
            tr['color'] = max(votes, key=votes.get) if votes else 'unknown'
            cls = next((m['class_name'] for m in f['members']
                        if m['class_name'] in self.cup_class_names),
                       f['members'][0]['class_name'] if f['members'] else 'cup')
            score = max((m['score'] for m in f['members']), default=0.0)
            tr['cls'] = cls
            tr['last_state'] = self._render_state(
                gid, tr['kf'].x.copy(), f['R'], f['size'], f['kind'],
                f['frustum'], f['residual'], tr['cams'],
                tr['color'], cls, score)
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
        self._publish_cups_on_table()
        # COAST /points like the boxes: a producer emits a fresh-but-EMPTY cloud
        # on any YOLO-dropout frame; blanking /points there made it flicker while
        # the (min_hits/keepalive-coasted) boxes stayed. So republish the last
        # non-empty union while tracks are still alive, and only clear once they
        # are fully evicted (_last_ids empty — same lifetime as the boxes).
        if objs:
            xyz = np.vstack([o['xyz'] for o in objs]).astype(np.float32)
            rgb = np.concatenate([o['rgb'] for o in objs])
            self._last_points_xyz, self._last_points_rgb = xyz, rgb
        elif self._last_ids and self._last_points_xyz is not None:
            xyz, rgb = self._last_points_xyz, self._last_points_rgb   # coast
        else:
            xyz = np.zeros((0, 3), np.float32)                       # all gone
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

    def _render_state(self, gid, center, R, size, kind, frustum, residual,
                      cams, color='unknown', cls='cup', score=0.0):
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
        # parse_label-compatible (verifier / plan_executor / boxes_to_detections
        # all read `c=<color>` + class): `#<id>_c=<color>_<class>_<score>`.
        label = f"#{gid}_c={color}_{cls}_{score:.2f}"
        return {'center': np.asarray(center), 'R': R, 'size': np.asarray(size),
                'top_world': top_world, 'frustum': frustum, 'label': label}

    def _is_publishable_track(self, tr) -> bool:
        """A track contributes to /digital_twin/boxes (and thus cups_on_table)
        only after min_hits consecutive matches with a rendered state — the same
        gate _publish_markers uses, so boxes and cups_on_table stay consistent."""
        return (tr.get('last_state') is not None
                and tr.get('hits', 0) >= self.min_hits)

    def _on_stack_ids(self, msg) -> None:
        self._stacked_ids = {int(x) for x in msg.data}

    def _publish_cups_on_table(self) -> None:
        """Color counts of upright, non-stacked cups (same {color:int} JSON the
        standalone point_cloud_node emitted) so the agent count lane survives in
        fusion mode. Excludes ids currently in /stack_track_ids."""
        counts = {c: 0 for c in self._cup_colors}
        counts['unknown'] = 0
        for gid, tr in self._tracks.items():
            if not self._is_publishable_track(tr):
                continue
            if gid in self._stacked_ids:
                continue
            if tr.get('cls') not in self.cup_class_names:
                continue
            col = tr.get('color') or 'unknown'
            counts[col] = counts.get(col, 0) + 1
        self.cups_on_table_pub.publish(String(data=json.dumps(counts)))

    # ------------------------------------------------------------------
    def _emit_dbg(self, pub, enabled, boxes, stamp) -> None:
        """Publish a debug MarkerArray: CUBE + label + optional cup frustum per
        box. Markers carry a short lifetime instead of a per-event DELETEALL:
        the full clear-and-redraw of raw, unfiltered per-event boxes made the
        1-3 debug overlays blink in RViz whenever a detection dropped for one
        event. Now a dropped box just lingers ~lifetime and ages out; same-id
        re-adds refresh in place. DELETEALL fires once on the enabled→disabled
        edge so toggling the checkbox off still clears the overlay."""
        ma = MarkerArray()
        if not enabled:
            if self._dbg_enabled_prev.get(pub, True):
                clr = Marker()
                clr.header.frame_id = self.world_frame
                clr.header.stamp = stamp
                clr.action = Marker.DELETEALL
                ma.markers.append(clr)
                pub.publish(ma)
            self._dbg_enabled_prev[pub] = False
            return
        self._dbg_enabled_prev[pub] = True
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
            cube.lifetime = self._dbg_lifetime
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
                txt.lifetime = self._dbg_lifetime
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
                    ln.lifetime = self._dbg_lifetime
                    ln.scale.x = 0.003
                    ln.color = ColorRGBA(r=col[0], g=col[1], b=col[2], a=0.9)
                    ln.pose.orientation.w = 1.0
                    for p in loop:
                        ln.points.append(MsgPoint(x=float(p[0]), y=float(p[1]),
                                                  z=float(p[2])))
                    ma.markers.append(ln)
        if ma.markers:
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
            if not self._is_publishable_track(tr):
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
