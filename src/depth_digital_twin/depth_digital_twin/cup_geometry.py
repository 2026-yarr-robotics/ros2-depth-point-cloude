"""cup_geometry — pure geometry/estimation helpers for silhouette-based cup
pose estimation (no ROS imports; unit-testable in isolation).

Core idea: the cup is a KNOWN truncated cone and an upright cup has only two
unknowns (axis x, y) once the base elevation is assumed. The most precise
evidence available is the YOLO mask *silhouette* + camera calibration — NOT
per-pixel depth (stereo depth on a cup wall is mm-to-cm biased; a mask contour
at 1 m with fx≈600 is ~1.5 mm/px). So:

* `fit_silhouette_xy`   — align the projected cone silhouette to the observed
                          mask contour (chamfer distance, 2-DOF least squares).
* `slide_xy_to_z`       — re-slide a fitted (x, y) along the view ray when the
                          consumer snaps the base to a different level.
* `xy_cov_from_px`      — pixel noise → world XY covariance, inflated along
                          the view direction for oblique views.
* `fuse_xy`             — inverse-covariance fusion of per-camera estimates,
                          returning per-camera residuals (extrinsic health).
* `snap_level`          — quantise a rough top-z onto the nesting lattice.

Conventions: world frame is Z-up (robot base); `R_wc`, `t_wc` map camera
optical frame → world (p_w = R_wc @ p_c + t_wc), matching point_cloud_node.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

__all__ = [
    'largest_contour', 'contour_touches_border', 'cone_silhouette_px',
    'fit_silhouette_xy', 'edge_snap_fit', 'ray_through_point',
    'slide_xy_to_z', 'xy_cov_from_px', 'fuse_xy', 'snap_level',
]


# ---------------------------------------------------------------------------
# Contour helpers
# ---------------------------------------------------------------------------

def largest_contour(mask_u8: np.ndarray) -> np.ndarray | None:
    """Largest external contour of a binary mask as an (N, 2) float64 array
    of (u, v) pixel coordinates, or None when the mask is empty."""
    cnts, _ = cv2.findContours(
        mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) <= 0.0:
        return None
    return c.reshape(-1, 2).astype(np.float64)


def contour_touches_border(contour: np.ndarray, width: int, height: int,
                           margin: int = 2) -> bool:
    """True when the contour reaches the image border (mask truncated — the
    silhouette is incomplete, so the fit deserves an inflated sigma)."""
    u = contour[:, 0]
    v = contour[:, 1]
    return bool(np.any(u <= margin) or np.any(v <= margin)
                or np.any(u >= width - 1 - margin)
                or np.any(v >= height - 1 - margin))


def _polyline_normals(pts: np.ndarray) -> np.ndarray:
    """Outward unit normals of a CLOSED convex polyline ((n,2) pixels).
    Orientation is fixed by pointing away from the polygon centroid, so it
    works regardless of the hull's winding direction."""
    t = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
    n = np.stack([t[:, 1], -t[:, 0]], axis=1)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    flip = np.sum((pts - pts.mean(axis=0)) * n, axis=1) < 0
    n[flip] *= -1.0
    return n


def _resample_polyline(pts: np.ndarray, n: int) -> np.ndarray:
    """Resample a CLOSED polygon to n points evenly spaced by arclength."""
    closed = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total <= 1e-9:
        return np.repeat(pts[:1], n, axis=0)
    s = np.linspace(0.0, total, n, endpoint=False)
    out = np.empty((n, 2), dtype=np.float64)
    out[:, 0] = np.interp(s, cum, closed[:, 0])
    out[:, 1] = np.interp(s, cum, closed[:, 1])
    return out


# ---------------------------------------------------------------------------
# Cone model projection
# ---------------------------------------------------------------------------

def _cone_circle_points(cx: float, cy: float, z_base: float, *, r_top: float,
                        r_bot: float, height: float,
                        n_theta: int) -> np.ndarray:
    """World points of the bottom + top circles of the truncated cone."""
    ang = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)
    ca, sa = np.cos(ang), np.sin(ang)
    bot = np.stack([cx + r_bot * ca, cy + r_bot * sa,
                    np.full(n_theta, z_base)], axis=1)
    top = np.stack([cx + r_top * ca, cy + r_top * sa,
                    np.full(n_theta, z_base + height)], axis=1)
    return np.vstack([bot, top])


def cone_silhouette_px(cx: float, cy: float, z_base: float, *, r_top: float,
                       r_bot: float, height: float, K: np.ndarray,
                       dist: np.ndarray, R_wc: np.ndarray, t_wc: np.ndarray,
                       n_theta: int = 36,
                       n_samples: int = 120) -> np.ndarray | None:
    """Projected silhouette outline of the cone as (n_samples, 2) pixels.

    A frustum is convex, so its silhouette is the convex hull of the two
    projected circles — no need to find the true tangent generatrices.
    Returns None when the cone is (partly) behind the camera.
    """
    P_w = _cone_circle_points(cx, cy, z_base, r_top=r_top, r_bot=r_bot,
                              height=height, n_theta=n_theta)
    R_cw = R_wc.T
    t_cw = -R_cw @ t_wc
    P_c = (R_cw @ P_w.T).T + t_cw
    if np.any(P_c[:, 2] <= 1e-3):
        return None
    rvec, _ = cv2.Rodrigues(R_cw)
    px, _ = cv2.projectPoints(
        P_w.reshape(-1, 1, 3), rvec, t_cw.reshape(3, 1), K,
        dist.reshape(-1) if dist is not None else None)
    px = px.reshape(-1, 2)
    if not np.all(np.isfinite(px)):
        return None
    hull = cv2.convexHull(px.astype(np.float32)).reshape(-1, 2)
    if hull.shape[0] < 3:
        return None
    return _resample_polyline(hull.astype(np.float64), n_samples)


# ---------------------------------------------------------------------------
# Silhouette chamfer fit (the measurement)
# ---------------------------------------------------------------------------

def fit_silhouette_xy(mask_u8: np.ndarray, *, K: np.ndarray, dist: np.ndarray,
                      R_wc: np.ndarray, t_wc: np.ndarray, r_top: float,
                      r_bot: float, height: float, z_base: float,
                      xy0: tuple[float, float], roi_pad_px: int = 60,
                      n_theta: int = 36, n_samples: int = 120,
                      f_scale_px: float = 2.0,
                      flip_profile: bool = False,
                      fit_boundary_offset: bool = False,
                      max_offset_px: float = 4.0) -> dict:
    """Fit the cup axis (x, y) at ASSUMED base elevation `z_base` by aligning
    the projected cone silhouette to the observed mask contour.

    The residual is model→observed chamfer distance (sampled from a distance
    transform of the contour), so extra observed contour (mask bleeding into a
    neighbour cup) does NOT attract the model; missing contour (occlusion)
    is down-weighted by the soft_l1 loss.

    flip_profile=True swaps r_top/r_bot — a MOUTH-UP cup (wide opening up,
    e.g. detectors without a dedicated mouth-up class label it upright).

    fit_boundary_offset=True adds a third parameter b (px, |b| ≤
    max_offset_px): the model silhouette is displaced by b along its outward
    contour normal before the chamfer residual. A YOLO mask that is
    UNIFORMLY over/under-segmented is absorbed into b instead of leaking
    into (x, y) through whatever asymmetry the view provides; the recovered
    b is also a per-camera diagnostic of the detector's boundary bias.

    Always returns a dict. Success: {'ok': True, 'x', 'y', 'b_px', 'rms_px',
    'iou', 'n_contour', 'truncated', 'flipped'}. Failure: {'ok': False,
    'fail': '<reason>'} so debug overlays / CSV can show WHY a fit dropped.
    """
    from scipy.optimize import least_squares

    if flip_profile:
        r_top, r_bot = r_bot, r_top
    contour = largest_contour(mask_u8)
    if contour is None or contour.shape[0] < 24:
        return {'ok': False, 'fail': 'few_contour'}
    h, w = mask_u8.shape[:2]
    truncated = contour_touches_border(contour, w, h)

    # Distance transform on a padded ROI around the observed contour: each
    # pixel = distance to the nearest contour pixel.
    u_lo = max(0, int(contour[:, 0].min()) - roi_pad_px)
    v_lo = max(0, int(contour[:, 1].min()) - roi_pad_px)
    u_hi = min(w, int(contour[:, 0].max()) + roi_pad_px + 1)
    v_hi = min(h, int(contour[:, 1].max()) + roi_pad_px + 1)
    roi_w, roi_h = u_hi - u_lo, v_hi - v_lo
    if roi_w < 8 or roi_h < 8:
        return {'ok': False, 'fail': 'roi_small'}
    canvas = np.full((roi_h, roi_w), 255, dtype=np.uint8)
    cu = np.clip(contour[:, 0].astype(np.int32) - u_lo, 0, roi_w - 1)
    cv_ = np.clip(contour[:, 1].astype(np.int32) - v_lo, 0, roi_h - 1)
    canvas[cv_, cu] = 0
    dt = cv2.distanceTransform(canvas, cv2.DIST_L2, 5)
    # Out-of-ROI model points clamp onto the border, where dt is already
    # large (≥ roi_pad_px away from any contour) — a natural escape penalty.

    def residuals(p):
        sil = cone_silhouette_px(
            float(p[0]), float(p[1]), z_base, r_top=r_top, r_bot=r_bot,
            height=height, K=K, dist=dist, R_wc=R_wc, t_wc=t_wc,
            n_theta=n_theta, n_samples=n_samples)
        if sil is None:
            return np.full(n_samples, 1e3)
        if fit_boundary_offset:
            sil = sil + float(p[2]) * _polyline_normals(sil)
        # Bilinear DT sampling — keeps the residual continuous in (x, y) so
        # finite-difference gradients are meaningful at sub-pixel steps
        # (nearest-pixel sampling plateaus and stalls the optimizer mm-scale).
        fu = np.clip(sil[:, 0] - u_lo, 0.0, roi_w - 1.001)
        fv = np.clip(sil[:, 1] - v_lo, 0.0, roi_h - 1.001)
        iu = fu.astype(np.int64)
        iv = fv.astype(np.int64)
        au = fu - iu
        av = fv - iv
        d00 = dt[iv, iu]
        d01 = dt[iv, iu + 1]
        d10 = dt[iv + 1, iu]
        d11 = dt[iv + 1, iu + 1]
        return ((1 - av) * ((1 - au) * d00 + au * d01)
                + av * ((1 - au) * d10 + au * d11)).astype(np.float64)

    if fit_boundary_offset:
        x0 = np.array([xy0[0], xy0[1], 0.0])
        lb = np.array([-np.inf, -np.inf, -max_offset_px])
        ub = np.array([np.inf, np.inf, max_offset_px])
        # b is in px while x/y are in metres — tell the optimizer the scale
        # difference or the numeric Jacobian misbehaves.
        x_scale = np.array([0.01, 0.01, 1.0])
    else:
        x0 = np.asarray(xy0, dtype=np.float64)
        lb, ub = -np.inf, np.inf
        x_scale = np.array([0.01, 0.01])
    try:
        res = least_squares(
            residuals, x0=x0, bounds=(lb, ub), x_scale=x_scale,
            loss='soft_l1', f_scale=f_scale_px, diff_step=5e-4, xtol=1e-7)
    except Exception:
        return {'ok': False, 'fail': 'ls_error'}
    if not np.all(np.isfinite(res.x)):
        return {'ok': False, 'fail': 'non_finite'}
    x_fit, y_fit = float(res.x[0]), float(res.x[1])
    b_px = float(res.x[2]) if fit_boundary_offset else 0.0
    r = residuals(res.x)
    rms = float(np.sqrt(np.mean(r ** 2)))

    # Verification: rendered silhouette vs observed mask IoU on the ROI.
    sil = cone_silhouette_px(
        x_fit, y_fit, z_base, r_top=r_top, r_bot=r_bot, height=height,
        K=K, dist=dist, R_wc=R_wc, t_wc=t_wc,
        n_theta=n_theta, n_samples=n_samples)
    if sil is not None and fit_boundary_offset:
        # IoU is rendered-vs-MASK agreement, so include the mask-bias term.
        sil = sil + b_px * _polyline_normals(sil)
    iou = 0.0
    if sil is not None:
        model_fill = np.zeros((roi_h, roi_w), dtype=np.uint8)
        poly = np.round(sil - np.array([u_lo, v_lo])).astype(np.int32)
        cv2.fillConvexPoly(model_fill, poly, 1)
        obs_fill = (mask_u8[v_lo:v_hi, u_lo:u_hi] > 0).astype(np.uint8)
        inter = int(np.sum(model_fill & obs_fill))
        union = int(np.sum(model_fill | obs_fill))
        iou = float(inter) / float(union) if union > 0 else 0.0

    return {'ok': True, 'x': x_fit, 'y': y_fit, 'b_px': b_px, 'rms_px': rms,
            'iou': iou, 'n_contour': int(contour.shape[0]),
            'truncated': truncated, 'flipped': bool(flip_profile)}


# ---------------------------------------------------------------------------
# Edge-snap refinement (image-gradient evidence replaces the YOLO boundary)
# ---------------------------------------------------------------------------

def edge_snap_fit(gray: np.ndarray, *, K: np.ndarray, dist: np.ndarray,
                  R_wc: np.ndarray, t_wc: np.ndarray, r_top: float,
                  r_bot: float, height: float, z_base: float,
                  xy0: tuple[float, float], search_px: float = 6.0,
                  min_grad: float = 8.0, n_theta: int = 36,
                  n_samples: int = 120, iters: int = 2,
                  min_cov: float = 0.3) -> dict:
    """Refine a silhouette fit by snapping the model boundary to IMAGE
    gradient edges instead of the YOLO mask contour.

    The mask only provides the starting pose (`xy0`, from the chamfer fit);
    the final boundary evidence is the gradient-magnitude peak along each
    silhouette sample's outward normal (sub-pixel via parabolic refinement).
    Samples without a strong edge (occlusion, cup-on-cup contact where there
    is no contrast) are dropped — `edge_cov` reports the surviving fraction.

    Success: {'ok': True, 'x', 'y', 'rms_px', 'edge_cov', 'n_snap'};
    failure: {'ok': False, 'fail': '<reason>'}.
    """
    from scipy.optimize import least_squares

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    h, w = gray.shape[:2]
    offs = np.linspace(-search_px, search_px, int(search_px * 4) + 1)
    step = offs[1] - offs[0]
    p = np.array([xy0[0], xy0[1]], dtype=np.float64)
    rms = 0.0
    cov = 0.0
    n_snap = 0

    def _sample(coords):
        u = np.clip(coords[..., 0], 0.0, w - 1.001)
        v = np.clip(coords[..., 1], 0.0, h - 1.001)
        iu = u.astype(np.int32)
        iv = v.astype(np.int32)
        au = u - iu
        av = v - iv
        return ((1 - av) * ((1 - au) * mag[iv, iu] + au * mag[iv, iu + 1])
                + av * ((1 - au) * mag[iv + 1, iu] + au * mag[iv + 1, iu + 1]))

    for _ in range(max(1, iters)):
        sil = cone_silhouette_px(
            p[0], p[1], z_base, r_top=r_top, r_bot=r_bot, height=height,
            K=K, dist=dist, R_wc=R_wc, t_wc=t_wc,
            n_theta=n_theta, n_samples=n_samples)
        if sil is None:
            return {'ok': False, 'fail': 'behind_cam'}
        nrm = _polyline_normals(sil)
        m = _sample(sil[:, None, :] + nrm[:, None, :] * offs[None, :, None])
        k = np.argmax(m, axis=1)
        ar = np.arange(len(k))
        peak = m[ar, k]
        valid = peak >= min_grad
        n_snap = int(valid.sum())
        cov = n_snap / float(n_samples)
        if cov < min_cov:
            return {'ok': False, 'fail': 'few_edges'}
        # Parabolic sub-sample peak refinement.
        km = np.clip(k - 1, 0, len(offs) - 1)
        kp = np.clip(k + 1, 0, len(offs) - 1)
        y0, y1, y2 = m[ar, km], peak, m[ar, kp]
        den = y0 - 2.0 * y1 + y2
        delta = np.where(np.abs(den) > 1e-6,
                         0.5 * (y0 - y2) / np.where(np.abs(den) > 1e-6,
                                                    den, 1.0), 0.0)
        snapped = sil + nrm * (offs[k]
                               + np.clip(delta, -1.0, 1.0) * step)[:, None]
        idx = np.where(valid)[0]

        def residuals(q):
            s2 = cone_silhouette_px(
                float(q[0]), float(q[1]), z_base, r_top=r_top, r_bot=r_bot,
                height=height, K=K, dist=dist, R_wc=R_wc, t_wc=t_wc,
                n_theta=n_theta, n_samples=n_samples)
            if s2 is None:
                return np.full(len(idx), 1e3)
            # Signed distance to the snapped edge along the (fixed) normal.
            return np.sum((s2[idx] - snapped[idx]) * nrm[idx], axis=1)

        try:
            res = least_squares(
                residuals, x0=p, loss='soft_l1', f_scale=1.5,
                x_scale=np.array([0.01, 0.01]), diff_step=5e-4, xtol=1e-7)
        except Exception:
            return {'ok': False, 'fail': 'ls_error'}
        if not np.all(np.isfinite(res.x)):
            return {'ok': False, 'fail': 'non_finite'}
        p = res.x
        r_fin = residuals(p)
        rms = float(np.sqrt(np.mean(r_fin ** 2)))

    return {'ok': True, 'x': float(p[0]), 'y': float(p[1]), 'rms_px': rms,
            'edge_cov': float(cov), 'n_snap': n_snap}


# ---------------------------------------------------------------------------
# Ray / covariance / fusion helpers (consumer side)
# ---------------------------------------------------------------------------

def ray_through_point(p_world: np.ndarray,
                      t_wc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(origin, unit direction) of the camera-centre ray through p_world."""
    origin = np.asarray(t_wc, dtype=np.float64)
    d = np.asarray(p_world, dtype=np.float64) - origin
    n = float(np.linalg.norm(d))
    if n <= 1e-9:
        return origin, np.array([0.0, 0.0, -1.0])
    return origin, d / n


def slide_xy_to_z(x0: float, y0: float, z0: float, ray_dir: np.ndarray,
                  z_new: float,
                  min_abs_dz: float = 0.15) -> tuple[float, float] | None:
    """Move a fitted axis point along its view ray from elevation z0 to z_new.

    First-order correction for a wrong assumed base level: the silhouette fit
    constrains the cup's bearing far better than its range, so a level change
    slides the estimate along the ray. None when the ray is too horizontal
    for a stable intersection (|dz| < min_abs_dz).
    """
    dz = float(ray_dir[2])
    if abs(dz) < min_abs_dz:
        return None
    s = (z_new - z0) / dz
    return x0 + s * float(ray_dir[0]), y0 + s * float(ray_dir[1])


def xy_cov_from_px(sigma_px: float, range_m: float, f_px: float,
                   ray_dir: np.ndarray) -> np.ndarray:
    """2x2 world-XY covariance of a silhouette-derived (x, y) measurement.

    Base: sigma_m = sigma_px * range / f (pixel error scaled to metres at the
    cup). Anisotropy: error along the view direction's XY projection grows as
    1/|dz| — a near-horizontal view barely constrains range, a top-down view
    (|dz|→1) is isotropic. This is what lets a vertical hand view and an
    oblique exo view fuse with mathematically correct weights.
    """
    sigma_m = max(1e-4, float(sigma_px) * max(0.05, float(range_m))
                  / max(1.0, float(f_px)))
    base = sigma_m ** 2
    g = np.asarray(ray_dir[:2], dtype=np.float64)
    gn = float(np.linalg.norm(g))
    dz = max(0.1, abs(float(ray_dir[2])))
    if gn < 1e-9:
        return base * np.eye(2)
    ghat = g / gn
    stretch = 1.0 / (dz * dz)            # 1 (nadir) … 100 (10° elevation)
    return base * (np.eye(2) + (stretch - 1.0) * np.outer(ghat, ghat))


def fuse_xy(measurements: list[tuple[np.ndarray, np.ndarray]]):
    """Inverse-covariance fusion of [(xy(2,), cov(2,2)), ...].

    Returns (xy_fused(2,), cov_fused(2,2), residuals list[(2,)]) — residuals
    are per-input (xy_i - xy_fused), the cross-camera consistency signal.
    """
    if not measurements:
        return None
    info = np.zeros((2, 2))
    vec = np.zeros(2)
    for xy, cov in measurements:
        w = np.linalg.inv(np.asarray(cov, dtype=np.float64)
                          + 1e-12 * np.eye(2))
        info += w
        vec += w @ np.asarray(xy, dtype=np.float64)
    cov_f = np.linalg.inv(info)
    xy_f = cov_f @ vec
    resid = [np.asarray(xy, dtype=np.float64) - xy_f
             for xy, _ in measurements]
    return xy_f, cov_f, resid


def snap_level(z_top_rough: float, *, table_z: float, cup_h: float,
               nest_off: float, max_levels: int = 14):
    """Quantise a rough top-of-cup z onto the nesting lattice.

    Level k means the cup's base sits k nesting offsets above the table
    (k=0: directly on the table). Returns (k, z_base, err_m) where err_m is
    the lattice mismatch — |err| > nest_off/2 never happens by construction,
    but a large err relative to nest_off flags an ambiguous classification.
    """
    if nest_off <= 1e-6:
        return 0, table_z, z_top_rough - (table_z + cup_h)
    k = int(round((float(z_top_rough) - table_z - cup_h) / nest_off))
    k = max(0, min(int(max_levels), k))
    z_base = table_z + k * nest_off
    err = float(z_top_rough) - (z_base + cup_h)
    return k, z_base, err
