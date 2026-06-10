"""Synthetic validation of cup_geometry — no ROS required.

Renders a truncated-cone mask analytically (dense surface points → convex
hull fill, a DIFFERENT path from the model projection used by the fit) and
asserts the silhouette fit recovers the known axis within millimetres.

Run directly:  python3 -m pytest src/depth_digital_twin/test/test_cup_geometry.py -v
"""
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from depth_digital_twin.cup_geometry import (  # noqa: E402
    cone_silhouette_px, contour_touches_border, edge_snap_fit,
    fit_silhouette_xy, fuse_xy, largest_contour, slide_xy_to_z, snap_level,
    xy_cov_from_px)

# Speed Stack cup model (params.yaml): wide circle at the BOTTOM.
R_TOP = 0.054 / 2.0
R_BOT = 0.078 / 2.0
CUP_H = 0.095

K = np.array([[615.0, 0.0, 320.0],
              [0.0, 615.0, 240.0],
              [0.0, 0.0, 1.0]])
DIST = np.zeros(5)
IMG_W, IMG_H = 640, 480


def look_at(eye, target, up_hint=(0.0, 0.0, 1.0)):
    """R_wc, t_wc for an optical-frame camera (z forward, x right, y down)."""
    eye = np.asarray(eye, dtype=np.float64)
    z_c = np.asarray(target, dtype=np.float64) - eye
    z_c /= np.linalg.norm(z_c)
    x_c = np.cross(z_c, np.asarray(up_hint, dtype=np.float64))
    n = np.linalg.norm(x_c)
    if n < 1e-6:                       # nadir view: up_hint ∥ z_c
        x_c = np.cross(z_c, np.array([1.0, 0.0, 0.0]))
        n = np.linalg.norm(x_c)
    x_c /= n
    y_c = np.cross(z_c, x_c)
    R_wc = np.stack([x_c, y_c, z_c], axis=1)
    return R_wc, eye


def render_cone_mask(cx, cy, z_base, R_wc, t_wc, n_theta=180, n_z=40,
                     r_top=R_TOP, r_bot=R_BOT):
    """Ground-truth mask: project a DENSE grid of cone surface points and
    fill their convex hull (independent of cone_silhouette_px's 2-circle
    shortcut)."""
    ang = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)
    zs = np.linspace(0.0, CUP_H, n_z)
    pts = []
    for zr in zs:
        r = r_bot + (r_top - r_bot) * (zr / CUP_H)
        pts.append(np.stack([cx + r * np.cos(ang), cy + r * np.sin(ang),
                             np.full(n_theta, z_base + zr)], axis=1))
    P_w = np.vstack(pts)
    R_cw = R_wc.T
    t_cw = -R_cw @ t_wc
    rvec, _ = cv2.Rodrigues(R_cw)
    px, _ = cv2.projectPoints(P_w.reshape(-1, 1, 3), rvec,
                              t_cw.reshape(3, 1), K, DIST)
    px = px.reshape(-1, 2)
    hull = cv2.convexHull(px.astype(np.float32)).astype(np.int32)
    mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    return mask


def run_fit(mask, R_wc, t_wc, z_base, xy0):
    return fit_silhouette_xy(
        mask, K=K, dist=DIST, R_wc=R_wc, t_wc=t_wc, r_top=R_TOP,
        r_bot=R_BOT, height=CUP_H, z_base=z_base, xy0=xy0)


def test_fit_recovers_known_xy_oblique_exo_view():
    true_xy = (0.02, -0.03)
    R_wc, t_wc = look_at(eye=(0.62, 0.05, 0.55), target=(0.0, 0.0, 0.05))
    mask = render_cone_mask(*true_xy, 0.0, R_wc, t_wc)
    fit = run_fit(mask, R_wc, t_wc, 0.0, (true_xy[0] + 0.025,
                                          true_xy[1] - 0.02))
    assert fit['ok']
    err_mm = 1e3 * math.hypot(fit['x'] - true_xy[0], fit['y'] - true_xy[1])
    assert err_mm < 3.0, f'oblique fit error {err_mm:.2f} mm'
    assert fit['iou'] > 0.90
    assert fit['rms_px'] < 2.5
    assert not fit['truncated']


def test_fit_recovers_known_xy_nadir_hand_view():
    true_xy = (-0.015, 0.025)
    R_wc, t_wc = look_at(eye=(-0.015, 0.025, 0.55),
                         target=(-0.015, 0.025, 0.0))
    mask = render_cone_mask(*true_xy, 0.0, R_wc, t_wc)
    fit = run_fit(mask, R_wc, t_wc, 0.0, (true_xy[0] - 0.02,
                                          true_xy[1] + 0.025))
    assert fit['ok']
    err_mm = 1e3 * math.hypot(fit['x'] - true_xy[0], fit['y'] - true_xy[1])
    assert err_mm < 3.0, f'nadir fit error {err_mm:.2f} mm'
    assert fit['iou'] > 0.90


def test_fit_robust_to_mask_bleed():
    """A blob glued to the cup mask (YOLO bleeding into a neighbour) must not
    drag the model — chamfer residual is model→observed only."""
    true_xy = (0.0, 0.0)
    R_wc, t_wc = look_at(eye=(0.6, 0.0, 0.55), target=(0.0, 0.0, 0.05))
    mask = render_cone_mask(*true_xy, 0.0, R_wc, t_wc)
    ys, xs = np.nonzero(mask)
    cx_px, cy_px = int(xs.mean()), int(ys.mean())
    cv2.circle(mask, (cx_px + 55, cy_px + 10), 28, 255, -1)  # attached blob
    fit = run_fit(mask, R_wc, t_wc, 0.0, (0.02, 0.02))
    assert fit['ok']
    err_mm = 1e3 * math.hypot(fit['x'] - true_xy[0], fit['y'] - true_xy[1])
    assert err_mm < 6.0, f'bleed fit error {err_mm:.2f} mm'


def test_fit_elevated_base_wrong_assumption_slides_along_ray():
    """Fit with a WRONG base assumption (cup actually one nest level up),
    then slide_xy_to_z must mostly cancel the induced bias."""
    true_xy = (0.01, 0.02)
    nest = 0.020
    z_true = nest                       # one level above the table
    R_wc, t_wc = look_at(eye=(0.62, 0.0, 0.55), target=(0.0, 0.0, 0.05))
    mask = render_cone_mask(*true_xy, z_true, R_wc, t_wc)
    fit = run_fit(mask, R_wc, t_wc, 0.0, true_xy)   # assumes z_base=0
    assert fit['ok']
    biased_mm = 1e3 * math.hypot(fit['x'] - true_xy[0],
                                 fit['y'] - true_xy[1])
    p = np.array([fit['x'], fit['y'], 0.0])
    d = p - t_wc
    d /= np.linalg.norm(d)
    slid = slide_xy_to_z(fit['x'], fit['y'], 0.0, d, z_true)
    assert slid is not None
    slid_mm = 1e3 * math.hypot(slid[0] - true_xy[0], slid[1] - true_xy[1])
    assert slid_mm < biased_mm, (biased_mm, slid_mm)
    assert slid_mm < 6.0, f'slide-corrected error {slid_mm:.2f} mm'


def test_slide_xy_to_z_closed_form():
    d = np.array([0.5, 0.0, -0.5])
    d /= np.linalg.norm(d)
    out = slide_xy_to_z(0.5, 0.0, 0.0, d, 0.1)
    assert out is not None
    assert abs(out[0] - 0.4) < 1e-9 and abs(out[1]) < 1e-9
    flat = np.array([1.0, 0.0, 0.01])
    assert slide_xy_to_z(0.0, 0.0, 0.0, flat, 0.1) is None


def test_snap_level():
    k, zb, err = snap_level(0.097, table_z=0.0, cup_h=0.095, nest_off=0.02)
    assert k == 0 and abs(zb) < 1e-9 and abs(err - 0.002) < 1e-9
    k, zb, err = snap_level(0.135, table_z=0.0, cup_h=0.095, nest_off=0.02)
    assert k == 2 and abs(zb - 0.04) < 1e-9 and abs(err) < 1e-9
    k, zb, _ = snap_level(0.05, table_z=0.0, cup_h=0.095, nest_off=0.02)
    assert k == 0 and abs(zb) < 1e-9          # never below the table


def test_fuse_xy_inverse_variance_weighting():
    m1 = (np.array([0.01, 0.0]), 1e-4 * np.eye(2))
    m2 = (np.array([0.03, 0.0]), 9e-4 * np.eye(2))
    xy, cov, resid = fuse_xy([m1, m2])
    assert abs(xy[0] - 0.012) < 1e-6
    assert cov[0, 0] < 1e-4                    # fused tighter than best input
    assert abs(resid[0][0] - (-0.002)) < 1e-6
    assert abs(resid[1][0] - 0.018) < 1e-6


def test_xy_cov_anisotropy():
    oblique = np.array([math.sqrt(0.5), 0.0, -math.sqrt(0.5)])
    cov = xy_cov_from_px(2.0, 0.8, 615.0, oblique)
    assert cov[0, 0] > 1.5 * cov[1, 1]         # stretched along view dir
    nadir = np.array([0.0, 0.0, -1.0])
    cov_n = xy_cov_from_px(2.0, 0.5, 615.0, nadir)
    assert abs(cov_n[0, 0] - cov_n[1, 1]) < 1e-12


def test_two_camera_end_to_end_fusion():
    """Full consumer-path math: exo (oblique) + hand (nadir) observe the same
    nested cup (level 1); both producers assume a slightly wrong base from
    noisy rough-z; snap → slide → inverse-covariance fuse must land < 3 mm."""
    from depth_digital_twin.cup_geometry import (fuse_xy, ray_through_point,
                                                 snap_level)
    true_xy = (0.015, -0.01)
    nest = 0.020
    z_true = nest                                  # level 1
    cams = [look_at(eye=(0.62, 0.05, 0.55), target=(0.0, 0.0, 0.05)),
            look_at(eye=(0.015, -0.01, 0.55), target=(0.015, -0.01, 0.0))]
    meas = []
    for R_wc, t_wc in cams:
        mask = render_cone_mask(*true_xy, z_true, R_wc, t_wc)
        z_top_rough = z_true + CUP_H + 0.008       # +8 mm depth bias
        z_base0 = z_top_rough - CUP_H              # producer's assumption
        fit = run_fit(mask, R_wc, t_wc, z_base0,
                      (true_xy[0] + 0.015, true_xy[1] - 0.015))
        assert fit['ok'] and fit['iou'] > 0.85
        k, z_base, _ = snap_level(z_top_rough, table_z=0.0, cup_h=CUP_H,
                                  nest_off=nest)
        assert k == 1 and abs(z_base - z_true) < 1e-9
        _, d = ray_through_point(
            np.array([fit['x'], fit['y'], z_base0]), t_wc)
        slid = slide_xy_to_z(fit['x'], fit['y'], z_base0, d, z_base)
        assert slid is not None
        rng = float(np.linalg.norm(
            np.array([slid[0], slid[1], z_base]) - t_wc))
        cov = xy_cov_from_px(2.0, rng, 615.0, d)
        meas.append((np.asarray(slid), cov))
    xy_f, cov_f, resid = fuse_xy(meas)
    err_mm = 1e3 * math.hypot(xy_f[0] - true_xy[0], xy_f[1] - true_xy[1])
    assert err_mm < 3.0, f'fused error {err_mm:.2f} mm'
    # cross-camera residuals small (consistent extrinsics in this synthetic)
    assert all(np.linalg.norm(r) < 0.01 for r in resid)


def test_boundary_offset_absorbs_uniform_mask_error():
    """A uniformly dilated/eroded mask (YOLO over/under-segmentation) must be
    absorbed by the boundary-offset parameter b, keeping (x, y) accurate and
    reporting the bias magnitude."""
    true_xy = (0.01, -0.02)
    R_wc, t_wc = look_at(eye=(0.62, 0.05, 0.55), target=(0.0, 0.0, 0.05))
    mask = render_cone_mask(*true_xy, 0.0, R_wc, t_wc)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    for op, b_lo, b_hi in ((cv2.dilate, 1.5, 4.0), (cv2.erode, -4.0, -1.5)):
        bad = op(mask, kernel)
        fit = fit_silhouette_xy(
            bad, K=K, dist=DIST, R_wc=R_wc, t_wc=t_wc, r_top=R_TOP,
            r_bot=R_BOT, height=CUP_H, z_base=0.0,
            xy0=(true_xy[0] + 0.015, true_xy[1] - 0.015),
            fit_boundary_offset=True)
        assert fit['ok']
        err_mm = 1e3 * math.hypot(fit['x'] - true_xy[0],
                                  fit['y'] - true_xy[1])
        assert err_mm < 3.0, f'{op.__name__} err {err_mm:.2f} mm'
        assert b_lo <= fit['b_px'] <= b_hi, \
            f"{op.__name__} b={fit['b_px']:.2f}px"


def test_edge_snap_recovers_from_asymmetric_mask_bias():
    """Mask bled 5 px to one side (asymmetric over-segmentation) biases the
    chamfer fit; edge-snap to the actual image gradient must pull the centre
    back to (near) truth."""
    true_xy = (0.0, 0.0)
    R_wc, t_wc = look_at(eye=(0.6, 0.0, 0.55), target=(0.0, 0.0, 0.05))
    mask = render_cone_mask(*true_xy, 0.0, R_wc, t_wc)
    # Synthetic photo: dark cup on bright table, true silhouette boundary.
    gray = np.full((IMG_H, IMG_W), 175, dtype=np.uint8)
    gray[mask > 0] = 60
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # Asymmetric bleed: union with a 5-px right-shifted copy.
    shifted = np.zeros_like(mask)
    shifted[:, 5:] = mask[:, :-5]
    bad = cv2.bitwise_or(mask, shifted)
    chamfer = fit_silhouette_xy(
        bad, K=K, dist=DIST, R_wc=R_wc, t_wc=t_wc, r_top=R_TOP,
        r_bot=R_BOT, height=CUP_H, z_base=0.0, xy0=(0.015, 0.015),
        fit_boundary_offset=True)
    assert chamfer['ok']
    err_chamfer = 1e3 * math.hypot(chamfer['x'] - true_xy[0],
                                   chamfer['y'] - true_xy[1])
    snap = edge_snap_fit(
        gray, K=K, dist=DIST, R_wc=R_wc, t_wc=t_wc, r_top=R_TOP,
        r_bot=R_BOT, height=CUP_H, z_base=0.0,
        xy0=(chamfer['x'], chamfer['y']))
    assert snap['ok'] and snap['edge_cov'] > 0.7
    err_snap = 1e3 * math.hypot(snap['x'] - true_xy[0],
                                snap['y'] - true_xy[1])
    assert err_snap < err_chamfer, (err_chamfer, err_snap)
    assert err_snap < 1.5, f'edge-snap error {err_snap:.2f} mm'


def test_fit_flip_profile_mouth_up_cup():
    """A mouth-up cup (wide opening UP — radii swapped) viewed obliquely:
    flip_profile=True must fit it better than the normal profile and land
    within tolerance, so a flip-and-pick-best strategy can disambiguate."""
    true_xy = (0.0, 0.0)
    R_wc, t_wc = look_at(eye=(0.6, 0.0, 0.55), target=(0.0, 0.0, 0.05))
    mask = render_cone_mask(*true_xy, 0.0, R_wc, t_wc,
                            r_top=R_BOT, r_bot=R_TOP)   # mouth-up render
    normal = fit_silhouette_xy(
        mask, K=K, dist=DIST, R_wc=R_wc, t_wc=t_wc, r_top=R_TOP,
        r_bot=R_BOT, height=CUP_H, z_base=0.0, xy0=(0.02, 0.02))
    flipped = fit_silhouette_xy(
        mask, K=K, dist=DIST, R_wc=R_wc, t_wc=t_wc, r_top=R_TOP,
        r_bot=R_BOT, height=CUP_H, z_base=0.0, xy0=(0.02, 0.02),
        flip_profile=True)
    assert flipped['ok'] and flipped['flipped']
    assert flipped['iou'] > normal['iou']
    err_mm = 1e3 * math.hypot(flipped['x'] - true_xy[0],
                              flipped['y'] - true_xy[1])
    assert err_mm < 3.0, f'flip fit error {err_mm:.2f} mm'


def test_contour_helpers():
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(mask, (50, 50), 20, 255, -1)
    c = largest_contour(mask)
    assert c is not None and c.shape[0] > 40
    assert not contour_touches_border(c, 100, 100)
    mask2 = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(mask2, (0, 50), 30, 255, -1)
    c2 = largest_contour(mask2)
    assert contour_touches_border(c2, 100, 100)


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
        print(f'PASS {fn.__name__}')
    print(f'{len(fns)} tests passed')
