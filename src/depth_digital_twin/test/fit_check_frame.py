#!/usr/bin/env python3
"""Offline silhouette-fit check on ONE recorded-sequence frame (no ROS).

Replicates the pipeline for a single exo frame: ArUco → world pose (same
marker offsets as params.yaml world_origin_node), YOLO-seg → masks, depth →
rough z / init, then cup_geometry.fit_silhouette_xy per cup. Prints a table
and writes a debug overlay PNG (observed mask edge green, fitted silhouette
cyan, depth-median init red dot).

Usage:
  python3 fit_check_frame.py --seq /home/eunwoo/Projects/cup_stack/seq_record/0010 \
      --frame 1666 [--cam exo] [--out /tmp/fit_check.png]
"""
import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from depth_digital_twin.cup_geometry import (cone_silhouette_px,  # noqa: E402
                                             fit_silhouette_xy)

# Speed Stack cup + workspace marker (mirror params.yaml).
R_TOP, R_BOT, CUP_H = 0.054 / 2, 0.078 / 2, 0.095
MARKER_LEN = 0.10
MARKER_OFF = np.array([0.367, 0.003, 0.0])      # marker centre in base frame
MARKER_ROT_Z_DEG = -90.0
EXO_MODEL = 'vision/yolo/0609_exo_best.pt'


def find_marker_pose(seq: Path, cam: str, K, dist, frame: int):
    """T_cam<-marker from the requested frame, else scan the sequence
    (the exo camera is static, so any frame with the marker works)."""
    aruco = cv2.aruco
    det = aruco.ArucoDetector(
        aruco.getPredefinedDictionary(aruco.DICT_4X4_50),
        aruco.DetectorParameters())
    half = MARKER_LEN / 2.0
    obj = np.array([[-half, half, 0], [half, half, 0],
                    [half, -half, 0], [-half, -half, 0]], dtype=np.float64)
    frames = [frame] + list(range(0, 2700, 30))
    for f in frames:
        p = seq / cam / 'rgb' / f'{f:06d}.png'
        if not p.is_file():
            continue
        img = cv2.imread(str(p))
        corners, ids, _ = det.detectMarkers(img)
        if ids is None or 0 not in ids.flatten():
            continue
        c = corners[list(ids.flatten()).index(0)].reshape(4, 2)
        ok, rvec, tvec = cv2.solvePnP(
            obj, c.astype(np.float64), K, dist,
            flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if ok:
            R, _ = cv2.Rodrigues(rvec)
            print(f'[aruco] marker 0 found in frame {f:06d}, '
                  f'cam-dist {float(np.linalg.norm(tvec)):.3f} m')
            return R, tvec.reshape(3)
    raise SystemExit('ArUco marker 0 not found anywhere in the sequence')


def world_from_camera(R_cm, t_cm):
    """R_wc, t_wc with world = robot base (marker offsets from params.yaml)."""
    a = math.radians(MARKER_ROT_Z_DEG)
    R_bm = np.array([[math.cos(a), -math.sin(a), 0.0],
                     [math.sin(a), math.cos(a), 0.0],
                     [0.0, 0.0, 1.0]])
    R_mc = R_cm.T
    t_mc = -R_mc @ t_cm
    R_wc = R_bm @ R_mc
    t_wc = R_bm @ t_mc + MARKER_OFF
    return R_wc, t_wc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', required=True)
    ap.add_argument('--frame', type=int, required=True)
    ap.add_argument('--cam', default='exo')
    ap.add_argument('--conf', type=float, default=0.35)
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    seq = Path(args.seq)
    repo = Path(__file__).resolve().parents[3]

    import yaml
    intr = yaml.safe_load((seq / f'{args.cam}_intrinsics.yaml').read_text())
    K = np.array(intr['camera_matrix']['data'], dtype=np.float64).reshape(3, 3)
    dist = np.array(intr['distortion_coefficients']['data'],
                    dtype=np.float64).reshape(-1)

    R_cm, t_cm = find_marker_pose(seq, args.cam, K, dist, args.frame)
    R_wc, t_wc = world_from_camera(R_cm, t_cm)
    print(f'[tf] world<-{args.cam} cam at '
          f'({t_wc[0]:+.3f},{t_wc[1]:+.3f},{t_wc[2]:+.3f}) m')

    rgb = cv2.imread(str(seq / args.cam / 'rgb' / f'{args.frame:06d}.png'))
    depth_raw = cv2.imread(str(seq / args.cam / 'depth' / f'{args.frame:06d}.png'),
                           cv2.IMREAD_UNCHANGED)
    z_img = depth_raw.astype(np.float32) * 0.001
    h, w = rgb.shape[:2]

    from ultralytics import YOLO
    model = YOLO(str(repo / EXO_MODEL))
    r = model.predict(rgb, conf=args.conf, imgsz=640, verbose=False)[0]
    if r.masks is None:
        raise SystemExit('no detections on this frame')
    masks = r.masks.data.cpu().numpy()
    overlay = rgb.copy()
    fx, fy, cx_k, cy_k = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    print(f"\n{'cup':>4} {'cls':12} {'conf':>5} | {'init_xy(depth)':>16} "
          f"{'fit_xy':>16} {'d(mm)':>6} | {'z_base0':>7} {'iou':>5} "
          f"{'rms_px':>6} {'ms':>5}")
    import time
    for i in range(masks.shape[0]):
        cls = model.names[int(r.boxes.cls[i])]
        conf = float(r.boxes.conf[i])
        if cls != 'upright-cup':
            continue
        m = cv2.resize(masks[i], (w, h), interpolation=cv2.INTER_NEAREST)
        mask_u8 = (m > 0.5).astype(np.uint8) * 255
        sel = (mask_u8 > 0) & (z_img > 0.1) & (z_img < 4.0)
        if sel.sum() < 32:
            print(f'{i:>4} {cls:12} {conf:5.2f} | (too few depth px)')
            continue
        vy, vx = np.where(sel)
        z = z_img[vy, vx]
        pc = np.stack([(vx - cx_k) * z / fx, (vy - cy_k) * z / fy, z], axis=1)
        pw = (R_wc @ pc.T).T + t_wc
        z_top = float(np.percentile(pw[:, 2], 97.0))
        z_base0 = z_top - CUP_H
        xy0 = (float(np.median(pw[:, 0])), float(np.median(pw[:, 1])))
        t0 = time.perf_counter()
        fit = fit_silhouette_xy(
            mask_u8, K=K, dist=dist, R_wc=R_wc, t_wc=t_wc, r_top=R_TOP,
            r_bot=R_BOT, height=CUP_H, z_base=z_base0, xy0=xy0)
        ms = (time.perf_counter() - t0) * 1e3
        if not fit['ok']:
            print(f"{i:>4} {cls:12} {conf:5.2f} | fit FAILED ({fit['fail']})")
            continue
        d_mm = 1e3 * math.hypot(fit['x'] - xy0[0], fit['y'] - xy0[1])
        print(f"{i:>4} {cls:12} {conf:5.2f} | "
              f"({xy0[0]:+.3f},{xy0[1]:+.3f}) "
              f"({fit['x']:+.3f},{fit['y']:+.3f}) {d_mm:6.1f} | "
              f"{z_base0:+.3f} {fit['iou']:5.2f} {fit['rms_px']:6.2f} "
              f"{ms:5.1f}")

        cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(overlay, cnts, -1, (0, 255, 0), 1)
        sil = cone_silhouette_px(
            fit['x'], fit['y'], z_base0, r_top=R_TOP, r_bot=R_BOT,
            height=CUP_H, K=K, dist=dist, R_wc=R_wc, t_wc=t_wc)
        if sil is not None:
            cv2.polylines(overlay, [np.round(sil).astype(np.int32)], True,
                          (255, 255, 0), 2)
        # init point projected at z_base0 (depth-median starting guess)
        p0c = R_wc.T @ (np.array([xy0[0], xy0[1], z_base0]) - t_wc)
        u0 = int(round(fx * p0c[0] / p0c[2] + cx_k))
        v0 = int(round(fy * p0c[1] / p0c[2] + cy_k))
        cv2.circle(overlay, (u0, v0), 4, (0, 0, 255), -1)

    out = args.out or f'/tmp/fit_check_{seq.name}_{args.frame:06d}_{args.cam}.png'
    cv2.imwrite(out, overlay)
    print(f'\noverlay → {out}')


if __name__ == '__main__':
    main()
