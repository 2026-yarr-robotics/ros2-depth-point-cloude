#!/usr/bin/env python3
"""Batch silhouette-fit harness over a recorded sequence (no ROS).

Runs the silhouette measurement path frame-by-frame over a recorded exo
sequence with ByteTrack ids (same tracker as the live detection_node), and
produces:

* a per-fit CSV          (frame, track, fit/init xy, iou, rms, fail reason …)
* an overlay video       (observed contour green / fitted silhouette cyan /
                          depth init red / per-cup text)
* a per-track summary    stationary-segment sigma_xy for BOTH the silhouette
                          fit and the depth-median init (the A/B number),
                          mean IoU/rms, fail rate, flip-better count.

This measures the Phase-1 exit criteria offline: fit sigma_xy < 2 mm on
stationary cups, fit success rate > 90 %.

Usage:
  python3 fit_check_sequence.py --seq /home/eunwoo/Projects/cup_stack/seq_record/0010 \
      [--start 0 --end -1 --stride 3] [--conf 0.35] [--try-flip] \
      [--csv /tmp/fit_0010.csv] [--video /tmp/fit_0010.mp4]
"""
import argparse
import csv as csv_mod
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from depth_digital_twin.cup_geometry import (cone_silhouette_px,  # noqa: E402
                                             edge_snap_fit,
                                             fit_silhouette_xy)
from fit_check_frame import (CUP_H, EXO_MODEL, R_BOT, R_TOP,  # noqa: E402
                             find_marker_pose, world_from_camera)

# A track is "stationary" within a segment as long as consecutive fits stay
# inside this jump gate; segments shorter than MIN_SEG fits are ignored.
SEG_JUMP_M = 0.025
MIN_SEG = 10


def load_intrinsics(seq: Path, cam: str):
    import yaml
    d = yaml.safe_load((seq / f'{cam}_intrinsics.yaml').read_text())
    K = np.array(d['camera_matrix']['data'], dtype=np.float64).reshape(3, 3)
    dist = np.array(d['distortion_coefficients']['data'],
                    dtype=np.float64).reshape(-1)
    return K, dist


def segment_sigmas(xy_list):
    """Split a track's (x, y) series into stationary segments at >SEG_JUMP_M
    jumps; return per-axis sigma (m) pooled over segments with >= MIN_SEG."""
    segs, cur = [], []
    for xy in xy_list:
        if cur and math.hypot(xy[0] - cur[-1][0], xy[1] - cur[-1][1]) \
                > SEG_JUMP_M:
            segs.append(cur)
            cur = []
        cur.append(xy)
    segs.append(cur)
    devs = []
    n_used = 0
    for s in segs:
        if len(s) < MIN_SEG:
            continue
        a = np.asarray(s)
        devs.append(a - a.mean(axis=0))
        n_used += len(s)
    if not devs:
        return None, None, 0
    d = np.vstack(devs)
    return float(d[:, 0].std()), float(d[:, 1].std()), n_used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', required=True)
    ap.add_argument('--cam', default='exo',
                    help='exo only (hand needs FK; use the ROS replay for it)')
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--end', type=int, default=-1)
    ap.add_argument('--stride', type=int, default=3)
    ap.add_argument('--conf', type=float, default=0.35)
    # 1280 = what the live pipeline runs (params.yaml); 640 halves YOLO cost
    # but quantises the mask and roughly doubles the per-frame fit sigma.
    ap.add_argument('--imgsz', type=int, default=1280)
    ap.add_argument('--try-flip', action='store_true',
                    help='also fit with swapped radii (mouth-up) & keep best')
    ap.add_argument('--no-boundary-offset', action='store_true',
                    help='disable the 3rd fit param b (mask-bias absorption)')
    ap.add_argument('--no-edge-snap', action='store_true',
                    help='disable image-gradient boundary refinement')
    ap.add_argument('--csv', default='')
    ap.add_argument('--video', default='',
                    help='.avi → MJPEG (plays everywhere, no codec deps); '
                         '.mp4 → mp4v + ffmpeg H.264 re-encode')
    args = ap.parse_args()
    seq = Path(args.seq)
    repo = Path(__file__).resolve().parents[3]

    K, dist = load_intrinsics(seq, args.cam)
    fx, fy, cx_k, cy_k = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    R_cm, t_cm = find_marker_pose(seq, args.cam, K, dist, args.start)
    R_wc, t_wc = world_from_camera(R_cm, t_cm)
    print(f'[tf] world<-{args.cam} cam at '
          f'({t_wc[0]:+.3f},{t_wc[1]:+.3f},{t_wc[2]:+.3f}) m')

    from ultralytics import YOLO
    model = YOLO(str(repo / EXO_MODEL))
    upright_ids = [cid for cid, n in model.names.items()
                   if n == 'upright-cup']

    rgb_dir = seq / args.cam / 'rgb'
    n_total = len(list(rgb_dir.glob('*.png')))
    end = n_total if args.end < 0 else min(args.end, n_total)
    frames = range(args.start, end, max(1, args.stride))

    csv_f = writer = None
    if args.csv:
        csv_f = open(args.csv, 'w', newline='')
        writer = csv_mod.writer(csv_f)
        writer.writerow(['frame', 'track', 'conf', 'ok', 'fail',
                         'x', 'y', 'init_x', 'init_y', 'z_base0',
                         'iou', 'rms_px', 'd_init_mm', 'flipped',
                         'b_px', 'edge_cov', 'ms'])
    vw = None

    stats = defaultdict(lambda: {'fit': [], 'init': [], 'iou': [], 'rms': [],
                                 'fail': 0, 'n': 0, 'flip_better': 0})
    t_start = time.perf_counter()
    n_frames = 0
    for f in frames:
        rgb_p = rgb_dir / f'{f:06d}.png'
        dep_p = seq / args.cam / 'depth' / f'{f:06d}.png'
        if not rgb_p.is_file() or not dep_p.is_file():
            continue
        rgb = cv2.imread(str(rgb_p))
        z_img = cv2.imread(str(dep_p), cv2.IMREAD_UNCHANGED) \
            .astype(np.float32) * 0.001
        h, w = rgb.shape[:2]
        gray = (cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
                if not args.no_edge_snap else None)
        n_frames += 1

        r = model.track(rgb, conf=args.conf, imgsz=args.imgsz, verbose=False,
                        classes=upright_ids or None, persist=True)[0]
        overlay = rgb.copy() if (args.video or vw is not None) else None
        if r.masks is not None and r.boxes is not None:
            masks = r.masks.data.cpu().numpy()
            ids_t = getattr(r.boxes, 'id', None)
            ids = (ids_t.cpu().numpy().astype(int)
                   if ids_t is not None else None)
            for i in range(masks.shape[0]):
                tid = int(ids[i]) if ids is not None else -1
                if tid < 0:
                    continue
                conf = float(r.boxes.conf[i])
                m = cv2.resize(masks[i], (w, h),
                               interpolation=cv2.INTER_NEAREST)
                mask_u8 = (m > 0.5).astype(np.uint8) * 255
                sel = (mask_u8 > 0) & (z_img > 0.1) & (z_img < 4.0)
                st = stats[tid]
                st['n'] += 1
                if sel.sum() < 32:
                    st['fail'] += 1
                    if writer:
                        writer.writerow([f, tid, f'{conf:.2f}', 0,
                                         'few_depth'] + [''] * 12)
                    continue
                vy, vx = np.where(sel)
                z = z_img[vy, vx]
                pc = np.stack([(vx - cx_k) * z / fx,
                               (vy - cy_k) * z / fy, z], axis=1)
                pw = (R_wc @ pc.T).T + t_wc
                z_base0 = float(np.percentile(pw[:, 2], 97.0)) - CUP_H
                xy0 = (float(np.median(pw[:, 0])),
                       float(np.median(pw[:, 1])))
                t0 = time.perf_counter()
                bo = not args.no_boundary_offset
                fit = fit_silhouette_xy(
                    mask_u8, K=K, dist=dist, R_wc=R_wc, t_wc=t_wc,
                    r_top=R_TOP, r_bot=R_BOT, height=CUP_H,
                    z_base=z_base0, xy0=xy0, fit_boundary_offset=bo)
                if args.try_flip:
                    alt = fit_silhouette_xy(
                        mask_u8, K=K, dist=dist, R_wc=R_wc, t_wc=t_wc,
                        r_top=R_TOP, r_bot=R_BOT, height=CUP_H,
                        z_base=z_base0, xy0=xy0, flip_profile=True,
                        fit_boundary_offset=bo)
                    if alt['ok'] and (not fit['ok']
                                      or alt['iou'] > fit['iou']):
                        fit = alt
                        st['flip_better'] += 1
                edge_cov = 0.0
                if fit['ok'] and gray is not None and not fit['flipped']:
                    snap = edge_snap_fit(
                        gray, K=K, dist=dist, R_wc=R_wc, t_wc=t_wc,
                        r_top=R_TOP, r_bot=R_BOT, height=CUP_H,
                        z_base=z_base0, xy0=(fit['x'], fit['y']))
                    if snap['ok']:
                        fit = {**fit, 'x': snap['x'], 'y': snap['y'],
                               'rms_px': snap['rms_px']}
                        edge_cov = snap['edge_cov']
                ms = (time.perf_counter() - t0) * 1e3
                if not fit['ok']:
                    st['fail'] += 1
                    if writer:
                        writer.writerow([f, tid, f'{conf:.2f}', 0,
                                         fit['fail']] + [''] * 12)
                    continue
                d_init = 1e3 * math.hypot(fit['x'] - xy0[0],
                                          fit['y'] - xy0[1])
                st['fit'].append((fit['x'], fit['y']))
                st['init'].append(xy0)
                st['iou'].append(fit['iou'])
                st['rms'].append(fit['rms_px'])
                if writer:
                    writer.writerow([
                        f, tid, f'{conf:.2f}', 1, '',
                        f"{fit['x']:.4f}", f"{fit['y']:.4f}",
                        f'{xy0[0]:.4f}', f'{xy0[1]:.4f}', f'{z_base0:.4f}',
                        f"{fit['iou']:.3f}", f"{fit['rms_px']:.2f}",
                        f'{d_init:.1f}', int(fit['flipped']),
                        f"{fit.get('b_px', 0.0):.2f}", f'{edge_cov:.2f}',
                        f'{ms:.1f}'])
                if overlay is not None:
                    cnts, _ = cv2.findContours(
                        mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    cv2.drawContours(overlay, cnts, -1, (0, 255, 0), 1)
                    sil = cone_silhouette_px(
                        fit['x'], fit['y'], z_base0,
                        r_top=(R_BOT if fit['flipped'] else R_TOP),
                        r_bot=(R_TOP if fit['flipped'] else R_BOT),
                        height=CUP_H, K=K, dist=dist, R_wc=R_wc, t_wc=t_wc)
                    if sil is not None:
                        cv2.polylines(overlay,
                                      [np.round(sil).astype(np.int32)],
                                      True, (255, 255, 0), 2)
                    x1, y1 = int(r.boxes.xyxy[i][0]), int(r.boxes.xyxy[i][1])
                    tag = 'F' if fit['flipped'] else ''
                    cv2.putText(overlay,
                                f"#{tid}{tag} {fit['iou']:.2f} "
                                f"b{fit.get('b_px', 0.0):+.1f} "
                                f"c{edge_cov:.1f}",
                                (x1, max(12, y1 - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (255, 255, 0), 1)
        if overlay is not None:
            cv2.putText(overlay, f'frame {f:06d}', (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if vw is None and args.video:
                fps = max(1.0, 30.0 / max(1, args.stride))
                # .avi → MJPEG: decodes with stock gstreamer "good" plugins,
                # no H.264/MPEG-4 codec packs needed. Anything else → mp4v
                # (+ ffmpeg H.264 re-encode below when available).
                ext = Path(args.video).suffix.lower()
                fourcc = cv2.VideoWriter_fourcc(
                    *('MJPG' if ext == '.avi' else 'mp4v'))
                vw = cv2.VideoWriter(args.video, fourcc, fps, (w, h))
            if vw is not None:
                vw.write(overlay)

    if vw is not None:
        vw.release()
        # OpenCV writes MPEG-4 Part 2 ('mp4v'), which many players cannot
        # decode. Re-encode to H.264 yuv420p when ffmpeg is available.
        import shutil
        import subprocess
        if (shutil.which('ffmpeg')
                and Path(args.video).suffix.lower() != '.avi'):
            tmp = args.video + '.h264.mp4'
            rc = subprocess.run(
                ['ffmpeg', '-y', '-v', 'error', '-i', args.video,
                 '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                 '-movflags', '+faststart', tmp]).returncode
            if rc == 0:
                Path(tmp).replace(args.video)
            else:
                print('[out] ffmpeg re-encode failed — keeping mp4v')
        print(f'[out] video → {args.video}')
    if csv_f is not None:
        csv_f.close()
        print(f'[out] csv   → {args.csv}')
    wall = time.perf_counter() - t_start
    print(f'[run] {n_frames} frames in {wall:.1f}s '
          f'({n_frames / max(wall, 1e-9):.1f} fps)\n')

    print(f"{'trk':>4} {'n':>5} {'fail%':>6} {'iou':>5} {'rms':>5} "
          f"{'σx_fit':>7} {'σy_fit':>7} {'σx_init':>8} {'σy_init':>8} "
          f"{'flip':>5}  (mm; σ over stationary segments)")
    for tid in sorted(stats):
        st = stats[tid]
        if st['n'] < MIN_SEG:
            continue
        fail_pct = 100.0 * st['fail'] / st['n']
        sx_f, sy_f, n_f = segment_sigmas(st['fit'])
        sx_i, sy_i, _ = segment_sigmas(st['init'])
        if sx_f is None:
            print(f'{tid:>4} {st["n"]:>5} {fail_pct:6.1f} (no stationary seg)')
            continue
        print(f"{tid:>4} {st['n']:>5} {fail_pct:6.1f} "
              f"{np.mean(st['iou']):5.2f} {np.mean(st['rms']):5.2f} "
              f"{1e3 * sx_f:7.2f} {1e3 * sy_f:7.2f} "
              f"{1e3 * (sx_i or 0):8.2f} {1e3 * (sy_i or 0):8.2f} "
              f"{st['flip_better']:>5}")


if __name__ == '__main__':
    main()
