"""Offline ArUco hand-eye calibration solver.

Pairs with `aruco_calibrate` (capture). Reads the JSON + PNG dataset that
capture writes, runs ArUco detection + solvePnP on each image, and feeds
the (T_base→ee, T_cam→marker) sample list into cv2.calibrateHandEye to
produce `T_exo2base.npy` (eye-to-hand) or `T_hand2base.npy` (eye-in-hand).

Mirrors the chessboard workflow: `capture_chessboard` (capture) then
`calibrate` (offline) → `intrinsics.yaml`. Here:
  `aruco_calibrate` (capture) → `data/aruco/...`
  `aruco_handeye`   (offline)  → `T_exo2base.npy`

The dataset layout matches `sample/Calibration_Tutorial/data_recording.py`:

  <data_dir>/
    calibrate_data.json     # {"poses": [[x,y,z,rx,ry,rz], ...], "file_name": [...]}
    aruco_000.png
    aruco_001.png
    ...

Robot poses are in Doosan's convention: translation in **mm**, rotation as
**ZYZ Euler in degrees** (matches `posx`).

Usage:
  ros2 run depth_digital_twin aruco_handeye \\
      --data-dir   ./data/aruco \\
      --intrinsics src/depth_digital_twin/config/intrinsics.yaml \\
      --output     src/depth_digital_twin/config/T_exo2base.npy \\
      --mode       exo \\
      --marker-length 0.05 \\
      --dict       4X4_50 \\
      --marker-id  0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from depth_digital_twin.intrinsics import load_intrinsics
from depth_digital_twin.aruco_calibrate import (
    _ARUCO_DICT_NAMES,
    _calibrate_exo,
    _calibrate_hand,
    _detect_marker,
    _make_aruco_detector,
    _solve_marker_pose,
)


def _load_dataset(data_dir: Path) -> tuple[list[str], list[list[float]]]:
    json_path = data_dir / 'calibrate_data.json'
    if not json_path.is_file():
        raise FileNotFoundError(
            f'No calibrate_data.json under {data_dir}. Run aruco_calibrate '
            'capture first.')
    with json_path.open('r') as f:
        data = json.load(f)
    files = list(data.get('file_name', []))
    poses = list(data.get('poses', []))
    if len(files) != len(poses):
        n = min(len(files), len(poses))
        print(f'WARN: file_name ({len(files)}) ≠ poses ({len(poses)}); '
              f'truncating to first {n}.', file=sys.stderr)
        files = files[:n]
        poses = poses[:n]
    return files, poses


def _detect_and_solve(img_path: Path, detector, dictionary,
                      marker_length: float, marker_id: int,
                      K: np.ndarray, dist: np.ndarray):
    """Returns (R_cam_marker, t_cam_marker_m, reason_or_None)."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None, None, 'cv2.imread returned None'
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids = _detect_marker(detector, dictionary, gray)
    if ids is None or len(ids) == 0:
        return None, None, 'no marker detected'
    picked = None
    if marker_id < 0:
        picked = corners[0]
    else:
        for i, mid in enumerate(ids.flatten().tolist()):
            if int(mid) == marker_id:
                picked = corners[i]
                break
    if picked is None:
        seen = ','.join(str(int(v)) for v in ids.flatten().tolist())
        return None, None, f'marker id={marker_id} missing (seen=[{seen}])'
    R_cm, t_cm = _solve_marker_pose(picked, marker_length, K, dist)
    if R_cm is None:
        return None, None, 'solvePnP failed'
    return R_cm, t_cm, None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', type=Path, required=True,
                        help='Capture output directory '
                             '(contains calibrate_data.json + aruco_NNN.png)')
    parser.add_argument('--intrinsics', type=Path, required=True,
                        help='YAML produced by `calibrate` CLI')
    parser.add_argument('--output', type=Path, required=True,
                        help='Output .npy path. Naming convention: '
                             'T_exo2base*.npy (exo) or T_hand2base*.npy (hand)')
    parser.add_argument('--mode', choices=['exo', 'hand'], default='exo',
                        help='exo = eye-to-hand, hand = eye-in-hand')
    parser.add_argument('--marker-length', type=float, required=True,
                        help='ArUco side length in METRES')
    parser.add_argument('--dict', dest='aruco_dict', default='4X4_50',
                        help=f'ArUco dictionary; '
                             f'choices: {sorted(_ARUCO_DICT_NAMES)}')
    parser.add_argument('--marker-id', type=int, default=0,
                        help='Marker id (default 0; -1 = first detected)')
    parser.add_argument('--min-samples', type=int, default=10,
                        help='Refuse to compute if fewer valid samples')
    args = parser.parse_args(argv)

    intr = load_intrinsics(args.intrinsics)
    K = intr.K
    dist = intr.dist.reshape(-1, 1)

    detector, dictionary = _make_aruco_detector(args.aruco_dict)
    files, poses = _load_dataset(args.data_dir)
    if not files:
        print(f'No samples found in {args.data_dir}.', file=sys.stderr)
        sys.exit(1)

    samples: list[dict] = []
    failures: list[tuple[str, str]] = []
    for fname, posx in zip(files, poses):
        R_cm, t_cm, why = _detect_and_solve(
            args.data_dir / fname, detector, dictionary,
            args.marker_length, args.marker_id, K, dist)
        if R_cm is None:
            failures.append((fname, why))
            continue
        samples.append({
            'posx': tuple(float(v) for v in posx),
            'R_cam_marker': R_cm,
            't_cam_marker_m': t_cm,
        })

    print(f'Detected ArUco in {len(samples)}/{len(files)} images '
          f'({len(failures)} skipped).')
    for fname, why in failures:
        print(f'  - skip {fname}: {why}')

    if len(samples) < args.min_samples:
        print(f'\nNeed at least {args.min_samples} valid samples '
              f'(have {len(samples)}). Aborting.', file=sys.stderr)
        sys.exit(1)

    if args.mode == 'exo':
        T = _calibrate_exo(samples)
        label = 'T_exo2base'
    else:
        T = _calibrate_hand(samples)
        label = 'T_hand2base'

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, T)

    print()
    print('=' * 60)
    print(f'{label} computed from {len(samples)} samples '
          f'({args.mode} mode, dict={args.aruco_dict}, '
          f'marker_id={args.marker_id})')
    print(f'R =')
    for row in T[:3, :3]:
        print(f'    {row.tolist()}')
    print(f't (mm) = ({T[0,3]:.3f}, {T[1,3]:.3f}, {T[2,3]:.3f})')
    print(f'SAVED → {args.output}')
    print('=' * 60)


if __name__ == '__main__':
    main(sys.argv[1:])
