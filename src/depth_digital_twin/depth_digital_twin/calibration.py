"""Compute camera intrinsics from chessboard images and save as YAML.

Mirrors `sample/Calibration_Tutorial/handeye_calibration.py:calibrate_camera_from_chessboard`.
"""
from __future__ import annotations

import argparse
import sys
from glob import glob
from pathlib import Path

import cv2
import numpy as np

from depth_digital_twin.intrinsics import from_calibration, save_intrinsics


def calibrate(image_paths: list[str],
              board_size: tuple[int, int],
              square_size_mm: float):
    cols, rows = board_size
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size_mm

    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []
    image_shape: tuple[int, int] | None = None
    used: list[str] = []
    skipped: list[str] = []

    for fname in image_paths:
        img = cv2.imread(fname)
        if img is None:
            skipped.append(f'{fname} (unreadable)')
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_shape is None:
            image_shape = gray.shape[::-1]  # (W, H)

        # Robust SB detector first; fall back to legacy + cornerSubPix.
        found, corners = cv2.findChessboardCornersSB(
            gray, (cols, rows),
            flags=cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_EXHAUSTIVE)
        if not found:
            f2, c2 = cv2.findChessboardCorners(
                gray, (cols, rows),
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
            if f2:
                corners = cv2.cornerSubPix(
                    gray, c2, (11, 11), (-1, -1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                     30, 0.001))
                found = True
        if not found:
            skipped.append(f'{fname} (corners not found)')
            continue

        obj_points.append(objp)
        img_points.append(corners)
        used.append(fname)

    if len(obj_points) < 3:
        raise RuntimeError(
            f'Not enough valid images: {len(obj_points)} (need >= 3, ideally 20+)')

    rms, K, dist, *_ = cv2.calibrateCamera(
        obj_points, img_points, image_shape, None, None)
    if not rms:
        raise RuntimeError('cv2.calibrateCamera did not converge')

    return rms, K, dist, image_shape, used, skipped


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=str, default='data/chess_*.png',
                        help='Glob pattern for input images (default: data/chess_*.png)')
    parser.add_argument('--board', type=str, default='10x7',
                        help='Inner corners as CxR (default: 10x7)')
    parser.add_argument('--square', type=float, default=25.0,
                        help='Square size in mm (default: 25.0)')
    parser.add_argument('--output', type=Path,
                        default=Path('src/depth_digital_twin/config/intrinsics.yaml'),
                        help='Output YAML path')
    parser.add_argument('--undistort-preview', type=Path, default=None,
                        help='If set, write a side-by-side preview to this path')
    args = parser.parse_args(argv)

    cols, rows = (int(x) for x in args.board.lower().split('x'))
    paths = sorted(glob(args.images))
    if not paths:
        print(f'[ERROR] No images matched: {args.images}', file=sys.stderr)
        sys.exit(2)

    print(f'[INFO] {len(paths)} candidate images, board={cols}x{rows}, square={args.square}mm')
    rms, K, dist, (W, H), used, skipped = calibrate(paths, (cols, rows), args.square)

    print('\n===== Calibration Result =====')
    print(f'Used    : {len(used)} / {len(paths)} images')
    if skipped:
        print(f'Skipped : {len(skipped)}')
        for s in skipped:
            print(f'  - {s}')
    print(f'Image   : {W} x {H}')
    print(f'fx, fy  : {K[0, 0]:.3f}, {K[1, 1]:.3f}')
    print(f'cx, cy  : {K[0, 2]:.3f}, {K[1, 2]:.3f}')
    print(f'dist    : {dist.flatten()}')
    print(f'RMS     : {rms:.4f} px  ({"OK" if rms < 1.0 else "HIGH — recapture"})')

    intr = from_calibration(W, H, K, dist.flatten(), rms)
    save_intrinsics(args.output, intr)
    print(f'\n[INFO] Wrote {args.output}')

    if args.undistort_preview and used:
        sample = cv2.imread(used[0])
        und = cv2.undistort(sample, K, dist)
        side = np.hstack([sample, und])
        args.undistort_preview.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.undistort_preview), side)
        print(f'[INFO] Wrote preview {args.undistort_preview}')


if __name__ == '__main__':
    main(sys.argv[1:])
