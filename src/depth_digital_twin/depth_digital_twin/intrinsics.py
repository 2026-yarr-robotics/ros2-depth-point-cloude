"""Camera intrinsics: dataclass + YAML I/O + pixel↔3D deprojection helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml


@dataclass
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    dist: np.ndarray  # shape (5,) — k1,k2,p1,p2,k3
    rms: float = 0.0

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]], dtype=np.float64)

    def deproject(self, u: float, v: float, z: float) -> np.ndarray:
        """Pixel (u,v) + depth z (metres) → 3D point in camera optical frame."""
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        return np.array([x, y, z], dtype=np.float64)

    def deproject_array(self, uv: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Vectorised deprojection.
        uv: (N,2) pixel coords; z: (N,) metres. Returns (N,3) in camera frame."""
        u = uv[:, 0]
        v = uv[:, 1]
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        return np.stack([x, y, z], axis=1)


def save_intrinsics(path: Path, intr: Intrinsics) -> None:
    data = {
        'image_width': int(intr.width),
        'image_height': int(intr.height),
        'camera_matrix': {
            'rows': 3, 'cols': 3,
            'data': [intr.fx, 0.0, intr.cx,
                     0.0, intr.fy, intr.cy,
                     0.0, 0.0, 1.0],
        },
        'distortion_coefficients': {
            'rows': 1, 'cols': int(intr.dist.size),
            'data': [float(v) for v in intr.dist.flatten()],
        },
        'reprojection_error': float(intr.rms),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        yaml.safe_dump(data, f, sort_keys=False)


def load_intrinsics(path: Path) -> Intrinsics:
    with Path(path).open('r') as f:
        d = yaml.safe_load(f)
    K = np.array(d['camera_matrix']['data'], dtype=np.float64).reshape(3, 3)
    dist = np.array(d['distortion_coefficients']['data'], dtype=np.float64).flatten()
    return Intrinsics(
        width=int(d['image_width']),
        height=int(d['image_height']),
        fx=float(K[0, 0]), fy=float(K[1, 1]),
        cx=float(K[0, 2]), cy=float(K[1, 2]),
        dist=dist,
        rms=float(d.get('reprojection_error', 0.0)),
    )


def from_calibration(width: int, height: int, K: np.ndarray,
                     dist: Iterable[float], rms: float) -> Intrinsics:
    K = np.asarray(K, dtype=np.float64)
    return Intrinsics(
        width=int(width), height=int(height),
        fx=float(K[0, 0]), fy=float(K[1, 1]),
        cx=float(K[0, 2]), cy=float(K[1, 2]),
        dist=np.asarray(list(dist), dtype=np.float64).flatten(),
        rms=float(rms),
    )
