"""seq_io — on-disk sequence format shared by the recorder and the player.

Layout of one sequence (``output_root/0001/``)::

    0001/
    ├── meta.json                 # cameras, intrinsics, robot/units, n_steps
    ├── exo/  rgb/000000.png …     # 8-bit BGR
    │         depth/000000.png …   # 16-bit, millimetres (RealSense Z16)
    ├── hand/ rgb/000000.png …
    │         depth/000000.png …
    └── trajectory.pkl            # per-step robot state + frame references

Every stream shares ONE step index: step *i* of exo, hand and the robot
trajectory were all sampled on the same recorder tick, so the timeline is
identical across streams (a hard requirement of the spec). A missing sample
for a given step is stored as ``None`` rather than shifting the index.
"""
from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np

META_NAME = 'meta.json'
TRAJ_NAME = 'trajectory.pkl'
FORMAT_VERSION = 1


def next_sequence_id(output_root: str) -> str:
    """Return the next zero-padded folder name (``0001`` …) under root."""
    os.makedirs(output_root, exist_ok=True)
    used = []
    for name in os.listdir(output_root):
        if name.isdigit() and os.path.isdir(os.path.join(output_root, name)):
            used.append(int(name))
    return f'{(max(used) + 1) if used else 1:04d}'


def list_sequences(output_root: str) -> list[str]:
    if not os.path.isdir(output_root):
        return []
    return sorted(
        os.path.join(output_root, n)
        for n in os.listdir(output_root)
        if n.isdigit() and os.path.isdir(os.path.join(output_root, n)))


def _frame_rel(view: str, kind: str, step: int) -> str:
    return f'{view}/{kind}/{step:06d}.png'


@dataclass
class CameraMeta:
    serial: str = ''
    width: int = 0
    height: int = 0
    K: list[float] = field(default_factory=list)        # row-major 3x3 (len 9)
    dist: list[float] = field(default_factory=list)
    frame_id: str = ''

    def to_dict(self) -> dict[str, Any]:
        return {
            'serial': self.serial, 'width': self.width,
            'height': self.height, 'K': self.K, 'dist': self.dist,
            'frame_id': self.frame_id,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> 'CameraMeta':
        return CameraMeta(
            serial=d.get('serial', ''), width=int(d.get('width', 0)),
            height=int(d.get('height', 0)), K=list(d.get('K', [])),
            dist=list(d.get('dist', [])), frame_id=d.get('frame_id', ''))


class SequenceWriter:
    """Create a new sequence folder and stream steps into it."""

    def __init__(self, output_root: str, record_rate_hz: float,
                 robot_meta: dict[str, Any]) -> None:
        self.seq_id = next_sequence_id(output_root)
        self.root = os.path.join(output_root, self.seq_id)
        for view in ('exo', 'hand'):
            for kind in ('rgb', 'depth'):
                os.makedirs(os.path.join(self.root, view, kind),
                            exist_ok=True)
        self.record_rate_hz = float(record_rate_hz)
        self.robot_meta = dict(robot_meta)
        self.cameras: dict[str, CameraMeta] = {
            'exo': CameraMeta(), 'hand': CameraMeta()}
        self._steps: list[dict[str, Any]] = []
        self._t0: float | None = None

    # -- per-step -----------------------------------------------------------
    # Lossless PNG but cheap compression effort: ~3-4x faster to encode
    # than the default level (3) while staying bit-exact. Crucial to keep
    # up with a 30 Hz x 4-image stream.
    _PNG = [cv2.IMWRITE_PNG_COMPRESSION, 1]

    def add_step(self, *, t_wall: float,
                  exo_rgb: np.ndarray | None, exo_depth: np.ndarray | None,
                  hand_rgb: np.ndarray | None, hand_depth: np.ndarray | None,
                  joint_names: list[str] | None,
                  joint_pos: list[float] | None,
                  ee_tcp: list[float] | None,
                  ee_flange: list[float] | None
                  ) -> tuple[int, list[tuple[str, np.ndarray, bool]]]:
        """Append the step's trajectory record and RETURN image-write jobs.

        No disk I/O happens here — the caller writes the returned
        ``(abs_path, image, is_depth)`` jobs (typically on worker threads)
        via :meth:`save_image`. The step index / timeline is fixed at
        call time, so playback order is correct regardless of when the
        image bytes actually land on disk.
        """
        step = len(self._steps)
        if self._t0 is None:
            self._t0 = t_wall
        rec: dict[str, Any] = {
            'step': step,
            't_wall': float(t_wall),
            't_rel': float(t_wall - self._t0),
            'joint_names': list(joint_names) if joint_names else None,
            'joint_pos': [float(v) for v in joint_pos] if joint_pos else None,
            'ee_tcp': [float(v) for v in ee_tcp] if ee_tcp else None,
            'ee_flange':
                [float(v) for v in ee_flange] if ee_flange else None,
        }
        jobs: list[tuple[str, np.ndarray, bool]] = []
        for view, rgb, depth in (('exo', exo_rgb, exo_depth),
                                 ('hand', hand_rgb, hand_depth)):
            if rgb is not None:
                p = _frame_rel(view, 'rgb', step)
                jobs.append((os.path.join(self.root, p), rgb, False))
                rec[f'{view}_rgb'] = p
            else:
                rec[f'{view}_rgb'] = None
            if depth is not None:
                p = _frame_rel(view, 'depth', step)
                jobs.append((os.path.join(self.root, p), depth, True))
                rec[f'{view}_depth'] = p
            else:
                rec[f'{view}_depth'] = None
        self._steps.append(rec)
        return step, jobs

    @staticmethod
    def save_image(abs_path: str, img: np.ndarray, is_depth: bool) -> None:
        """Encode + write one frame (call from a worker thread)."""
        if is_depth:
            # 16-bit single-channel PNG preserves millimetre Z16 exactly.
            cv2.imwrite(abs_path, img.astype(np.uint16),
                        SequenceWriter._PNG)
        else:
            cv2.imwrite(abs_path, img, SequenceWriter._PNG)

    def write_step(self, **kw) -> int:
        """Synchronous convenience (tests / non-realtime use)."""
        step, jobs = self.add_step(**kw)
        for path, img, is_depth in jobs:
            self.save_image(path, img, is_depth)
        return step

    def set_camera(self, view: str, meta: CameraMeta) -> None:
        self.cameras[view] = meta

    @property
    def n_steps(self) -> int:
        return len(self._steps)

    # -- finalise -----------------------------------------------------------
    def _persist(self) -> None:
        meta = {
            'version': FORMAT_VERSION,
            'created': datetime.now(timezone.utc).isoformat(),
            'record_rate_hz': self.record_rate_hz,
            'n_steps': len(self._steps),
            'cameras': {k: v.to_dict() for k, v in self.cameras.items()},
            'robot': self.robot_meta,
        }
        # Atomic-ish: write to tmp then replace, so a kill mid-write never
        # leaves a truncated meta/trajectory.
        mp = os.path.join(self.root, META_NAME)
        with open(mp + '.tmp', 'w') as f:
            json.dump(meta, f, indent=2)
        os.replace(mp + '.tmp', mp)
        tp = os.path.join(self.root, TRAJ_NAME)
        with open(tp + '.tmp', 'wb') as f:
            pickle.dump({'meta': meta, 'steps': self._steps}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tp + '.tmp', tp)

    def checkpoint(self) -> None:
        """Persist meta+trajectory mid-recording (crash/kill safety)."""
        self._persist()

    def close(self) -> None:
        self._persist()


class SequenceReader:
    """Read a sequence folder; lazily decode frame images by step index."""

    def __init__(self, sequence_dir: str) -> None:
        self.root = os.path.abspath(sequence_dir)
        if not os.path.isdir(self.root):
            raise FileNotFoundError(f'sequence dir not found: {self.root}')
        with open(os.path.join(self.root, META_NAME)) as f:
            self.meta: dict[str, Any] = json.load(f)
        with open(os.path.join(self.root, TRAJ_NAME), 'rb') as f:
            self._traj = pickle.load(f)
        self.steps: list[dict[str, Any]] = self._traj['steps']
        self.cameras = {
            k: CameraMeta.from_dict(v)
            for k, v in self.meta.get('cameras', {}).items()}

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def record_rate_hz(self) -> float:
        return float(self.meta.get('record_rate_hz', 30.0))

    def robot_meta(self) -> dict[str, Any]:
        return self.meta.get('robot', {})

    def step_record(self, i: int) -> dict[str, Any]:
        return self.steps[max(0, min(i, self.n_steps - 1))]

    def _imread(self, rel: str | None, *, depth: bool) -> np.ndarray | None:
        if not rel:
            return None
        path = os.path.join(self.root, rel)
        flag = cv2.IMREAD_UNCHANGED if depth else cv2.IMREAD_COLOR
        img = cv2.imread(path, flag)
        return img

    def frame(self, i: int, view: str, kind: str) -> np.ndarray | None:
        """``view`` ∈ {exo,hand}, ``kind`` ∈ {rgb,depth}."""
        rec = self.step_record(i)
        return self._imread(rec.get(f'{view}_{kind}'), depth=kind == 'depth')
