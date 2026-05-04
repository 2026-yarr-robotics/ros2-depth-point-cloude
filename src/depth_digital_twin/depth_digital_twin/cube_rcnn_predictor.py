"""Thin wrapper around Meta's Omni3D / Cube R-CNN model.

Cube R-CNN takes a single RGB image + camera intrinsics and returns 3D oriented
bounding boxes (centre, rotation, size) in the camera optical frame for every
detected object. Reference:
    https://github.com/facebookresearch/omni3d
    Brazil et al., "Omni3D: A Large Benchmark and Model for 3D Object
    Detection in the Wild", CVPR 2023.

Installation (run once on the target machine — heavy):
    # PyTorch first (match your CUDA / CPU)
    pip install --user torch torchvision

    # Detectron2 (Cube R-CNN's framework)
    pip install --user 'git+https://github.com/facebookresearch/detectron2.git'

    # Omni3D (Cube R-CNN)
    git clone https://github.com/facebookresearch/omni3d
    cd omni3d
    pip install --user -r requirements.txt
    pip install --user -e .

    # Pretrained weights — pick one from MODEL_ZOO.md, e.g.:
    wget https://dl.fbaipublicfiles.com/cubercnn/omni3d/cubercnn_DLA34_FPN.pth

The wrapper imports lazily so the rest of the package stays importable on
machines where Cube R-CNN is not installed.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _ensure_cubercnn_on_sys_path() -> None:
    """Make sure the Omni3D clone is on sys.path before import.

    The installer (`dependence/setup.bash`) drops a `omni3d.pth` file into the
    venv site-packages so plain `import cubercnn` works, but ROS 2 launch can
    spawn a child Python process whose site initialisation does not always pick
    up that .pth (PYTHONPATH overlays from the workspace setup.bash can cause
    it to be skipped). Resolve the path explicitly here as a safety net.
    """
    # 1) Already importable? Nothing to do.
    try:
        import cubercnn  # noqa: F401
        return
    except ImportError:
        pass

    candidates: list[str] = []
    # 2) Env var exported by dependence/activate.bash
    deps = os.environ.get('DEPTH_DIGITAL_TWIN_DEPS', '')
    if deps:
        candidates.append(os.path.join(deps, 'src', 'omni3d'))
    # 3) Override via dedicated env var if the user wants
    omni3d_dir = os.environ.get('OMNI3D_DIR', '')
    if omni3d_dir:
        candidates.insert(0, omni3d_dir)
    # 4) Check the .pth file we wrote during setup.bash, regardless of where
    #    the current Python interpreter looks for site-packages.
    for site in sys.path:
        pth = os.path.join(site, 'omni3d.pth')
        if os.path.isfile(pth):
            try:
                with open(pth) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            candidates.append(line)
            except OSError:
                pass
    # 5) Fall back to a sibling clone next to this workspace
    here = Path(__file__).resolve()
    for parent in here.parents:
        guess = parent / 'dependence' / 'src' / 'omni3d'
        if guess.is_dir():
            candidates.append(str(guess))
            break

    seen: set[str] = set()
    for c in candidates:
        if not c or c in seen or not os.path.isdir(c):
            continue
        seen.add(c)
        if c not in sys.path:
            sys.path.insert(0, c)


_ensure_cubercnn_on_sys_path()


@dataclass
class CubeDetection:
    class_id: int
    class_name: str
    score: float
    bbox2d: tuple[int, int, int, int]      # (x_min, y_min, x_max, y_max)
    center_cam: np.ndarray                  # (3,) m, in camera optical frame
    R_cam: np.ndarray                       # (3, 3) box-local axes -> camera optical
    size: np.ndarray                        # (3,) full extents along the box's local axes


class CubeRcnnPredictor:
    """Lazy wrapper. Construct it once; call `predict(bgr, K)` per frame.

    Falls back gracefully when the runtime is missing — the constructor raises
    a ``RuntimeError`` that points at the install instructions above.
    """

    def __init__(self,
                 config_file: str,
                 weights: str,
                 device: str = 'cuda',
                 score_threshold: float = 0.25,
                 categories: list[str] | None = None) -> None:
        self.config_file = config_file
        self.weights = weights
        self.device = device
        self.score_threshold = float(score_threshold)
        self.target_categories: set[str] | None = (
            {s.lower() for s in categories} if categories else None)

        # ---- Lazy imports (heavy + optional) ------------------------------
        try:
            import torch  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                'PyTorch is not installed; required by Cube R-CNN. See module '
                'docstring for setup instructions.') from e
        try:
            from detectron2.config import get_cfg  # type: ignore
            from detectron2.checkpoint import DetectionCheckpointer  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                'detectron2 is not installed; required by Cube R-CNN.') from e
        try:
            from cubercnn.config import get_cfg_defaults  # type: ignore
            from cubercnn.modeling.meta_arch import build_model  # type: ignore
            # Ensure custom modules are registered before model build.
            import cubercnn.modeling.proposal_generator  # noqa: F401
            import cubercnn.modeling.roi_heads          # noqa: F401
            import cubercnn.modeling.backbone           # noqa: F401
            import cubercnn.modeling.meta_arch          # noqa: F401
            from cubercnn import util as cubercnn_util  # type: ignore
        except ImportError as e:
            # Diagnostic dump so we can see what sys.path looked like.
            tail = '\n  '.join(sys.path[-8:])
            raise RuntimeError(
                'cubercnn (Omni3D) is not importable from this Python '
                f'(executable={sys.executable}). sys.path tail:\n  {tail}\n'
                'Run ./dependence/setup.bash and source dependence/activate.bash, '
                'or set OMNI3D_DIR=/path/to/omni3d before launching.'
            ) from e

        self._torch = torch

        # ---- Build config + model ----------------------------------------
        cfg = get_cfg()
        get_cfg_defaults(cfg)
        if not Path(config_file).is_file():
            raise FileNotFoundError(f'Cube R-CNN config not found: {config_file}')
        cfg.merge_from_file(config_file)
        cfg.MODEL.WEIGHTS = weights
        cfg.MODEL.DEVICE = device
        cfg.freeze()
        self.cfg = cfg

        self.model = build_model(cfg)
        self.model.eval()
        DetectionCheckpointer(self.model).load(weights)

        # Class id -> name. Cube R-CNN stores category metadata in MetadataCatalog
        # for the dataset; fall back to indices if metadata is missing.
        try:
            from detectron2.data import MetadataCatalog  # type: ignore
            ds_name = cfg.DATASETS.TEST[0] if cfg.DATASETS.TEST else None
            md = MetadataCatalog.get(ds_name) if ds_name else None
            self.class_names: list[str] = list(md.thing_classes) if md else []
        except Exception:
            self.class_names = []

        # Quiet ref to util (used elsewhere).
        self._cubercnn_util = cubercnn_util

    # ----------------------------------------------------------------------
    def predict(self, bgr: np.ndarray, K: np.ndarray) -> list[CubeDetection]:
        """Run inference on a single BGR frame with the supplied intrinsic K."""
        torch = self._torch
        h, w = bgr.shape[:2]
        # Cube R-CNN expects a CHW float tensor in BGR (detectron2 default).
        image_tensor = torch.as_tensor(
            bgr.astype('float32').transpose(2, 0, 1).copy())
        K_t = torch.as_tensor(np.asarray(K, dtype=np.float32))

        batched = [{
            'image': image_tensor,
            'height': h,
            'width': w,
            'K': K_t,
        }]
        with torch.no_grad():
            outputs = self.model(batched)

        if not outputs or 'instances' not in outputs[0]:
            return []
        inst = outputs[0]['instances'].to('cpu')
        n = len(inst)
        if n == 0:
            return []

        scores = inst.scores.numpy()
        classes = inst.pred_classes.numpy().astype(int)
        boxes2d = inst.pred_boxes.tensor.numpy()  # (N, 4) xyxy
        # Required 3D fields (Omni3D adds these to Instances).
        centers = inst.pred_center_cam.numpy()    # (N, 3)
        dims = inst.pred_dimensions.numpy()       # (N, 3) full extents
        poses = inst.pred_pose.numpy()            # (N, 3, 3) rotation matrices

        results: list[CubeDetection] = []
        for i in range(n):
            if scores[i] < self.score_threshold:
                continue
            cls_id = int(classes[i])
            cls_name = (self.class_names[cls_id]
                        if 0 <= cls_id < len(self.class_names) else str(cls_id))
            if (self.target_categories is not None
                    and cls_name.lower() not in self.target_categories):
                continue
            x_min, y_min, x_max, y_max = boxes2d[i].tolist()
            results.append(CubeDetection(
                class_id=cls_id,
                class_name=cls_name,
                score=float(scores[i]),
                bbox2d=(int(round(x_min)), int(round(y_min)),
                        int(round(x_max)), int(round(y_max))),
                center_cam=np.asarray(centers[i], dtype=np.float64),
                R_cam=np.asarray(poses[i], dtype=np.float64),
                size=np.asarray(dims[i], dtype=np.float64),
            ))
        return results


def rotation_matrix_to_quaternion(R: np.ndarray) -> tuple[float, float, float, float]:
    """3x3 rotation → (x, y, z, w). Numerically stable variant."""
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    tr = m00 + m11 + m22
    if tr > 0.0:
        s = float(np.sqrt(tr + 1.0)) * 2.0
        qw = 0.25 * s
        qx = (m21 - m12) / s
        qy = (m02 - m20) / s
        qz = (m10 - m01) / s
    elif (m00 > m11) and (m00 > m22):
        s = float(np.sqrt(1.0 + m00 - m11 - m22)) * 2.0
        qw = (m21 - m12) / s
        qx = 0.25 * s
        qy = (m01 + m10) / s
        qz = (m02 + m20) / s
    elif m11 > m22:
        s = float(np.sqrt(1.0 + m11 - m00 - m22)) * 2.0
        qw = (m02 - m20) / s
        qx = (m01 + m10) / s
        qy = 0.25 * s
        qz = (m12 + m21) / s
    else:
        s = float(np.sqrt(1.0 + m22 - m00 - m11)) * 2.0
        qw = (m10 - m01) / s
        qx = (m02 + m20) / s
        qy = (m12 + m21) / s
        qz = 0.25 * s
    return float(qx), float(qy), float(qz), float(qw)


def points_inside_obb(points_cam: np.ndarray,
                      center: np.ndarray,
                      R: np.ndarray,
                      size: np.ndarray,
                      pad: float = 0.0) -> np.ndarray:
    """Boolean mask of points (N, 3) lying inside an oriented box.

    `R` columns are the box's local axes expressed in the camera frame (i.e.
    R @ p_local + center == p_cam). `size` is the full extent along each axis.
    """
    half = size * 0.5 + pad
    local = (points_cam - center) @ R  # equivalent to R.T @ (p - c) per row
    return np.all(np.abs(local) <= half, axis=1)


def obb_corners(center: np.ndarray, R: np.ndarray, size: np.ndarray) -> np.ndarray:
    """Return the 8 corners of an OBB in the camera frame, ordered for line-list."""
    h = size * 0.5
    s = np.array([
        [-1, -1, -1], [+1, -1, -1], [+1, +1, -1], [-1, +1, -1],
        [-1, -1, +1], [+1, -1, +1], [+1, +1, +1], [-1, +1, +1],
    ], dtype=np.float64) * h
    return (R @ s.T).T + center
