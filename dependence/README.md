# dependence/

Self-contained installer for the heavy ML dependencies needed by the Cube R-CNN
pipeline of `depth_digital_twin` (PyTorch + detectron2 + pytorch3d + Omni3D).
Everything is installed under this folder so it never pollutes the system.

## What gets installed

```
dependence/
├── setup.bash       ← installer (idempotent)
├── activate.bash    ← source this in every new terminal
├── clean.bash       ← wipe everything to redo setup
├── venv/            ← Python venv (created)
├── src/             ← cloned source repos
│   ├── detectron2/
│   ├── pytorch3d/
│   └── omni3d/
├── models/          ← pretrained weights
│   └── cubercnn_DLA34_FPN.pth
└── logs/            ← per-step build logs
```

The venv is created with `--system-site-packages` so ROS 2's Python modules
(`rclpy`, `cv_bridge`, etc.) remain importable.

## Install

```bash
# (one-time, ~30–60 min depending on machine)
./dependence/setup.bash
```

Re-running is safe — completed steps are skipped via probe imports.

Optional environment overrides:

```bash
PYTHON_BIN=python3.10 \
CUDA_TAG=cu118 \                # cu121 | cu118 | cpu  (auto-detected if unset)
MAX_JOBS=2 \                    # lower if pytorch3d build runs out of memory
./dependence/setup.bash
```

## Use

In every new terminal where you want to launch ROS 2 nodes that need cubercnn:

```bash
source dependence/activate.bash
```

This sources, in order:
1. `/opt/ros/humble/setup.bash` (if not already loaded)
2. `dependence/venv/bin/activate`
3. `<workspace>/install/setup.bash` (if it exists)
4. Sets `CUBERCNN_CONFIG` and `CUBERCNN_WEIGHTS` env vars.

Then launch the Cube R-CNN pipeline:

```bash
ros2 launch depth_digital_twin cube_rcnn.launch.py \
  config_file:=$CUBERCNN_CONFIG \
  weights:=$CUBERCNN_WEIGHTS \
  device:=cuda
```

## Reset

```bash
./dependence/clean.bash            # confirms before deleting
./dependence/clean.bash --force    # no prompt
```

## What setup.bash actually does (matches the README guide step by step)

| Step | Purpose | Probe used to skip |
|------|---------|--------------------|
| 1 | NumPy < 2 (cv_bridge ABI) | `import numpy; numpy.__version__.startswith('1.')` |
| 2 | PyTorch 2.1.2 + Torchvision 0.16.2 (cu121 / cu118 / cpu) | `import torch; torch.__version__.startswith('2.1.')` |
| 3 | iopath, fvcore | `import iopath`, `import fvcore` |
| 4 | detectron2 (source build) | `import detectron2` |
| 5 | pytorch3d (source build) | `import pytorch3d` |
| 6 | opencv-python, scipy, seaborn, cython, cocoapi, pyyaml, ultralytics | per module |
| 7 | omni3d (cubercnn) editable install with `--no-deps` | `from cubercnn.config import get_cfg_defaults` |
| 8 | Cube R-CNN pretrained weights download | file presence check |
| 9 | Smoke import test of every module above | runtime check |

## Troubleshooting

- **`ModuleNotFoundError: No module named 'torch'` while building pytorch3d /
  detectron2** — caused by PEP 517 build isolation. setup.bash now passes
  `--no-build-isolation` for both source-built packages. If you ran an older
  copy of setup.bash, just pull the latest setup.bash and re-run — the
  earlier successful steps (torch, iopath, etc.) are skipped automatically.

- **`NVIDIA GeForce RTX 5080/5090 with CUDA capability sm_120 is not
  compatible with the current PyTorch installation`** — Blackwell GPUs need
  PyTorch ≥ 2.7 + CUDA 12.8 wheels. setup.bash auto-detects this via
  `nvidia-smi --query-gpu=compute_cap`. If it picked the wrong tag, force it:
  ```bash
  CUDA_TAG=cu128 ./dependence/setup.bash
  ```
  setup.bash will reinstall torch, re-pin numpy<2, and uninstall+rebuild
  detectron2/pytorch3d so their CUDA extensions match the new torch ABI.

- **detectron2 / pytorch3d build fails with `nvcc fatal: unsupported gpu architecture`**
  Set the architecture list before re-running:
  ```bash
  export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
  ./dependence/setup.bash
  ```

- **`Killed` during pytorch3d build** — RAM exhaustion. Use `MAX_JOBS=1` and add swap.

- **`ImportError: numpy.core.multiarray failed to import` after install**
  Step 1 of setup.bash pins `numpy<2`. If something later upgrades numpy, run
  `pip install 'numpy<2'` inside the venv (after `source activate.bash`).

- **`cubercnn` import fails after step 7** — Omni3D occasionally adds new
  module deps; run `cd dependence/src/omni3d && pip install <missing>` then
  re-run setup.bash.

- **Restart pytorch3d build manually without redoing the whole script:**
  ```bash
  source dependence/venv/bin/activate
  MAX_JOBS=2 pip install --no-build-isolation -e dependence/src/pytorch3d \
      2>&1 | tee dependence/logs/pytorch3d.log
  ```
