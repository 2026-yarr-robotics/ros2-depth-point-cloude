#!/usr/bin/env bash
# =============================================================================
# depth_digital_twin / Cube R-CNN dependency installer
# =============================================================================
# Installs PyTorch + detectron2 + pytorch3d + Omni3D (Cube R-CNN) into a self-
# contained Python venv under this folder, plus pretrained weights.
#
# Usage (run once):
#   ./dependence/setup.bash
#
# After install, every new terminal does:
#   source dependence/activate.bash
#
# Re-running setup.bash is safe — completed steps are skipped via probe imports.
#
# Optional environment overrides:
#   PYTHON_BIN=python3.10        # interpreter to base the venv on
#   CUDA_TAG=cu121|cu118|cpu     # PyTorch wheel index (auto-detected if unset)
#   MAX_JOBS=2                   # parallel C++ build jobs (low RAM machines)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SRC_DIR="$SCRIPT_DIR/src"
VENV_DIR="$SCRIPT_DIR/venv"
MODELS_DIR="$SCRIPT_DIR/models"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$SRC_DIR" "$MODELS_DIR" "$LOG_DIR"

# ---- pretty logging --------------------------------------------------------
log()  { printf '\033[1;34m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[setup]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[setup]\033[0m %s\n' "$*" >&2; exit 1; }
hr()   { printf '\033[1;34m[setup]\033[0m -----------------------------------\n'; }

# ---- detect Python interpreter --------------------------------------------
if [[ -z "${PYTHON_BIN:-}" ]]; then
    if command -v python3.10 >/dev/null 2>&1; then PYTHON_BIN=python3.10
    else PYTHON_BIN=python3; fi
fi
log "Python interpreter : $($PYTHON_BIN -V 2>&1)  ($PYTHON_BIN)"
"$PYTHON_BIN" -c "import sys; assert sys.version_info[:2]==(3,10), sys.version" \
    || warn "Python 3.10 (Humble default) is recommended. Continuing anyway."

# ---- detect GPU compute capability ----------------------------------------
# Returns the highest sm_XX (compute cap × 10) reported by nvidia-smi, or empty.
detect_compute_cap() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then echo ""; return; fi
    nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
        | head -1 | tr -d '.'
}

# ---- detect CUDA wheel tag for PyTorch ------------------------------------
# RTX 5080/5090 (Blackwell, sm_120) needs PyTorch ≥ 2.7 + cu128 wheels.
# Older GPUs (Ampere/Ada/Hopper) work with cu121 / cu118 / cu117.
detect_cuda_tag() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then echo "cpu"; return; fi
    local compute_cap drv_cuda major minor
    compute_cap=$(detect_compute_cap)
    if [[ -n "$compute_cap" ]] && (( compute_cap >= 100 )); then
        echo "cu128"; return
    fi
    drv_cuda=$(nvidia-smi 2>/dev/null \
        | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | head -1 | awk '{print $3}') || true
    if [[ -z "$drv_cuda" ]]; then echo "cpu"; return; fi
    IFS='.' read -r major minor <<< "$drv_cuda"
    if   (( major >= 12 ));                       then echo "cu121"
    elif (( major == 11 && minor >= 8 ));         then echo "cu118"
    elif (( major == 11 ));                       then echo "cu117"
    else                                                echo "cpu"; fi
}
CUDA_TAG="${CUDA_TAG:-$(detect_cuda_tag)}"
COMPUTE_CAP="$(detect_compute_cap)"
log "GPU compute cap   : ${COMPUTE_CAP:-<n/a>}"
log "PyTorch wheel index: $CUDA_TAG"
[[ "$CUDA_TAG" == "cpu" ]] && warn "No GPU detected — Cube R-CNN inference will be slow (seconds per frame)."

# ---- pick PyTorch / Torchvision version per CUDA tag ----------------------
# cu128 (Blackwell sm_120 e.g. RTX 5080/5090) requires torch ≥ 2.7.
case "$CUDA_TAG" in
    cu128) DEFAULT_TORCH=2.7.0;  DEFAULT_TV=0.22.0 ;;
    *)     DEFAULT_TORCH=2.1.2;  DEFAULT_TV=0.16.2 ;;
esac
TORCH_VERSION="${TORCH_VERSION:-$DEFAULT_TORCH}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-$DEFAULT_TV}"
log "PyTorch target    : torch==$TORCH_VERSION  torchvision==$TORCHVISION_VERSION"

# ---- create / activate venv (with system-site-packages so rclpy is reachable)
if [[ ! -d "$VENV_DIR" ]]; then
    log "Creating venv (system-site-packages enabled) at $VENV_DIR"
    "$PYTHON_BIN" -m venv --system-site-packages "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
log "Activated venv: $(python -c 'import sys; print(sys.prefix)')"
python -m pip install --upgrade pip wheel setuptools >/dev/null

# ---- skip-if-already-installed helper -------------------------------------
need() {
    local probe="$1"; shift
    local label="$1"; shift
    if python -c "$probe" >/dev/null 2>&1; then
        log "[skip] $label  (already present)"
        return 1
    fi
    log "[install] $label"
    return 0
}

# ---- 1) NumPy 1.x  (cv_bridge ABI) ----------------------------------------
hr
if need "import numpy, sys; sys.exit(0 if numpy.__version__.startswith('1.') else 1)" "numpy<2 (cv_bridge ABI)"; then
    pip install 'numpy<2'
fi

# ---- 2) PyTorch + Torchvision (version chosen above per CUDA tag) ---------
# Probe checks BOTH version match AND compute-capability support — if the
# venv has an older torch that does not list this GPU's sm_XX, we reinstall.
hr
TORCH_PROBE=$(cat <<PY
import torch, sys
want = "$TORCH_VERSION"
ok_ver = torch.__version__.split("+")[0] == want
sm = "$COMPUTE_CAP"
if sm and torch.cuda.is_available():
    archs = torch.cuda.get_arch_list()                  # ['sm_70', 'sm_75', ...]
    ok_arch = any(a == f"sm_{sm}" for a in archs)
else:
    ok_arch = True
sys.exit(0 if (ok_ver and ok_arch) else 1)
PY
)
TORCH_REINSTALLED=0
if need "$TORCH_PROBE" "PyTorch $TORCH_VERSION / Torchvision $TORCHVISION_VERSION ($CUDA_TAG)"; then
    # Force reinstall to clobber any incompatible existing torch.
    INDEX_URL="https://download.pytorch.org/whl/$CUDA_TAG"
    [[ "$CUDA_TAG" == "cpu" ]] && INDEX_URL="https://download.pytorch.org/whl/cpu"
    pip install --upgrade --force-reinstall --index-url "$INDEX_URL" \
        "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION"
    TORCH_REINSTALLED=1
fi

# Re-pin numpy<2 — torch reinstall can pull numpy 2.x which breaks ROS Humble's
# cv_bridge (compiled against the NumPy 1.x C-API).
if ! python -c "import numpy, sys; sys.exit(0 if numpy.__version__.startswith('1.') else 1)" >/dev/null 2>&1; then
    log "[fix] re-pinning numpy<2 (torch reinstall pulled numpy 2.x)"
    pip install 'numpy<2'
fi

# Record the active torch version so downstream source builds (detectron2,
# pytorch3d) can detect ABI mismatches and rebuild themselves.
ACTIVE_TORCH=$(python -c 'import torch; print(torch.__version__)')
echo "$ACTIVE_TORCH" > "$SCRIPT_DIR/.torch_version"
log "Recorded active torch version: $ACTIVE_TORCH"
python - <<'PY'
import torch
print(f"  torch={torch.__version__}  cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device     = '{torch.cuda.get_device_name(0)}'")
    cap = torch.cuda.get_device_capability(0)
    print(f"  compute    = sm_{cap[0]}{cap[1]}")
    print(f"  supported  = {torch.cuda.get_arch_list()}")
PY

# ---- 3) iopath + fvcore (Meta common deps) --------------------------------
hr
if need "import iopath" "iopath";   then pip install 'git+https://github.com/facebookresearch/iopath.git'; fi
if need "import fvcore" "fvcore";   then pip install 'git+https://github.com/facebookresearch/fvcore.git'; fi

# ---- helper: rebuild a source-built torch extension if torch ABI changed --
torch_abi_check() {
    # $1 = pkg name (detectron2 / pytorch3d), $2 = marker file
    local pkg="$1" marker="$2"
    local built; built=$(cat "$marker" 2>/dev/null || echo "")
    if [[ -z "$built" ]] || [[ "$built" != "$ACTIVE_TORCH" ]]; then
        if python -c "import $pkg" >/dev/null 2>&1; then
            log "[rebuild] $pkg was built against torch '${built:-unknown}' but active torch is '$ACTIVE_TORCH' → uninstalling for rebuild"
            pip uninstall -y "$pkg" >/dev/null 2>&1 || true
        fi
    fi
}

# ---- helper: detect that pytorch3d was built CPU-only (must rebuild) ------
pytorch3d_cuda_check() {
    if ! python -c "import pytorch3d" >/dev/null 2>&1; then return 0; fi
    if ! python -c "import torch; assert torch.cuda.is_available()" >/dev/null 2>&1; then
        return 0  # no GPU → CPU-only is fine
    fi
    if ! python - <<'PY' >/dev/null 2>&1
import torch
from pytorch3d.renderer.mesh import rasterize_meshes  # noqa: F401
from pytorch3d.structures import Meshes
v = torch.zeros(1, 3, 3, device="cuda")
f = torch.tensor([[[0, 1, 2]]], device="cuda")
m = Meshes(verts=v, faces=f)
from pytorch3d.renderer.mesh.rasterize_meshes import rasterize_meshes as _r
_r(m, image_size=8, blur_radius=0.0, faces_per_pixel=1)
PY
    then
        log "[rebuild] pytorch3d was built CPU-only — uninstalling for CUDA rebuild"
        pip uninstall -y pytorch3d >/dev/null 2>&1 || true
    fi
}

# ---- detect nvcc + sm_XX -> arch list for source builds -------------------
configure_cuda_build_env() {
    if [[ -n "${CUDA_HOME:-}" ]] && [[ -x "$CUDA_HOME/bin/nvcc" ]]; then
        :
    elif [[ -d /usr/local/cuda-12.8 ]] && [[ -x /usr/local/cuda-12.8/bin/nvcc ]]; then
        export CUDA_HOME=/usr/local/cuda-12.8
    elif [[ -d /usr/local/cuda ]] && [[ -x /usr/local/cuda/bin/nvcc ]]; then
        export CUDA_HOME=/usr/local/cuda
    elif command -v nvcc >/dev/null 2>&1; then
        export CUDA_HOME=$(dirname "$(dirname "$(which nvcc)")")
    fi
    if [[ -n "${CUDA_HOME:-}" ]]; then
        export PATH="$CUDA_HOME/bin:$PATH"
        local nvcc_ver
        nvcc_ver=$("$CUDA_HOME/bin/nvcc" --version 2>/dev/null \
            | grep -oE 'release [0-9]+\.[0-9]+' | awk '{print $2}')
        log "CUDA_HOME=$CUDA_HOME (nvcc $nvcc_ver)"
        export FORCE_CUDA=1
        # Map sm_XX (e.g. sm_120 for Blackwell) to the dotted form torch expects.
        if [[ -n "$COMPUTE_CAP" ]]; then
            local maj=${COMPUTE_CAP:0:-1}
            local min=${COMPUTE_CAP: -1}
            export TORCH_CUDA_ARCH_LIST="${maj}.${min}"
            log "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST  FORCE_CUDA=1"
        fi
    else
        warn "nvcc not found — pytorch3d/detectron2 will build CPU-only."
        warn "  Install CUDA 12.8 for RTX 50xx (Blackwell sm_120):"
        warn "    https://developer.nvidia.com/cuda-12-8-0-download-archive"
    fi
}
configure_cuda_build_env

# ---- 4) detectron2 (source build) -----------------------------------------
# `--no-build-isolation` is required because detectron2/pytorch3d setup.py
# import torch at module load time; PEP 517 isolation creates a fresh build
# env without torch and breaks. We already installed torch in step 2 so the
# venv satisfies all build-time imports.
hr
torch_abi_check detectron2 "$SCRIPT_DIR/.detectron2_torch"
if need "import detectron2" "detectron2  (source build, 5–15 min)"; then
    if [[ ! -d "$SRC_DIR/detectron2/.git" ]]; then
        git clone https://github.com/facebookresearch/detectron2 "$SRC_DIR/detectron2"
    fi
    if [[ -d /usr/local/cuda ]]; then export CUDA_HOME=/usr/local/cuda; fi
    MAX_JOBS="${MAX_JOBS:-4}" pip install --no-build-isolation -e "$SRC_DIR/detectron2" 2>&1 \
        | tee "$LOG_DIR/detectron2.log"
    echo "$ACTIVE_TORCH" > "$SCRIPT_DIR/.detectron2_torch"
fi

# ---- 5) pytorch3d (source build, heavy) -----------------------------------
hr
torch_abi_check pytorch3d "$SCRIPT_DIR/.pytorch3d_torch"
pytorch3d_cuda_check
if need "import pytorch3d" "pytorch3d  (source build, 10–20 min)"; then
    if [[ ! -d "$SRC_DIR/pytorch3d/.git" ]]; then
        git clone https://github.com/facebookresearch/pytorch3d "$SRC_DIR/pytorch3d"
    fi
    MAX_JOBS="${MAX_JOBS:-2}" pip install --no-build-isolation -e "$SRC_DIR/pytorch3d" 2>&1 \
        | tee "$LOG_DIR/pytorch3d.log"
    echo "$ACTIVE_TORCH" > "$SCRIPT_DIR/.pytorch3d_torch"
fi

# ---- 6) Other Python deps -------------------------------------------------
hr
if need "import cv2"          "opencv-python";  then pip install opencv-python; fi
if need "import scipy"        "scipy";          then pip install scipy; fi
if need "import seaborn"      "seaborn";        then pip install seaborn; fi
if need "import Cython"       "cython";         then pip install cython; fi
if need "import pycocotools"  "cocoapi";        then
    pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
fi
if need "import yaml"         "PyYAML";         then pip install pyyaml; fi
if need "import ultralytics"  "ultralytics (YOLO seg)"; then pip install ultralytics; fi
# colcon inside the venv → `colcon build` resolves to the venv's bin and
# scripts get a venv shebang. Without this, system colcon shebangs all node
# entry points with /usr/bin/python3, which can't see the venv's modules.
if need "import colcon" "colcon-common-extensions"; then
    pip install colcon-common-extensions
fi

# ---- 7) Omni3D / Cube R-CNN -----------------------------------------------
# Omni3D ships *no* setup.py / pyproject.toml — it's run as a script from its
# clone directory. We make `cubercnn` (and `demo/` etc.) importable from
# anywhere by dropping a single .pth file into the venv's site-packages, which
# Python adds to sys.path automatically at startup.
hr
if need "from cubercnn.config import get_cfg_defaults" "Omni3D / cubercnn"; then
    if [[ ! -d "$SRC_DIR/omni3d/.git" ]]; then
        git clone https://github.com/facebookresearch/omni3d "$SRC_DIR/omni3d"
    fi
    SITE_PACKAGES=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
    PTH_FILE="$SITE_PACKAGES/omni3d.pth"
    echo "$SRC_DIR/omni3d" > "$PTH_FILE"
    log "Wrote $PTH_FILE -> $SRC_DIR/omni3d"
fi

# ---- 8) Pretrained Cube R-CNN weights -------------------------------------
hr
WEIGHT_FILE="$MODELS_DIR/cubercnn_DLA34_FPN.pth"
WEIGHT_URL="https://dl.fbaipublicfiles.com/cubercnn/omni3d/cubercnn_DLA34_FPN.pth"
if [[ -s "$WEIGHT_FILE" ]]; then
    log "[skip] weights already at $WEIGHT_FILE"
else
    log "Downloading $(basename "$WEIGHT_FILE")  (~250 MB)"
    if command -v wget >/dev/null 2>&1; then
        wget -q --show-progress -O "$WEIGHT_FILE" "$WEIGHT_URL"
    else
        curl -L --progress-bar -o "$WEIGHT_FILE" "$WEIGHT_URL"
    fi
fi

CONFIG_FILE="$SRC_DIR/omni3d/configs/Base_Omni3D.yaml"
[[ -f "$CONFIG_FILE" ]] || warn "Config file not found at $CONFIG_FILE; expected after omni3d clone."

# ---- 9) Final smoke test --------------------------------------------------
hr
log "Smoke test: import all critical modules + verify CUDA compute support"
python - <<'PY' || die "Import smoke test failed — check the [install] logs above."
import importlib, sys
mods = ["numpy", "torch", "torchvision", "iopath", "fvcore",
        "detectron2", "pytorch3d", "cv2", "scipy", "seaborn",
        "pycocotools", "yaml", "ultralytics", "cubercnn",
        "cubercnn.config"]
ok = True
for m in mods:
    try:
        importlib.import_module(m)
        print(f"  ✓ {m}")
    except Exception as e:
        ok = False
        print(f"  ✗ {m}  ->  {type(e).__name__}: {e}")

# numpy 1.x guard (cv_bridge ABI)
import numpy
if not numpy.__version__.startswith("1."):
    ok = False
    print(f"  ✗ numpy {numpy.__version__} — must be <2 for ROS Humble cv_bridge")
else:
    print(f"  ✓ numpy {numpy.__version__} (cv_bridge compatible)")

# CUDA compute-capability sanity check
import torch
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    sm = f"sm_{cap[0]}{cap[1]}"
    archs = torch.cuda.get_arch_list()
    if archs and sm not in archs:
        ok = False
        print(f"  ✗ GPU {sm} not in torch supported list {archs}")
    else:
        print(f"  ✓ CUDA  device='{torch.cuda.get_device_name(0)}'  {sm}  in {archs}")

# Quick CUDA op smoke test (catches torch/detectron2 ABI mismatch immediately)
try:
    import detectron2.layers as _d2l   # noqa: F401
    import pytorch3d._C as _p3dC       # noqa: F401
    if torch.cuda.is_available():
        _ = torch.zeros(2, 3, device="cuda") + 1
    print("  ✓ detectron2.layers + pytorch3d._C loadable")
except Exception as e:
    ok = False
    print(f"  ✗ CUDA extension import failed: {type(e).__name__}: {e}")

sys.exit(0 if ok else 1)
PY

# ---- summary --------------------------------------------------------------
hr
log "Done."
log ""
log "  venv     : $VENV_DIR"
log "  src      : $SRC_DIR"
log "  weights  : $WEIGHT_FILE"
log "  config   : $CONFIG_FILE"
log ""
log "Activate the env in every new terminal:"
log "    source $SCRIPT_DIR/activate.bash"
log ""
log "Then launch the Cube R-CNN pipeline:"
log "    ros2 launch depth_digital_twin cube_rcnn.launch.py \\"
log "      config_file:=\$CUBERCNN_CONFIG \\"
log "      weights:=\$CUBERCNN_WEIGHTS device:=cuda"
hr
