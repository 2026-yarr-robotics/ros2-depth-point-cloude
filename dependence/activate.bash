#!/usr/bin/env bash
# =============================================================================
# Activate the depth_digital_twin Python venv + ROS overlay in one go.
#
# Usage (must be sourced — do NOT execute):
#   source dependence/activate.bash
# =============================================================================

# Detect script dir whether sourced from bash or zsh.
if [[ -n "${BASH_SOURCE:-}" ]]; then _DEP_SRC="${BASH_SOURCE[0]}"
else _DEP_SRC="${(%):-%x}"; fi
DEP_DIR="$( cd "$( dirname "$_DEP_SRC" )" && pwd )"
WS_DIR="$( cd "$DEP_DIR/.." && pwd )"
unset _DEP_SRC

if [[ ! -d "$DEP_DIR/venv" ]]; then
    echo "[activate] venv not found at $DEP_DIR/venv" >&2
    echo "[activate] run ./dependence/setup.bash first" >&2
    return 1 2>/dev/null || exit 1
fi

# 1) ROS underlay (if not yet sourced)
if [[ -z "${ROS_DISTRO:-}" ]]; then
    if [[ -f /opt/ros/humble/setup.bash ]]; then
        source /opt/ros/humble/setup.bash
    else
        echo "[activate] /opt/ros/humble/setup.bash not found — install ROS 2 Humble or adjust this script" >&2
    fi
fi

# 2) Python venv
# shellcheck disable=SC1091
source "$DEP_DIR/venv/bin/activate"

# 3) Workspace overlay (post colcon build)
if [[ -f "$WS_DIR/install/setup.bash" ]]; then
    # shellcheck disable=SC1091
    source "$WS_DIR/install/setup.bash"
fi

# 4) Convenience env vars used by Cube R-CNN launch.
# Pick whichever Base_Omni3D*.yaml exists (Omni3D has renamed the file across
# revisions: Base_Omni3D.yaml / Base_Omni3D_demo.yaml / Base_Omni3D_in.yaml).
_CFG_DIR="$DEP_DIR/src/omni3d/configs"
_CFG_PICK=""
for c in Base_Omni3D.yaml Base_Omni3D_demo.yaml Base_Omni3D_in.yaml \
         cubercnn_DLA34_FPN.yaml ; do
    if [[ -f "$_CFG_DIR/$c" ]]; then _CFG_PICK="$_CFG_DIR/$c"; break; fi
done
export CUBERCNN_CONFIG="$_CFG_PICK"
export CUBERCNN_WEIGHTS="$DEP_DIR/models/cubercnn_DLA34_FPN.pth"
export DEPTH_DIGITAL_TWIN_DEPS="$DEP_DIR"

# Warn loudly if either path is wrong — saves a surprise at ros2 launch time.
_WARN=""
[[ -z "$CUBERCNN_CONFIG" ]] && _WARN+="\n  ⚠ no Base_Omni3D*.yaml found under $_CFG_DIR"
[[ -n "$CUBERCNN_CONFIG" && ! -f "$CUBERCNN_CONFIG" ]] && _WARN+="\n  ⚠ CUBERCNN_CONFIG does not exist: $CUBERCNN_CONFIG"
[[ ! -f "$CUBERCNN_WEIGHTS" ]] && _WARN+="\n  ⚠ CUBERCNN_WEIGHTS does not exist: $CUBERCNN_WEIGHTS"

# Verify colcon is the venv copy. If system colcon (/usr/bin/colcon) is what
# resolves, builds will get system-python shebangs that can't see the venv.
_COLCON_PATH=$(command -v colcon || true)
case "$_COLCON_PATH" in
    "$DEP_DIR/venv/bin/"*) _COLCON_NOTE="" ;;
    "")  _COLCON_NOTE=$'\n  ⚠ colcon not found at all. run ./dependence/setup.bash again.' ;;
    *)   _COLCON_NOTE=$"\n  ⚠ colcon resolves to $_COLCON_PATH (system). Build will use system python; node shebangs will NOT use the venv. Either run\n      pip install colcon-common-extensions\n    inside the venv, or always invoke\n      python -m colcon build --symlink-install" ;;
esac

cat <<EOF
[activate] depth_digital_twin env ready
  python      : $(python -c 'import sys; print(sys.executable)')
  colcon      : ${_COLCON_PATH:-<missing>}
  ROS_DISTRO  : ${ROS_DISTRO:-<not set>}
  workspace   : $WS_DIR
  cubercnn cfg: ${CUBERCNN_CONFIG:-<not found>}
  cubercnn wts: $CUBERCNN_WEIGHTS$( [[ -n "$_WARN$_COLCON_NOTE" ]] && printf "%b" "$_WARN$_COLCON_NOTE" )
EOF
unset _CFG_DIR _CFG_PICK _WARN _COLCON_PATH _COLCON_NOTE c
