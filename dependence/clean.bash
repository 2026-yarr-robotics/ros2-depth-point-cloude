#!/usr/bin/env bash
# =============================================================================
# Wipe everything installed by setup.bash so the next run starts from scratch.
#
# Usage:
#   ./dependence/clean.bash            # interactive confirmation
#   ./dependence/clean.bash --force    # skip confirmation
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [[ "${1:-}" != "--force" ]]; then
    echo "About to delete:"
    for d in venv src models logs; do
        path="$SCRIPT_DIR/$d"
        [[ -e "$path" ]] && echo "  $path" || echo "  $path  (missing)"
    done
    read -r -p "Proceed? [y/N] " ans
    [[ "$ans" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 1; }
fi

rm -rf "$SCRIPT_DIR/venv" "$SCRIPT_DIR/src" "$SCRIPT_DIR/models" "$SCRIPT_DIR/logs"
echo "[clean] done."
