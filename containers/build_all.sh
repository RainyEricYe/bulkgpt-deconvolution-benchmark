#!/bin/bash
# =============================================================================
# Build all SIFs from definition files
#
# Usage:
#   bash build_all.sh                    # Build all SIFs
#   bash build_all.sh --method tape      # Build only TAPE
#   bash build_all.sh --method squid     # Build only SQUID
#
# Requirements:
#   - Apptainer >= 1.1 (or Singularity >= 3.8)
#   - Internet connection (for Docker pulls and package downloads)
#   - ~10 GB free disk space (for all SIFs)
#   - Root or user-namespace unprivileged build support
#
# Output: ./sif/{method}.sif for each method
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SIF_DIR="$SCRIPT_DIR/sif"
mkdir -p "$SIF_DIR"

# List of methods to build
METHODS=("tape" "squid" "demixsc" "condecon" "sweetwater" "hspe" "mixupvi")

# Optional: only build specified method
if [ "$1" = "--method" ] && [ -n "$2" ]; then
    METHODS=("$2")
fi

echo "=== Building container SIFs ==="
echo "SIF output directory: $SIF_DIR"

build_sif() {
    local method="$1"
    local def_file="$SCRIPT_DIR/$method/$method.def"

    if [ ! -f "$def_file" ]; then
        echo "  [SKIP] $method: definition file not found at $def_file"
        return
    fi

    echo ""
    echo "--- Building $method.sif ---"
    cd "$SCRIPT_DIR/$method"
    apptainer build "$SIF_DIR/$method.sif" "$method.def" 2>&1 | tail -10
    echo "--- $method.sif done (exit code: $?) ---"
}

for method in "${METHODS[@]}"; do
    build_sif "$method"
done

echo ""
echo "=== All builds complete ==="
echo "Built SIFs:"
ls -lh "$SIF_DIR"/*.sif 2>/dev/null || echo "(none)"
