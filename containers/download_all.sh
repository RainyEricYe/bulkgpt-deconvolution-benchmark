#!/bin/bash
# =============================================================================
# Download pre-built SIFs (placeholder)
#
# Pre-built SIF binaries are NOT distributed with this release. The supported
# way to obtain a SIF for a containerized method is to build it from its
# Apptainer definition file:
#
#     bash build_all.sh                    # build all methods
#     bash build_all.sh --method tape      # build one method
#
# This script is kept as a placeholder so that pre-built SIFs can be published
# later without changing the interface.
# =============================================================================

echo "============================================================"
echo "Pre-built SIF files are not distributed with this release."
echo ""
echo "Build SIFs from their Apptainer definitions instead:"
echo "  bash build_all.sh                    # all methods"
echo "  bash build_all.sh --method tape      # single method"
echo ""
echo "Requires Apptainer >= 1.1 (see containers/README.md)."
echo "============================================================"
exit 1
