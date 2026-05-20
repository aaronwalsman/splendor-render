#!/bin/bash
# Demonstrate splendor-render command-line tools.
# Run from the repo root: bash examples/09_cli_usage.sh

set -e
OUTDIR="cli_output"
mkdir -p "$OUTDIR"

echo "=== Rendering scenes with splendor_render ==="

# Color render
splendor_render examples/02_primitives \
    "$OUTDIR/primitives_color.png" \
    --resolution 768x512
echo "  -> $OUTDIR/primitives_color.png"

# Mask render
splendor_render examples/02_primitives \
    "$OUTDIR/primitives_mask.png" \
    --resolution 768x512 \
    --render-mode mask
echo "  -> $OUTDIR/primitives_mask.png"

# Depth render
splendor_render examples/03_shadows \
    "$OUTDIR/shadows_depth.npy" \
    --resolution 512x512 \
    --render-mode depth
echo "  -> $OUTDIR/shadows_depth.npy"

# Radial distortion
splendor_render examples/06_radial_distortion \
    "$OUTDIR/distortion.png" \
    --resolution 512x512
echo "  -> $OUTDIR/distortion.png"

echo ""
echo "=== Interactive viewer ==="
echo "To open any scene in the interactive viewer:"
echo "  splendor_viewer examples/03_shadows"
echo ""
echo "Viewer hotkeys:"
echo "  Left-drag   — orbit"
echo "  Right-drag  — pan"
echo "  Scroll      — zoom"
echo "  S           — save screenshot"
echo "  M           — toggle color/mask mode"

echo ""
echo "Done. Output files in $OUTDIR/"
