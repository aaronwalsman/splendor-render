#!/bin/bash
# Generate all tutorial images from the example scripts/scenes.
# Run from the repo root: bash docs/generate_images.sh
set -e

IMGDIR="docs/images"
mkdir -p "$IMGDIR"

echo "Generating tutorial images..."

# 01 — hello cube
python examples/01_hello_cube.py "$IMGDIR/01_hello_cube.png"

# 02 — primitives
splendor_render examples/02_primitives \
    --output "$IMGDIR/02_primitives.png" --resolution 768x512

# 03 — shadows
splendor_render examples/03_shadows \
    --output "$IMGDIR/03_shadows.png" --resolution 512x512

# 04 — debug overlays
splendor_render examples/04_debug_overlays \
    --output "$IMGDIR/04_debug_overlays.png" --resolution 768x512

# 05 — materials
splendor_render examples/05_materials \
    --output "$IMGDIR/05_materials.png" --resolution 768x512

# 06 — radial distortion
splendor_render examples/06_radial_distortion \
    --output "$IMGDIR/06_radial_distortion.png" --resolution 512x512

# 07 — render modes (generates multiple images)
python examples/07_render_modes.py "$IMGDIR/07_render_modes"

# 08 — multi light
splendor_render examples/08_multi_light \
    --output "$IMGDIR/08_multi_light.png" --resolution 768x512

# 10 — coord render (generates multiple images)
python examples/10_coord_render.py "$IMGDIR/10_coord_render"

echo "Done. Images saved to $IMGDIR/"
ls -la "$IMGDIR/"
