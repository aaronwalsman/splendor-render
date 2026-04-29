#!/usr/bin/env python
"""Render coordinate maps: each pixel encodes the object-space XYZ position
of the surface, normalized to [0,1] within the instance's coord_box.

This is useful for dense correspondence, pose estimation, and spatial
reasoning — each pixel's RGB value tells you where on the object's local
surface that ray hit.
"""
import math
import sys

import numpy

from splendor.contexts.egl import EGLContext
from splendor.core import SplendorRender
from splendor.camera import projection_matrix
from splendor.image import save_image

WIDTH, HEIGHT = 512, 512
OUTPUT_PREFIX = sys.argv[1] if len(sys.argv) > 1 else 'coord_render'

ctx = EGLContext()
try:
    renderer = SplendorRender()

    # camera
    proj = projection_matrix(math.radians(60.), WIDTH / HEIGHT)
    renderer.load_camera(
        'main',
        view_matrix=[0.3, -0.25, 0, 6.0, 0, 0],
        projection=proj,
    )
    renderer.load_sensor('rgb', WIDTH, HEIGHT)

    # geometry
    renderer.load_mesh(
        'cube',
        mesh_primitive={'shape': 'cube', 'bezel': 0.03},
        color_mode='flat_color',
    )
    renderer.load_mesh(
        'sphere',
        mesh_primitive={'shape': 'sphere', 'radius': 1.0},
        color_mode='flat_color',
    )
    renderer.load_material(
        'mat',
        flat_color=(0.7, 0.3, 0.2),
        metal=0, rough=0.5, base_reflect=0.04, ambient=0.15,
    )

    # add instances with coord_box matching their mesh extents
    renderer.add_instance(
        'box',
        mesh_name='cube',
        material_name='mat',
        transform=[[1,0,0,-1.5],[0,1,0,1],[0,0,1,0],[0,0,0,1]],
        mask_color=[1, 0, 0],
        coord_box=[[-1, -1, -1], [1, 1, 1]],
    )
    renderer.add_instance(
        'ball',
        mesh_name='sphere',
        material_name='mat',
        transform=[[1,0,0,1.5],[0,1,0,1],[0,0,1,0],[0,0,0,1]],
        mask_color=[0, 1, 0],
        coord_box=[[-1, -1, -1], [1, 1, 1]],
    )

    # lighting (needed for color render, not for coord render)
    renderer.set_ambient_color((0.1, 0.1, 0.12))

    # color render for reference
    renderer.add_direction_light('sun', color=[1.2, 1.1, 1.0])
    renderer.color_render('main', sensor='rgb')
    color_image = renderer.read_sensor('rgb')
    save_image(color_image, f'{OUTPUT_PREFIX}_color.png')
    print(f'Saved {OUTPUT_PREFIX}_color.png')

    # coord render — RGB = normalized (x, y, z) within each object's coord_box
    renderer.coord_render('main', sensor='rgb')
    coord_image = renderer.read_sensor('rgb')
    save_image(coord_image, f'{OUTPUT_PREFIX}_coords.png')
    print(f'Saved {OUTPUT_PREFIX}_coords.png')
    print('In the coord image, RGB = (x, y, z) in [0,1] within each '
          "object's bounding box.")

finally:
    ctx.close()
