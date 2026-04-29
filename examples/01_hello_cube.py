#!/usr/bin/env python
"""Minimal splendor example: render a colored cube with a direction light."""
import math
import sys

from splendor.contexts.egl import EGLContext
from splendor.core import SplendorRender
from splendor.camera import projection_matrix, direction_light_pose
from splendor.image import save_image

WIDTH, HEIGHT = 512, 512
OUTPUT = sys.argv[1] if len(sys.argv) > 1 else 'hello_cube.png'

ctx = EGLContext()
try:
    renderer = SplendorRender()

    # camera
    projection = projection_matrix(math.radians(60.), WIDTH / HEIGHT)
    renderer.load_camera(
        'main',
        view_matrix=[0.2, -0.3, 0, 4.0, 0, 0],
        projection=projection,
    )
    renderer.load_sensor('rgb', WIDTH, HEIGHT)

    # geometry
    renderer.load_mesh(
        'cube',
        mesh_primitive={'shape': 'cube'},
        color_mode='flat_color',
    )
    renderer.load_material(
        'blue_mat',
        flat_color=(0.2, 0.4, 0.9),
        metal=0.0,
        rough=0.4,
        base_reflect=0.04,
        ambient=0.15,
    )
    renderer.add_instance('cube', mesh_name='cube', material_name='blue_mat')

    # lighting
    renderer.add_direction_light(
        'sun',
        pose=direction_light_pose((1.0, -1.0, -0.5)),
        color=(1.2, 1.1, 1.0),
    )
    renderer.set_ambient_color((0.08, 0.08, 0.1))

    # render and save
    renderer.color_render('main', sensor='rgb')
    image = renderer.read_sensor('rgb')
    save_image(image, OUTPUT)
    print(f'Saved {OUTPUT} ({WIDTH}x{HEIGHT})')

finally:
    ctx.close()
