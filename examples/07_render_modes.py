#!/usr/bin/env python
"""Render the same scene in color, mask, and depth modes side by side."""
import math
import sys

import numpy

from splendor.contexts.egl import EGLContext
from splendor.core import SplendorRender
from splendor.camera import projection_matrix, direction_light_pose
from splendor.image import save_image, save_depth

OUTPUT_PREFIX = sys.argv[1] if len(sys.argv) > 1 else 'render_modes'
WIDTH, HEIGHT = 512, 512

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
    renderer.load_sensor('rgb', WIDTH, HEIGHT, anti_alias=True)

    # scene: a few objects on a ground plane
    renderer.load_mesh(
        'ground',
        mesh_primitive={
            'shape': 'cube',
            'x_extents': [-6, 6],
            'y_extents': [-0.1, 0.1],
            'z_extents': [-6, 6],
        },
        color_mode='flat_color',
    )
    renderer.load_mesh(
        'cube',
        mesh_primitive={'shape': 'cube', 'bezel': 0.03},
        color_mode='flat_color',
    )
    renderer.load_mesh(
        'sphere',
        mesh_primitive={'shape': 'sphere', 'radius': 0.8},
        color_mode='flat_color',
    )
    renderer.load_mesh(
        'cylinder',
        mesh_primitive={
            'shape': 'cylinder',
            'start_height': 0, 'end_height': 1.5,
            'radius': 0.5, 'start_cap': True, 'end_cap': True,
        },
        color_mode='flat_color',
    )

    renderer.load_material('ground_mat',
        flat_color=(0.5, 0.48, 0.45),
        metal=0, rough=0.9, base_reflect=0.02, ambient=0.3)
    renderer.load_material('red',
        flat_color=(0.8, 0.15, 0.1),
        metal=0, rough=0.4, base_reflect=0.04, ambient=0.1)
    renderer.load_material('green',
        flat_color=(0.15, 0.7, 0.2),
        metal=0, rough=0.4, base_reflect=0.04, ambient=0.1)
    renderer.load_material('blue',
        flat_color=(0.2, 0.3, 0.85),
        metal=0, rough=0.4, base_reflect=0.04, ambient=0.1)

    renderer.add_instance('ground', mesh_name='ground',
        material_name='ground_mat',
        transform=numpy.eye(4).tolist(), mask_color=[0, 0, 0])
    renderer.add_instance('box', mesh_name='cube',
        material_name='red',
        transform=[[1,0,0,-2],[0,1,0,1],[0,0,1,0],[0,0,0,1]],
        mask_color=[1, 0, 0])
    renderer.add_instance('ball', mesh_name='sphere',
        material_name='green',
        transform=[[1,0,0,0],[0,1,0,0.8],[0,0,1,0],[0,0,0,1]],
        mask_color=[0, 1, 0])
    renderer.add_instance('cyl', mesh_name='cylinder',
        material_name='blue',
        transform=[[1,0,0,2],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
        mask_color=[0, 0, 1])

    renderer.add_direction_light('sun',
        pose=direction_light_pose((0.5, -1.0, -0.4)),
        color=(1.2, 1.1, 1.0))
    renderer.set_ambient_color((0.1, 0.1, 0.12))

    # color render
    renderer.color_render('main', sensor='rgb')
    color_image = renderer.read_sensor('rgb')
    save_image(color_image, f'{OUTPUT_PREFIX}_color.png')
    print(f'Saved {OUTPUT_PREFIX}_color.png')

    # mask render
    renderer.mask_render('main', sensor='rgb')
    mask_image = renderer.read_sensor('rgb')
    save_image(mask_image, f'{OUTPUT_PREFIX}_mask.png')
    print(f'Saved {OUTPUT_PREFIX}_mask.png')

    # depth render (from the color pass)
    renderer.color_render('main', sensor='rgb')
    depth_image = renderer.read_sensor('rgb', read_depth=True, projection=proj)
    save_depth(depth_image, f'{OUTPUT_PREFIX}_depth.npy')
    # also save a visualization
    d = depth_image[:, :, 0]
    valid = d < d.max()
    if valid.any():
        d_min, d_max = d[valid].min(), d[valid].max()
        d_vis = numpy.clip((d - d_min) / (d_max - d_min + 1e-6), 0, 1)
        d_vis = (d_vis * 255).astype(numpy.uint8)
        d_vis = numpy.stack([d_vis, d_vis, d_vis], axis=-1)
        save_image(d_vis, f'{OUTPUT_PREFIX}_depth_vis.png')
        print(f'Saved {OUTPUT_PREFIX}_depth_vis.png')

finally:
    ctx.close()
