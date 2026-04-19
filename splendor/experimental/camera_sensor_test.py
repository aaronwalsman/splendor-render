"""
Basic test of the camera/sensor API.

Renders a colored cube with radial distortion offscreen using EGL, reads back
the color image and a depth map, and saves both to /tmp/.
"""

import math

import numpy as np

import splendor.camera as camera
from splendor.camera import direction_light_pose
from splendor.contexts.egl import EGLContext
from splendor.core import SplendorRender
from splendor.image import save_image, save_depth

WIDTH, HEIGHT = 512, 512

# Noticeable barrel distortion
RADIAL_K1 = 0.3
RADIAL_K2 = 0.05

def test_render():
    ctx = EGLContext()
    try:
        _test_render()
    finally:
        ctx.close()

def _test_render():
    renderer = SplendorRender()

    # --- camera and sensor ---
    projection = camera.projection_matrix(math.radians(60.), WIDTH / HEIGHT)
    view = camera.view_matrix([0.3, 0.2, 0, 3, 0, 0])  # slight angle
    renderer.load_camera(
        'main',
        view_matrix=view,
        projection=projection,
        radial_k1=RADIAL_K1,
        radial_k2=RADIAL_K2,
    )
    renderer.load_sensor('rgb', WIDTH, HEIGHT, anti_alias=True,
                         enable_radial_distortion=True)

    # --- scene geometry ---
    renderer.load_mesh(
        'cube_mesh',
        mesh_primitive={'shape': 'cube'},
        color_mode='flat_color',
    )
    renderer.load_material(
        'cube_mat',
        flat_color=(0.2, 0.5, 1.0),
        metal=0.0,
        rough=0.5,
        base_reflect=0.04,
        ambient=0.3,
    )
    renderer.add_instance(
        'cube',
        mesh_name='cube_mesh',
        material_name='cube_mat',
    )

    # --- lighting ---
    renderer.add_direction_light(
        'sun',
        pose=direction_light_pose((1.0, -1.0, -1.0)),
        color=(1.0, 1.0, 1.0),
    )
    renderer.set_ambient_color((0.1, 0.1, 0.1))

    # --- render color ---
    renderer.color_render('main', sensor='rgb', flip_y=True)
    color_image = renderer.read_sensor('rgb')
    save_image(color_image, '/tmp/camera_sensor_test_color.png')
    print('Saved color image to /tmp/camera_sensor_test_color.png')

    # --- read depth from the same render pass ---
    depth_image = renderer.read_sensor(
        'rgb',
        read_depth=True,
        projection=projection,
    )
    save_depth(depth_image, '/tmp/camera_sensor_test_depth.npy')
    print('Saved depth map to /tmp/camera_sensor_test_depth.npy')

    # --- verify we got something non-trivial ---
    assert color_image.shape == (HEIGHT, WIDTH, 3), color_image.shape
    assert color_image.max() > 0, 'Color image is all black'
    assert depth_image.min() < depth_image.max(), 'Depth image has no variation'
    print('All assertions passed.')

if __name__ == '__main__':
    test_render()

