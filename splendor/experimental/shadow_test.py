"""
Shadow map test scene.

Loads shadow_test.json and renders a color image.
Run with:
    python shadow_test.py   -> renders to ./shadow_test_color.png
"""

import math

from splendor.contexts.egl import EGLContext
import splendor.core as core
import splendor.camera as camera
from splendor.image import save_image

WIDTH, HEIGHT = 512, 512


def test_render():
    """Load a shadow test scene, render it, and verify the output is non-trivial."""
    ctx = EGLContext()
    try:
        renderer = core.SplendorRender()
        renderer.load_scene('shadow_test')

        # Scene camera has a view_matrix but no projection; set one.
        proj = camera.projection_matrix(math.radians(60.), WIDTH / HEIGHT)
        renderer.set_camera_projection('main', proj)

        renderer.load_sensor('output', WIDTH, HEIGHT, anti_alias=True)

        renderer.color_render('main', sensor='output', flip_y=True)
        image = renderer.read_sensor('output')
        save_image(image, './shadow_test_color.png')
        print('Saved color image to ./shadow_test_color.png')

        assert image.max() > 0, 'Image is all black'
        print('All assertions passed.')
    finally:
        ctx.close()


if __name__ == '__main__':
    test_render()
