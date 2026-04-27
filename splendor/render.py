"""Headless scene rendering — load a scene, render, and save the result."""
import math
import warnings

from splendor.contexts.egl import EGLContext
import splendor.core as core
import splendor.camera as camera
from splendor.image import save_image, save_depth

DEFAULT_RESOLUTION = '512x512'
DEFAULT_ANTI_ALIAS_SAMPLES = 8

def render_scene(
    scene,
    assets=None,
    output_file=None,
    camera='main',
    sensor='output',
    resolution=None,
    anti_alias_samples=None,
    render_mode='color',
    device=None,
):
    """Render a scene headlessly using EGL.

    scene can be a scene name (looked up in the asset library) or a dict.
    render_mode is 'color', 'mask', or 'depth'.  Returns the rendered image
    as a numpy array.  If output_file is specified, also saves to disk.
    """
    ctx = EGLContext(device=device)
    try:
        renderer = core.SplendorRender(assets=assets)
        renderer.load_scene(scene)

        # Ensure the named camera exists and has a resolution-appropriate projection.
        # If the scene defined the camera (with a view_matrix but no projection),
        # load_camera's default 90° 1:1 is already in place; we only create/replace
        # when the camera is absent entirely.
        if not renderer.camera_exists(camera):
            proj = _default_projection(resolution)
            renderer.load_camera(camera, projection=proj)

        # Resolve the sensor: use scene-defined one or create from CLI args
        if renderer.sensor_exists(sensor):
            if resolution is not None or anti_alias_samples is not None:
                warnings.warn(
                    f'Sensor {sensor!r} is already defined in the scene; '
                    f'--resolution and --anti-alias-samples are ignored.')
        else:
            w, h = _parse_resolution(resolution or DEFAULT_RESOLUTION)
            samples = (anti_alias_samples
                       if anti_alias_samples is not None
                       else DEFAULT_ANTI_ALIAS_SAMPLES)
            anti_alias = samples != 0
            k1 = renderer.get_camera_radial_k1(camera)
            k2 = renderer.get_camera_radial_k2(camera)
            renderer.load_sensor(
                sensor, w, h,
                enable_radial_distortion=(k1 != 0.0 or k2 != 0.0),
                anti_alias=anti_alias,
                anti_alias_samples=samples,
            )

        if render_mode in ('color', 'depth'):
            renderer.color_render(camera, sensor=sensor, flip_y=True)
        elif render_mode == 'mask':
            renderer.mask_render(camera, sensor=sensor, flip_y=True)
        else:
            raise ValueError(f'Unknown render_mode: {render_mode!r}')

        image = renderer.read_sensor(
            sensor,
            read_depth=(render_mode == 'depth'),
            projection=renderer.get_camera_projection(camera),
        )

    finally:
        ctx.close()

    if output_file is not None:
        if render_mode == 'depth':
            save_depth(image, output_file)
        else:
            save_image(image, output_file)

    return image


def _parse_resolution(resolution):
    """Parse a resolution string like '512x512' into (width, height) integers."""
    w, h = resolution.lower().split('x')
    return int(w), int(h)

def _default_projection(resolution):
    """Create a default 90-degree perspective projection for the given resolution."""
    w, h = _parse_resolution(resolution or DEFAULT_RESOLUTION)
    return camera.projection_matrix(math.radians(90.), w / h)
