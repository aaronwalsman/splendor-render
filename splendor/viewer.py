"""Interactive scene viewer with orbit camera and hotkeys."""
import time
import math
import os

import numpy

from splendor.contexts.glfw import GLFWContext
import splendor.core as core
import splendor.camera as camera
from splendor.interactive_camera_glfw import InteractiveCameraGLFW
from splendor.image import save_image


def start_viewer(
    file_path,
    width=512,
    height=512,
    poll_frequency=64,
    anti_alias=True,
    anti_alias_samples=8,
    assets=None,
    print_fps=False,
):
    """Launch an interactive viewer window for a scene file. Reloads the scene automatically when the file changes on disk."""
    with GLFWContext(width=width, height=height,
                    anti_alias=anti_alias,
                    anti_alias_samples=anti_alias_samples) as ctx:
        renderer = core.SplendorRender(assets=assets)

        file_path = renderer.asset_library['scenes'][file_path]
        projection = camera.projection_matrix(math.radians(90.), width/height)
        renderer.load_camera('main', projection=projection)

        def load_display_sensor():
            renderer.load_sensor(
                'display', width, height,
                enable_radial_distortion=True,
                anti_alias=anti_alias,
                anti_alias_samples=anti_alias_samples,
            )

        load_display_sensor()

        state = {
            'steps': 0,
            'recent_file_change_time': -1,
            'batch_time': time.time(),
            'render_mode': 'color',
        }

        def reload_scene():
            while True:
                try:
                    change_time = os.stat(file_path).st_mtime
                    if change_time != state['recent_file_change_time']:
                        prior_view = renderer.get_camera_view_matrix('main')
                        first_load = state['recent_file_change_time'] == -1

                        renderer.load_scene(file_path, clear_scene=True)

                        # Grab whatever view_matrix the scene specified, if any
                        scene_view = (
                            renderer.get_camera_view_matrix('main')
                            if renderer.camera_exists('main') else None
                        )

                        # Restore viewer's projection (scene may have set its own)
                        k1 = (renderer.get_camera_radial_k1('main')
                              if renderer.camera_exists('main') else 0.)
                        k2 = (renderer.get_camera_radial_k2('main')
                              if renderer.camera_exists('main') else 0.)
                        renderer.load_camera(
                            'main', projection=projection,
                            radial_k1=k1, radial_k2=k2)

                        if first_load and scene_view is not None:
                            # Use the scene's initial camera pose
                            renderer.set_camera_view_matrix('main', scene_view)
                        elif not first_load:
                            # Preserve the user's current camera pose
                            renderer.set_camera_view_matrix('main', prior_view)

                        load_display_sensor()
                        state['recent_file_change_time'] = change_time
                        print('Loaded: %s' % file_path)
                except (OSError, ValueError) as e:
                    print('Warning: could not load file: %s (%s)' % (
                        file_path, e))
                    break
                except Exception:
                    print('Unable to load file: %s' % file_path)
                    raise
                else:
                    break

        def render():
            if state['steps'] % poll_frequency == 0:
                reload_scene()
                t_now = time.time()
                if print_fps:
                    print('fps: %.04f' % (
                        poll_frequency / (t_now - state['batch_time'])))
                state['batch_time'] = t_now
            state['steps'] += 1

            fbw, fbh = ctx.framebuffer_size()
            renderer.viewport_scissor(0, 0, fbw, fbh)

            if state['render_mode'] == 'color':
                renderer.color_render('main', sensor='display', flip_y=False)
            elif state['render_mode'] == 'mask':
                renderer.mask_render('main', sensor='display', flip_y=False)
            renderer.display_sensor('display')

        camera_control = InteractiveCameraGLFW(ctx, renderer, 'main')
        ctx.set_mouse_button_callback(camera_control.mouse_callback)
        ctx.set_cursor_pos_callback(camera_control.mouse_move)
        ctx.set_scroll_callback(camera_control.scroll_callback)

        import glfw as _glfw
        def key_callback(window, key, scancode, action, mods):
            if action == _glfw.PRESS:
                if key == _glfw.KEY_S:
                    image = renderer.read_sensor('display')
                    stamp = time.strftime('%Y%m%d_%H%M%S')
                    path = os.path.join(os.getcwd(), 'splendor_%s.png' % stamp)
                    save_image(image, path)
                    print('Saved screenshot: %s' % path)
                elif key == _glfw.KEY_M:
                    modes = ['color', 'mask']
                    state['render_mode'] = modes[
                        (modes.index(state['render_mode']) + 1) % len(modes)]
                    print('Render mode: %s' % state['render_mode'])
            camera_control.key_callback(window, key, scancode, action, mods)

        ctx.set_key_callback(key_callback)

        while not ctx.should_close():
            ctx.poll_events()
            render()
            ctx.swap_buffers()
