import time
import math
import os

import numpy

from splendor.contexts.glfw import GLFWContext
import splendor.core as core
import splendor.camera as camera
from splendor.interactive_camera_glfw import InteractiveCameraGLFW


def start_viewer(
    file_path,
    width=512,
    height=512,
    poll_frequency=64,
    assets=None,
    print_fps=False,
):
    with GLFWContext(width=width, height=height) as ctx:
        renderer = core.SplendorRender(assets=assets)

        file_path = renderer.asset_library['scenes'][file_path]
        projection = camera.projection_matrix(math.radians(90.), width/height)

        state = {
            'steps': 0,
            'recent_file_change_time': -1,
            'batch_time': time.time(),
        }

        def reload_scene():
            while True:
                try:
                    change_time = os.stat(file_path).st_mtime
                    if change_time != state['recent_file_change_time']:
                        view_matrix = renderer.get_view_matrix()
                        renderer.load_scene(file_path, clear_scene=True)
                        if state['recent_file_change_time'] != -1:
                            renderer.set_view_matrix(view_matrix)
                        renderer.set_projection(projection)
                        state['recent_file_change_time'] = change_time
                        print('Loaded: %s' % file_path)
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
            renderer.color_render(flip_y=False)

        camera_control = InteractiveCameraGLFW(ctx, renderer)
        ctx.set_mouse_button_callback(camera_control.mouse_callback)
        ctx.set_cursor_pos_callback(camera_control.mouse_move)
        ctx.set_key_callback(camera_control.key_callback)
        ctx.set_scroll_callback(camera_control.scroll_callback)

        while not ctx.should_close():
            ctx.poll_events()
            render()
            ctx.swap_buffers()
