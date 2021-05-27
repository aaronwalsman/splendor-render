import time
import math
import sys
import os

import numpy

import splendor.contexts.glut as glut
import splendor.core as core
import splendor.camera as camera
from splendor.interactive_camera import InteractiveCamera

def start_viewer(
    file_path,
    width = 512,
    height = 512,
    poll_frequency = 1024,
    anti_alias = True,
    anti_alias_samples = 8,
    assets = None,
    print_fps = False,
):

    glut.initialize()
    window = glut.GlutWindowWrapper('Color', width, height)
    
    renderer = core.SplendorRender(assets=assets)
    window.show_window()
    window.enable_window()

    file_path = renderer.asset_library['scenes'][file_path]
    
    projection = camera.projection_matrix(math.radians(90.), width/height)
    camera_control = InteractiveCamera(window, renderer)

    state = {
        'steps' : 0,
        'recent_file_change_time' : -1,
        'batch_time' : time.time()
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
                    print('Loaded: %s'%file_path)
            except:
                print('Unable to load file: %s'%file_path)
                raise
                time.sleep(1)
                print('Retrying...')
            else:
                break
    
    def render():
        if state['steps'] % poll_frequency == 0:
            reload_scene()
            t_now = time.time()
            if print_fps:
                print('fps: %.04f'%(
                        poll_frequency / (t_now - state['batch_time'])))
            state['batch_time'] = t_now
        state['steps'] += 1
        renderer.color_render(flip_y=False)
    
    window.register_callbacks(
            glutDisplayFunc = render,
            glutIdleFunc = render,
            glutMouseFunc = camera_control.mouse_button,
            glutMotionFunc = camera_control.mouse_move)
    
    glut.start_main_loop()
