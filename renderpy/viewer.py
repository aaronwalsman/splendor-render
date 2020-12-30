import time
import sys
import os

import numpy

import renderpy.buffer_manager_glut as buffer_manager
import renderpy.core as core
import renderpy.camera as camera
from renderpy.interactive_camera import InteractiveCamera

def start_viewer(
        file_path,
        width = 512,
        height = 512,
        poll_frequency = 1024,
        anti_alias = True,
        anti_alias_samples = 8):

    manager = buffer_manager.initialize_shared_buffer_manager(
            width, height, anti_alias, anti_alias_samples)
    renderer = core.Renderpy()
    manager.show_window()
    manager.enable_window()

    file_path = renderer.asset_library['scenes'][file_path]

    camera_control = InteractiveCamera(manager, renderer)

    state = {
        'steps' : 0,
        'recent_file_change_time' : -1,
    }
    
    def reload_scene():
        while True:
            try:
                change_time = os.stat(file_path).st_mtime
                if change_time != state['recent_file_change_time']:
                    camera_pose = renderer.get_camera_pose()
                    renderer.load_scene(file_path, clear_scene=True)
                    if state['recent_file_change_time'] != -1:
                        renderer.set_camera_pose(camera_pose)
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
        state['steps'] += 1
        renderer.color_render(flip_y=False)

    manager.start_main_loop(
            glutDisplayFunc = render,
            glutIdleFunc = render,
            glutMouseFunc = camera_control.mouse_button,
            glutMotionFunc = camera_control.mouse_move)
