import time
import math
import sys
import os

import numpy

import renderpy.contexts.glut as glut
import renderpy.core as core
import renderpy.camera as camera
from renderpy.interactive_camera import InteractiveCamera

def start_double_viewer(
        file_path,
        width = 512,
        height = 512,
        poll_frequency = 1024):
    
    glut.initialize()
    color_window = glut.GlutWindowWrapper(
            'Color', width, height)
    color_renderer = core.Renderpy()
    mask_window = glut.GlutWindowWrapper(
            'Mask', width, height)
    mask_renderer = core.Renderpy()
    
    file_path = color_renderer.asset_library['scenes'][file_path]
    
    camera_control = InteractiveCamera(color_window, color_renderer)
    
    state = {
        'steps' : 0,
        'recent_file_change_time' : -1,
    }
    
    def reload_scene():
        while True:
            try:
                change_time = os.stat(file_path).st_mtime
                if change_time != state['recent_file_change_time']:
                    camera_pose = color_renderer.get_camera_pose()
                    color_renderer.load_scene(file_path, clear_scene=True)
                    mask_renderer.load_scene(file_path, clear_scene=True)
                    if state['recent_file_change_time'] != -1:
                        color_renderer.set_camera_pose(camera_pose)
                        mask_renderer.set_camera_pose(camera_pose)
                    state['recent_file_change_time'] = change_time
                    print('Loaded: %s'%file_path)
            except:
                print('Unable to load file: %s'%file_path)
                raise
                time.sleep(1)
                print('Retrying...')
            else:
                break
    
    def render_color():
        if state['steps'] % poll_frequency == 0:
            reload_scene()
        state['steps'] += 1
        color_window.set_active()
        #color_window.enable_window()
        color_renderer.color_render(flip_y=False)
        
        #mask_window.enable_window()
        #renderer.mask_render(flip_Y=False)
    
    def render_mask():
        if state['steps'] % poll_frequency == 0:
            reload_scene()
        state['steps'] += 1
        mask_window.set_active()
        #mask_window.enable_window()
        mask_renderer.mask_render(flip_y=False)
    
    color_window.register_callbacks(
            glutDisplayFunc = render_color,
            glutIdleFunc = render_color,
            glutMouseFunc = camera_control.mouse_button,
            glutMotionFunc = camera_control.mouse_move,
    )
    mask_window.register_callbacks(
            glutDisplayFunc = render_mask,
            glutIdleFunc = render_mask,
            #glutMouseFunc = camera_control.mouse_button,
            #glutMotionFunc = camera_control.mouse_move,
    )
    glut.start_main_loop()

def start_viewer(
        file_path,
        width = 512,
        height = 512,
        poll_frequency = 1024,
        anti_alias = True,
        anti_alias_samples = 8):

    glut.initialize()
    window = glut.GlutWindowWrapper('Color', width, height)
    
    renderer = core.Renderpy()
    window.show_window()
    window.enable_window()

    file_path = renderer.asset_library['scenes'][file_path]
    
    projection = camera.projection_matrix(math.radians(90.), width/height)
    camera_control = InteractiveCamera(window, renderer)

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
        state['steps'] += 1
        renderer.color_render(flip_y=False)
    
    window.register_callbacks(
            glutDisplayFunc = render,
            glutIdleFunc = render,
            glutMouseFunc = camera_control.mouse_button,
            glutMotionFunc = camera_control.mouse_move)
    
    glut.start_main_loop()
