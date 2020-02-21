#!/usr/bin/env python
import time
import sys
import os
import numpy
import buffer_manager
import core

def spin(
        scene_file,
        width = 512,
        height = 512,
        poll_frequency = 1024):
    
    manager = buffer_manager.initialize_shared_buffer_manager(width, height)
    renderer = core.Renderpy()
    
    manager.show_window()
    manager.enable_window()
    
    render_state = {
        'static_camera_pose' : numpy.eye(4),
        'camera_pose_delta' : None,
        'integrated_camera_pose_delta' : numpy.eye(4),
        'recent_change_time' : -1
    }
    
    def reload_scene():
        while True:
            try:
                change_time = os.stat(scene_file).st_mtime
                if change_time != render_state['recent_change_time']:
                    renderer.load_scene(scene_file, clear_existing=True)
                    render_state['recent_change_time'] = change_time
                    render_state['static_camera_pose'] = numpy.linalg.inv(
                            renderer.get_camera_pose())
                    render_state['camera_pose_delta'] = (
                            renderer.get_camera_pose_delta())
                    if render_state['camera_pose_delta'] is None:
                        render_state['integrated_camera_pose_delta'] = (
                                numpy.eye(4))
                    print('Loaded: %s'%scene_file)
            except:
                print('Unable to load file: %s'%scene_file)
                time.sleep(1)
                print('Retrying...')
            else:
                break
    
    def render():
        renderer.color_render(flip_y = False)
    
    steps = 0
    pose_delta = numpy.eye(4)
    integrated_pose_delta = numpy.eye(4)
    while True:
        if steps % poll_frequency == 0:
            # reload the scene if necessary
            reload_scene()
        steps += 1
        
        if render_state['camera_pose_delta'] is not None:
            render_state['integrated_camera_pose_delta'] = numpy.dot(
                    render_state['integrated_camera_pose_delta'],
                    render_state['camera_pose_delta'])
        camera_pose = numpy.dot(
                render_state['integrated_camera_pose_delta'],
                render_state['static_camera_pose'])
        renderer.set_camera_pose(numpy.linalg.inv(camera_pose))
        
        render()

if __name__ == '__main__':
    width = 512
    height = 512
    
    if len(sys.argv) < 2:
        raise Exception('Please specify one scene file')
    
    scene_file = sys.argv[1]
    
    spin(   scene_file,
            width=width,
            height=height)
