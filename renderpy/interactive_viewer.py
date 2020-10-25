#!/usr/bin/env python
import time
import sys
import os
import numpy
import buffer_manager_glut as buffer_manager
import core
import camera

def start_viewer(
        file_path,
        width = 512,
        height = 512,
        poll_frequency = 1024):
    
    manager = buffer_manager.initialize_shared_buffer_manager(width, height)
    renderer = core.Renderpy()
    manager.show_window()
    manager.enable_window()
    
    state = {
        'steps' : 0,
        'recent_file_change_time' : -1,
        'mouse_button' : -1,
        'mouse_click_position' : (0,0),
        'mouse_click_depth' : 0
    }
    
    def reload_scene():
        while True:
            try:
                change_time = os.stat(file_path).st_mtime
                if change_time != state['recent_file_change_time']:
                    camera_pose = renderer.get_camera_pose()
                    renderer.load_scene(file_path, clear_existing=True)
                    if state['recent_file_change_time'] != -1:
                        renderer.set_camera_pose(camera_pose)
                    state['recent_file_change_time'] = change_time
                    print('Loaded: %s'%file_path)
            except:
                print('Unable to load file: %s'%scene_file)
                time.sleep(1)
                print('Retrying...')
            else:
                break
    
    def mouse_button(button, button_state, x, y):
        if button in (0,2,3,4):
            if button_state == 0:
                depth = manager.read_pixels(
                        None,
                        read_depth = True,
                        near = 50,
                        far = 5000)
                state['mouse_button'] = button
                state['mouse_click_position'] = (x,y)
                state['mouse_click_depth'] = depth[height-y,x]
                
                if button in (3,4):
                    camera_pose = renderer.get_camera_pose()
                    z_direction = camera_pose[2,0:3]
                    
                    distance = 0.1 * state['mouse_click_depth']
                    z_offset = z_direction * distance
                    if button == 3:
                        z_offset *= -1.
                    
                    camera_pose = numpy.linalg.inv(camera_pose)
                    camera_pose[0:3,3] += z_offset
                    camera_pose = numpy.linalg.inv(camera_pose)
                    renderer.set_camera_pose(camera_pose)
                    
            else:
                state['mouse_button'] = -1
    
    def mouse_move(x, y):
        if state['mouse_button'] == 0:
            # orbit
            delta_x = (x - state['mouse_click_position'][0])/width
            delta_y = (y - state['mouse_click_position'][1])/height
            camera_pose = renderer.get_camera_pose()
            camera_pose = numpy.linalg.inv(camera_pose)
            
            inverse_pivot = numpy.eye(4)
            inverse_pivot[2,3] = state['mouse_click_depth']
            pivot = numpy.eye(4)
            pivot[2,3] = -state['mouse_click_depth']
            
            azimuthal_pose = [delta_x*2, delta_y*2, 0, 0, 0, 0]
            pose_offset = camera.azimuthal_pose_to_matrix(azimuthal_pose)
            pose_offset = numpy.dot(numpy.dot(pivot, pose_offset),
                    inverse_pivot)
            camera_pose = numpy.dot(camera_pose, pose_offset)
            camera_pose = numpy.linalg.inv(camera_pose)
            renderer.set_camera_pose(camera_pose)
            
            state['mouse_click_position'] = (x,y)
        
        if state['mouse_button'] == 2:
            # pan
            delta_x = (x - state['mouse_click_position'][0])/width
            delta_y = (y - state['mouse_click_position'][1])/height
            camera_pose = renderer.get_camera_pose()
            x_direction = camera_pose[0,0:3]
            y_direction = camera_pose[1,0:3]
            
            x_offset = -x_direction * delta_x * state['mouse_click_depth']
            y_offset = y_direction * delta_y * state['mouse_click_depth']
            
            camera_pose = numpy.linalg.inv(camera_pose)
            camera_pose[0:3,3] += x_offset + y_offset
            camera_pose = numpy.linalg.inv(camera_pose)
            renderer.set_camera_pose(camera_pose)
            
            state['mouse_click_position'] = (x,y)
    
    def render():
        if state['steps'] % poll_frequency == 0:
            reload_scene()
        state['steps'] += 1
        renderer.color_render(flip_y=False)
    
    manager.start_main_loop(
            glutDisplayFunc = render,
            glutIdleFunc = render,
            glutMouseFunc = mouse_button,
            glutMotionFunc = mouse_move)

if __name__ == '__main__':
    width = 512
    height = 512
    
    if len(sys.argv) < 2:
        raise Exception('Please specify one scene file')
    
    file_path = sys.argv[1]
    
    start_viewer(
            file_path,
            width=width,
            height=height)