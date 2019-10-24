#!/usr/bin/env python
import sys
import os
import numpy
import buffer_manager
import core
import detection_utils.camera_utils as camera_utils
import IPython

if __name__ == '__main__':
    width = 512
    height = 512
    
    if len(sys.argv) < 2:
        raise Exception('Please specify one mesh')
    mesh_path = sys.argv[1]
    texture_path = mesh_path.replace('.obj', '.png')
    if not os.path.isfile(texture_path):
        texture_path = mesh_path.replace('.obj', '.jpg')
    if not os.path.isfile(texture_path):
        texture_path == None
    
    if len(sys.argv) < 3:
        print('No background specified')
        background_path = None
    else:
        background_path = sys.argv[2]
    
    manager = buffer_manager.initialize_shared_buffer_manager(width)
    manager.add_frame('viewer', width, height)
    
    renderer = core.Renderpy()
    renderer.load_mesh(
            'viewer_mesh',
            mesh_path)
    renderer.load_material(
            'viewer_material',
            texture = texture_path)
    renderer.add_instance(
            'viewer_instance',
            mesh_name = 'viewer_mesh',
            material_name = 'viewer_material')
    
    if background_path:
        renderer.load_image_light(
                name = 'viewer_image_light',
                texture_directory = background_path)
    else:
        renderer.set_ambient_color((1,1,1))
    
    manager.show_window()
    manager.enable_window()
    
    camera_data = {}
    camera_data['distance'] = 1.0
    #camera_data['orientation'] = 0.0
    camera_data['azimuth'] = 0.0
    camera_data['elevation'] = 0
    camera_data['spin'] = 0
    
    def set_camera_pose():
        '''
        camera_pose = camera.turntable_pose(
                camera_data['distance'],
                camera_data['orientation'],
                camera_data['elevation'],
                camera_data['spin'])
        '''
        camera_orientation = camera_utils.turntable_orientation(
                camera_data['azimuth'],
                camera_data['elevation'],
                camera_data['spin'])
        viewpoint = camera_utils.orientations_to_framed_mesh_viewpoints(
                [camera_orientation],
                renderer.loaded_data['meshes']['viewer_mesh'],
                renderer.get_projection(),
                [camera_data['distance'], camera_data['distance']])[0]
        renderer.set_camera_pose(viewpoint)
    
    def render():
        set_camera_pose()
        renderer.color_render(flip_y = False)
    
    set_camera_pose()
    render()
    
    IPython.embed()
