#!/usr/bin/env python
import math
import sys
import os
import numpy
import PIL.Image as Image

from . import buffer_manager
from . import core
from . import camera

def reflection_maps_from_scene(
        scene,
        camera_pose,
        width,
        height,
        ignore_instances=[],
        light_instances=[],
        ambient_color=(1.0,1.0,1.0),
        fake_hdri=True):
    
    manager = buffer_manager.initialize_shared_buffer_manager(width, height)
    manager.add_frame('reflection', width, height)
    manager.enable_frame('reflection')
    
    renderer = core.Renderpy()
    
    renderer.load_scene(scene, clear_existing=True)
    #renderer.set_active_image_light = None
    renderer.set_ambient_color(ambient_color)
    
    camera_projection = camera.projection_matrix(math.radians(90.), 1.)
    renderer.set_projection(camera_projection)
    camera_offsets = {
            'px' : numpy.array([
                [ 0, 0,-1, 0],
                [ 0, 1, 0, 0],
                [ 1, 0, 0, 0],
                [ 0, 0, 0, 1]]),
            'nx' : numpy.array([
                [ 0, 0, 1, 0],
                [ 0, 1, 0, 0],
                [-1, 0, 0, 0],
                [ 0, 0, 0, 1]]),
            'py' : numpy.array([
                [-1, 0, 0, 0],
                [ 0, 0,-1, 0],
                [ 0,-1, 0, 0],
                [ 0, 0, 0, 1]]),
            'ny' : numpy.array([
                [-1, 0, 0, 0],
                [ 0, 0, 1, 0],
                [ 0, 1, 0, 0],
                [ 0, 0, 0, 1]]),
            'pz' : numpy.array([
                [-1, 0, 0, 0],
                [ 0, 1, 0, 0],
                [ 0, 0,-1, 0],
                [ 0, 0, 0, 1]]),
            'nz' : numpy.array([
                [ 1, 0, 0, 0],
                [ 0, 1, 0, 0],
                [ 0, 0, 1, 0],
                [ 0, 0, 0, 1]])}
    
    output_images = {}
    for name, offset in camera_offsets.items():
        pose = numpy.dot(offset, camera_pose)
        renderer.set_camera_pose(pose)
        renderer.color_render()
        output_image = manager.read_pixels('reflection')
        
        if fake_hdri:
            renderer.mask_render()
            brightness_mask = manager.read_pixels('reflection')
            brightness_mask = brightness_mask.astype(numpy.float32) / 255.
            output_image = (output_image * brightness_mask).astype(numpy.uint8)
        
        output_images[name] = output_image
    
    return output_images

if __name__ == '__main__':
    scene_file = sys.argv[1]
    output_dir = sys.argv[2]
    output_images = reflection_maps_from_scene(
            scene_file,
            numpy.eye(4),
            256,
            256)
    for name, image in output_images.items():
        image = Image.fromarray(image)
        image.save(os.path.join(output_dir, '%s_ref.png'%name))

