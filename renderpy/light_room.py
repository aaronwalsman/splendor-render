#!/usr/bin/env python
import random
import os
import sys
import json

import tqdm

import numpy

from image import save_image
from renderpy.reflection_maps_from_scene import reflection_maps_from_scene
from renderpy.reflection_to_diffuse import reflection_to_diffuse

def sample_color(
        intensity_range,
        color_offset_range):
    
    intensity = random.uniform(*intensity_range)
    offset = random.uniform(*color_offset_range)
    color = [min(max(intensity + random.uniform(-1,1) * offset, 0), 1)
            for _ in range(3)]
    
    return color

def sample_window(
        p_disc = 0.0,
        scale_range = (0.1, 0.5),
        intensity_range = (0.8, 1.0),
        color_offset_range = (0, 0.2)):
    
    is_disc = random.random() < p_disc
    if is_disc:
        light_type = 'disc'
        radius = random.uniform(*scale_range)
        scale = [radius, radius, radius]
    else:
        light_type = 'rect'
        scale = [random.uniform(*scale_range) for _ in range(3)]
    
    direction = random.choice(((-1,0,0),(1,0,0),(0,0,-1),(0,0,1)))
    if direction == (-1, 0, 0) or direction == (1, 0, 0):
        position = (-direction[0], random.uniform(-1,1), random.uniform(-1,1))
    else:
        position = (random.uniform(-1,1), random.uniform(-1,1), -direction[2])
    
    light = {
        'type' : light_type,
        'position' : position,
        'direction' : direction,
        'up_vector' : (0,1,0),
        'scale' : scale,
        'color' : sample_color(intensity_range, color_offset_range)
    }
    
    return light

def sample_overhead_light(
        p_disc = 0.8,
        scale_range = (0.01, 0.1),
        intensity_range = (0.8, 1.0),
        color_offset_range = (0, 0.2)):
    
    is_disc = random.random() < p_disc
    if is_disc:
        light_type = 'disc'
        radius = random.uniform(*scale_range)
        scale = [radius, radius, radius]
    else:
        light_type = 'rect'
        scale = [random.uniform(*scale_range) for _ in range(3)]
    
    position = [random.uniform(-1, 1), 1.0, random.uniform(-1, 1)]
    
    light = {
        'type' : light_type,
        'position' : position,
        'direction' : (0,-1,0),
        'up_vector' : (0,0,1),
        'scale' : scale,
        'color' : sample_color(intensity_range, color_offset_range)
    }
    
    return light

def sample_room(
        overhead_light_range = (0,3),
        window_range = (0,4),
        background_intensity_range = (0.1, 0.4),
        background_color_offset_range = (0.0, 0.1)):
    
    total_lights = 0
    while total_lights == 0:
        num_overhead_lights = random.randint(*overhead_light_range)
        num_windows = random.randint(*window_range)
        total_lights = num_overhead_lights + num_windows
    
    background_color = sample_color(
            background_intensity_range, background_color_offset_range)
    
    lights = [
            sample_overhead_light() for _ in range(num_overhead_lights)] + [
            sample_window() for _ in range(num_windows)]
    
    return make_light_room(lights, background_color)

def sample_and_export_rooms(
        num_rooms,
        destination):
    
    for i in tqdm.tqdm(range(num_rooms)):
        scene = sample_room()
        reflection_maps = reflection_maps_from_scene(
                scene,
                numpy.eye(4),
                512,
                512,
                fake_hdri=False)
        
        diffuse_maps = reflection_to_diffuse(
                reflection_maps,
                128)
        
        out_dir = os.path.join(destination, 'room_%i'%i)
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        
        for name, reflection_map in reflection_maps.items():
            diffuse_map = diffuse_maps[name]
            save_image(reflection_map, os.path.join(out_dir, '%s_ref.png'%name))
            save_image(diffuse_map, os.path.join(out_dir, '%s_dif.png'%name))

def make_light_room(
        lights = (),
        background_color = (0,0,0)):
    
    scene_description = {
        'meshes' : {},
        'materials' : {},
        'instances' : {},
        'background_color' : background_color,
        #'camera' : {
        #    'pose':[0.0, 0.0, 0, 0, 0, 0],
        #    'projection':[
        #            [1.0,   0,  0,    0],
        #            [  0, 1.0,  0,    0],
        #            [  0,   0, -1, -0.1],
        #            [  0,   0, -1,    0]],
        #    'pose_delta' : [0.0004,0.0001,0,0,0,0]
        #},
        #'ambient_color' : (1,1,1)
    }
    
    scene_description['meshes']['disc'] = {
        'mesh_path' : './example_meshes/disc.obj'
    }
    scene_description['meshes']['rect'] = {
        'mesh_path' : './example_meshes/rect.obj'
    }
    
    for i, light in enumerate(lights):
        sx, sy, sz = light['scale']
        scale_transform = numpy.array([
                [sx, 0,  0, 0],
                [0, sy,  0, 0],
                [0,  0, sz, 0],
                [0,  0,  0, 1]])
        direction = numpy.array(light['direction'])
        up_vector = numpy.array(light['up_vector'])
        cross_vector = numpy.cross(up_vector, direction)
        cross_vector = cross_vector / numpy.linalg.norm(cross_vector)
        double_cross = numpy.cross(direction, cross_vector)
        orientation_transform = numpy.array([
                [cross_vector[0], double_cross[0], direction[0], 0],
                [cross_vector[1], double_cross[1], direction[1], 0],
                [cross_vector[2], double_cross[2], direction[2], 0],
                [              0,               0,            0, 1]])
        tx,ty,tz = light['position']
        translate_transform = numpy.array([
                [1, 0, 0, tx],
                [0, 1, 0, ty],
                [0, 0, 1, tz],
                [0, 0, 0,  1]])
        transform = numpy.dot(translate_transform, orientation_transform)
        transform = numpy.dot(transform, scale_transform)
        scene_description['instances']['disc_%i'%i] = {
            'mesh_name' : light['type'],
            'material_name' : 'mat_%i'%i,
            'transform' : transform.tolist(),
        }
        texture = numpy.ones((16,16,3)) * 255
        texture[:,:] *= light['color']
        texture = texture.astype(numpy.uint8).tolist()
        scene_description['materials']['mat_%i'%i] = {
            'texture' : texture
        }
    
    return scene_description

if __name__ == '__main__':
    sample_and_export_rooms(int(sys.argv[1]), sys.argv[2])
