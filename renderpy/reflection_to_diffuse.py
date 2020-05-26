#!/usr/bin/env python

# system
import random
import math
import os
import sys

# opengl
from OpenGL.GL import *
from OpenGL.GL import shaders
from OpenGL.arrays import vbo

# numpy
import numpy

# local
from . import shader_definitions
from . import buffer_manager_egl as buffer_manager
from . import camera
from .image import load_image, save_image

NUM_SAMPLES = 512

def sample_sphere_surface():
    while True:
        point = numpy.random.random(3) * 2 - 1.
        n = numpy.sum(point**2)
        if n <= 1.:
            return point / (n**0.5)

def reflection_to_diffuse(
        reflection_cube,
        width,
        brightness = 0,
        contrast = 1,
        importance_threshold = 0.95,
        num_importance_samples = 64,
        importance_sample_gain = 0.3,
        random_sample_gain = 0.7):
    
    num_random_samples = NUM_SAMPLES - num_importance_samples
    
    # initialize the buffer manager
    manager = buffer_manager.initialize_shared_buffer_manager()
    try:
        manager.add_frame('reflection_to_diffuse', width, width)
    except buffer_manager.FrameExistsError:
        pass
    manager.enable_frame('reflection_to_diffuse')
    
    # compile the shaders
    vertex_shader = shaders.compileShader(
            shader_definitions.background_vertex_shader,
            GL_VERTEX_SHADER)
    fragment_shader = shaders.compileShader(
            shader_definitions.reflection_to_diffuse_fragment_shader,
            GL_FRAGMENT_SHADER)
    program = shaders.compileProgram(vertex_shader, fragment_shader)
    
    glUseProgram(program)
    
    # get shader variable locations
    projection_location = glGetUniformLocation(program, 'projection_matrix')
    camera_location = glGetUniformLocation(program, 'camera_pose')
    brightness_location = glGetUniformLocation(program, 'brightness')
    contrast_location = glGetUniformLocation(program, 'contrast')
    color_scale_location = glGetUniformLocation(program, 'color_scale')
    direction_location = glGetUniformLocation(program, 'sphere_samples')
    num_importance_sample_location = glGetUniformLocation(
            program, 'num_importance_samples')
    importance_gain_location = glGetUniformLocation(
            program, 'importance_sample_gain')
    random_gain_location = glGetUniformLocation(
            program, 'random_sample_gain')

    sampler_location = glGetUniformLocation(program, 'reflection_sampler')
    glUniform1i(sampler_location, 0)
    
    # load mesh
    vertex_floats = numpy.array([
            [-1,-1,0],
            [-1, 1,0],
            [ 1, 1,0],
            [ 1,-1,0]])
    vertex_buffer = vbo.VBO(vertex_floats)
    
    face_ints = numpy.array([
            [0,1,2],
            [2,3,0]], dtype=numpy.int32)
    face_buffer = vbo.VBO(face_ints, target = GL_ELEMENT_ARRAY_BUFFER)
    
    vertex_buffer.bind()
    face_buffer.bind()
    
    # brightness/contrast
    glUniform1f(brightness_location, brightness)
    glUniform1f(contrast_location, contrast)
    
    '''
    # color scale
    # this first color scale ensures that the range of values written to the
    # image are not too compressed
    pre_color_scale = 0.
    for cube_face in reflection_cube:
        cube_image = reflection_cube[cube_face]
        max_channel = numpy.max(cube_image, axis=0) / 255
        pre_color_scale += numpy.sum(max_channel) / max_channel.size
    pre_color_scale /= 6.
    pre_color_scale = 2./pre_color_scale
    '''
    
    directions = numpy.zeros((NUM_SAMPLES,3))
    
    # importance direction samples
    intensity_cube = {}
    max_intensity = 0
    pre_color_scale = 0
    for cube_face in reflection_cube:
        image = reflection_cube[cube_face].astype(numpy.float32)
        intensity_cube[cube_face] = (
                0.2989 * image[:,:,0] +
                0.5870 * image[:,:,1] +
                0.1140 * image[:,:,2])
        pre_color_scale += (
                numpy.sum(intensity_cube[cube_face]) /
                (numpy.sum(intensity_cube[cube_face].size) * 255.))
        max_intensity = max(max_intensity, numpy.max(intensity_cube[cube_face]))
    
    pre_color_scale /= 6.
    pre_color_scale = 0.5/pre_color_scale
    glUniform1f(color_scale_location, pre_color_scale)
    
    intensity_threshold = max_intensity * importance_threshold
    pixel_locations = []
    for cube_face in intensity_cube:
        y,x = numpy.where(
                intensity_cube[cube_face] > intensity_threshold)
        for i in range(len(x)):
            pixel_locations.append((cube_face, x[i], y[i]))
    
    camera_poses = {
            'px' : numpy.array([
                [ 0, 0,-1, 0],
                [ 0,-1, 0, 0],
                [-1, 0, 0, 0],
                [ 0, 0, 0, 1]]),
            'nx' : numpy.array([
                [ 0, 0, 1, 0],
                [ 0,-1, 0, 0],
                [ 1, 0, 0, 0],
                [ 0, 0, 0, 1]]),
            'py' : numpy.array([
                [ 1, 0, 0, 0],
                [ 0, 0, 1, 0],
                [ 0,-1, 0, 0],
                [ 0, 0, 0, 1]]),
            'ny' : numpy.array([
                [ 1, 0, 0, 0],
                [ 0, 0,-1, 0],
                [ 0, 1, 0, 0],
                [ 0, 0, 0, 1]]),
            'pz' : numpy.array([
                [ 1, 0, 0, 0],
                [ 0,-1, 0, 0],
                [ 0, 0,-1, 0],
                [ 0, 0, 0, 1]]),
            'nz' : numpy.array([
                [-1, 0, 0, 0],
                [ 0,-1, 0, 0],
                [ 0, 0, 1, 0],
                [ 0, 0, 0, 1]])}
    
    face_rotations = {
            'px':numpy.array([
                    [ 0, 0,-1],
                    [ 0,-1, 0],
                    [-1, 0, 0]]),
            'nx':numpy.array([
                    [ 0, 0, 1],
                    [ 0,-1, 0],
                    [ 1, 0, 0]]),
            'py':numpy.array([
                    [ 1, 0, 0],
                    [ 0, 0,-1],
                    [ 0, 1, 0]]),
            'ny':numpy.array([
                    [ 1, 0, 0],
                    [ 0, 0, 1],
                    [ 0,-1, 0]]),
            'pz':numpy.array([
                    [ 1, 0, 0],
                    [ 0,-1, 0],
                    [ 0, 0,-1]]),
            'nz':numpy.array([
                    [-1, 0, 0],
                    [ 0,-1, 0],
                    [ 0, 0, 1]])}
    
    face_width = reflection_cube['px'].shape[0]
    half_width = face_width / 2
    for i in range(num_importance_samples):
        cube_face, x, y = random.choice(pixel_locations)
        position = (x-half_width, y-half_width,-half_width)
        d = (position[0]**2 + position[1]**2 + position[2]**2)**0.5
        direction = numpy.array(
                [position[0]/d, position[1]/d, position[2]/d, 0])
        #direction = numpy.dot(
        #        face_rotations[cube_face], direction)
        direction = numpy.dot(
                numpy.linalg.inv(camera_poses[cube_face]), direction)
        directions[i] = direction[:3]
    
    # random direction samples
    for i in range(num_random_samples):
        directions[num_importance_samples + i] = sample_sphere_surface()
    
    glUniform3fv(direction_location, NUM_SAMPLES, directions)
    glUniform1i(num_importance_sample_location, num_importance_samples)
    glUniform1f(importance_gain_location, importance_sample_gain)
    glUniform1f(random_gain_location, random_sample_gain)
    
    # textures
    texture_buffer = glGenTextures(1)
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_CUBE_MAP, texture_buffer)
    ordered_reflection_cube = [
            reflection_cube['px'],
            reflection_cube['nx'],
            reflection_cube['py'],
            reflection_cube['ny'],
            reflection_cube['pz'],
            reflection_cube['nz']]
    for i, cube_image in enumerate(ordered_reflection_cube):
        glTexImage2D(
                GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                0, GL_RGB, cube_image.shape[1], cube_image.shape[0],
                0, GL_RGB, GL_UNSIGNED_BYTE, cube_image)
    
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    
    # cameras
    projection_matrix = camera.projection_matrix(
            math.radians(90), 1.0, 0.01, 1.0)
    
    glUniformMatrix4fv(projection_location, 1, GL_TRUE, projection_matrix)
    
    output_images = {}
    
    max_color = 0.
    for cube_face in camera_poses:
        camera_pose = camera_poses[cube_face]
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUniformMatrix4fv(camera_location, 1, GL_TRUE, camera_pose)
        glDrawElements(GL_TRIANGLES, 2*3, GL_UNSIGNED_INT, None)
        output_images[cube_face] = (
                manager.read_pixels('reflection_to_diffuse'))
        max_color = max(max_color, numpy.max(output_images[cube_face]))
    
    for cube_face in output_images:
        output_images[cube_face] = (
                output_images[cube_face].astype(numpy.float32) /
                max_color * 255).astype(numpy.uint8)
    
    vertex_buffer.unbind()
    face_buffer.unbind()
    glBindTexture(GL_TEXTURE_CUBE_MAP, 0)
    glDeleteTextures(texture_buffer)
    
    return output_images

if __name__ == '__main__':
    arg_path = sys.argv[1]
    cube_images = {}
    for cube_face in 'px', 'nx', 'py', 'ny', 'pz', 'nz':
        try:
            #image = numpy.array(Image.open(
            #        os.path.join(arg_path, '%s_ref.jpg'%cube_face)))
            image = load_image(os.path.join(arg_path, '%s_ref.jpg'%cube_face))
        except OSError:
            #image = numpy.array(Image.open(
            #        os.path.join(arg_path, '%s_ref.png'%cube_face)))
            image = load_image(os.path.join(arg_path, '%s_ref.png'%cube_face))
        cube_images[cube_face] = image[:,:,:3]
    
    out_images = reflection_to_diffuse(
            cube_images, 128, brightness=0, contrast=1)
    for cube_face in out_images:
        #imageio.imsave(
        #        os.path.join(arg_path, '%s_dif.jpg'%cube_face),
        #        out_images[cube_face])
        #image = Image.fromarray(out_images[cube_face])
        #image.save(os.path.join(arg_path, '%s_dif.jpg'%cube_face))
        save_image(out_images[cube_face],
                os.path.join(arg_path, '%s_dif.jpg'%cube_face))

