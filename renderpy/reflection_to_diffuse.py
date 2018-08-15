#!/usr/bin/env python

# system
import random
import math

# opengl
from OpenGL.GL import *
from OpenGL.GL import shaders
from OpenGL.arrays import vbo

# numpy
import numpy

# imageio
import imageio

# local
import shader_definitions
import buffer_manager
import camera

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
        contrast = 1):
    
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
    
    # color scale
    # this first color scale ensures that the range of values written to the
    # image are not too compressed
    pre_color_scale = 0.
    for cube_face in reflection_cube:
        max_channel = numpy.max(cube_face, axis=0) / 255
        pre_color_scale += numpy.sum(max_channel) / max_channel.size
    pre_color_scale /= 6.
    pre_color_scale = 2./pre_color_scale
    
    glUniform1f(color_scale_location, pre_color_scale)
    
    # direction samples
    directions = numpy.zeros((NUM_SAMPLES,3))
    for i in range(NUM_SAMPLES):
        directions[i] = sample_sphere_surface()
    
    glUniform3fv(direction_location, NUM_SAMPLES, directions)
    
    # textures
    texture_buffer = glGenTextures(1)
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_CUBE_MAP, texture_buffer)
    for i, cube_face in enumerate(reflection_cube):
        glTexImage2D(
                GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                0, GL_RGB, cube_face.shape[1], cube_face.shape[0],
                0, GL_RGB, GL_UNSIGNED_BYTE, cube_face)
    
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    
    # cameras
    projection_matrix = camera.projection_matrix(
            math.radians(90), 1.0, 0.01, 1.0)
    
    camera_poses = {
            'nz' : numpy.array([
                [-1, 0, 0, 0],
                [ 0,-1, 0, 0],
                [ 0, 0, 1, 0],
                [ 0, 0, 0, 1]]),
            'pz' : numpy.array([
                [ 1, 0, 0, 0],
                [ 0,-1, 0, 0],
                [ 0, 0,-1, 0],
                [ 0, 0, 0, 1]]),
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
                [ 0, 0, 0, 1]])}
    
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
    
    return output_images

if __name__ == '__main__':
    cube_images = []
    for cube_face in 'px', 'nx', 'py', 'ny', 'pz', 'nz':
        image = numpy.array(imageio.imread(
                '/home/awalsman/Development/cube_maps/blue_cave/%s.png'%
                cube_face))
        cube_images.append(image[:,:,:3])
    
    out_images = reflection_to_diffuse(cube_images, 128)
    for cube_face in out_images:
        imageio.imsave('./%s.png'%cube_face, out_images[cube_face])
