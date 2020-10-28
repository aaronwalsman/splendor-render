#!/usr/bin/env python

# system
import os
import sys
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

def panorama_to_cube(panorama_image, cube_width, filter_panorama=True):
    
    # initialize the buffer manager
    manager = buffer_manager.initialize_shared_buffer_manager()
    try:
        manager.add_frame('panorama_to_cube', cube_width, cube_width)
    except buffer_manager.FrameExistsError:
        pass
    manager.enable_frame('panorama_to_cube')
    
    # compile the shaders
    vertex_shader = shaders.compileShader(
            shader_definitions.background_vertex_shader,
            GL_VERTEX_SHADER)
    fragment_shader = shaders.compileShader(
            shader_definitions.panorama_to_cube_fragment_shader,
            GL_FRAGMENT_SHADER)
    program = shaders.compileProgram(vertex_shader, fragment_shader)
    
    glUseProgram(program)
    
    # get shader variable locations
    projection_location = glGetUniformLocation(program, 'projection_matrix')
    camera_location = glGetUniformLocation(program, 'camera_pose')
    sampler_location = glGetUniformLocation(program, 'texture_sampler')
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
    
    # textures
    texture_buffer = glGenTextures(1)
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, texture_buffer)
    glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGB,
            panorama_image.shape[1], panorama_image.shape[0], 0,
            GL_RGB, GL_UNSIGNED_BYTE, panorama_image)
    
    if filter_panorama:
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                GL_LINEAR_MIPMAP_LINEAR)
        glGenerateMipmap(GL_TEXTURE_2D)
    else:
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT)
    
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
    
    for cube_face in camera_poses:
        camera_pose = camera_poses[cube_face]
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUniformMatrix4fv(camera_location, 1, GL_TRUE, camera_pose)
        glDrawElements(GL_TRIANGLES, 2*3, GL_UNSIGNED_INT, None)
        output_images[cube_face] = (
                manager.read_pixels('panorama_to_cube'))
    
    vertex_buffer.unbind()
    face_buffer.unbind()
    glBindTexture(GL_TEXTURE_2D, 0)
    glDeleteTextures(texture_buffer)
    
    return output_images

if __name__ == '__main__':
    assert len(sys.argv) >= 3
    panorama_image = imageio.imread(sys.argv[1])
    output_dir = sys.argv[2]
    out_images = panorama_to_cube(panorama_image, 512)
    for cube_face in out_images:
        imageio.imsave(
                os.path.join(output_dir, '%s_ref.png'%cube_face),
                out_images[cube_face])
