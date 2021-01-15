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
import shaders.shader_definitions as shader_definitions
import buffer_manager
import camera

def resize_image(
        image,
        width,
        height):
    
    # initialize the buffer manager
    manager = buffer_manager.initialize_shared_buffer_manager()
    try:
        manager.add_frame('resize_image', width, height)
    except buffer_manager.FrameExistsError:
        pass
    manager.enable_frame('resize_image')
    
    # compile the shaders
    vertex_shader = shaders.compileShader(
            shader_definitions.background_vertex_shader,
            GL_VERTEX_SHADER)
    fragment_shader = shaders.compileShader(
            shader_definitions.background_2D_fragment_shader,
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
            GL_TEXTURE_2D,
            0, GL_RGB, image.shape[1], image.shape[0],
            0, GL_RGB, GL_UNSIGNED_BYTE, image)
    
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
            GL_LINEAR_MIPMAP_LINEAR)
    glGenerateMipmap(GL_TEXTURE_2D)
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glDrawElements(GL_TRIANGLES, 2*3, GL_UNSIGNED_INT, None)
    output_image = manager.read_pixels('resize_image')
        
    vertex_buffer.unbind()
    face_buffer.unbind()
    glBindTexture(GL_TEXTURE_2D, 0)
    glDeleteTextures(texture_buffer)
    
    return output_image

if __name__ == '__main__':
    image = numpy.array(imageio.imread(
            '/home/awalsman/Development/renderpy/renderpy/example_textures/'
            'spinner_tex.png'))[:,:,:3]
    
    out_images = resize_image(image, 128, 128)
    imageio.imsave('./tmp.png', out_images)

