#!/usr/bin/env python

# system
import random
import math

# opengl
from renderpy.opengl_wrapper import vbo, GL

# numpy
import numpy

# imageio
import imageio

# local
from renderpy import shader_definitions
import renderpy.buffer_manager_glut as buffer_manager
from renderpy import camera

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
    vertex_shader = GL.shaders.compileShader(
            shader_definitions.background_vertex_shader,
            GL.GL_VERTEX_SHADER)
    fragment_shader = GL.shaders.compileShader(
            shader_definitions.background_2D_fragment_shader,
            GL_FRAGMENT_SHADER)
    program = GL.shaders.compileProgram(vertex_shader, fragment_shader)

    GL.glUseProgram(program)

    # get shader variable locations
    projection_location = GL.glGetUniformLocation(program, 'projection_matrix')
    camera_location = GL.glGetUniformLocation(program, 'camera_pose')
    sampler_location = GL.glGetUniformLocation(program, 'texture_sampler')
    GL.glUniform1i(sampler_location, 0)

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
    face_buffer = vbo.VBO(face_ints, target = GL.GL_ELEMENT_ARRAY_BUFFER)

    vertex_buffer.bind()
    face_buffer.bind()

    # textures
    texture_buffer = GL.glGenTextures(1)
    GL.glActiveTexture(GL.GL_TEXTURE0)
    GL.glBindTexture(GL.GL_TEXTURE_2D, texture_buffer)

    glTexImage2D(
            GL.GL_TEXTURE_2D,
            0, GL.GL_RGB, image.shape[1], image.shape[0],
            0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, image)

    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER,
            GL.GL_LINEAR_MIPMAP_LINEAR)
    GL.glGenerateMipmap(GL.GL_TEXTURE_2D)

    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
    GL.glDrawElements(GL.GL_TRIANGLES, 2*3, GL.GL_UNSIGNED_INT, None)
    output_image = manager.read_pixels('resize_image')

    vertex_buffer.unbind()
    face_buffer.unbind()
    GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
    GL.glDeleteTextures(texture_buffer)

    return output_image


if __name__ == '__main__':
    image = numpy.array(imageio.imread(
            '/home/awalsman/Development/renderpy/renderpy/example_textures/'
            'spinner_tex.png'))[:,:,:3]

    out_images = resize_image(image, 128, 128)
    imageio.imsave('./tmp.png', out_images)
