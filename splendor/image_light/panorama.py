import math

from OpenGL import GL
from OpenGL.GL import shaders
from OpenGL.arrays import vbo

import numpy

from splendor.shaders.background import background_vertex_shader
from splendor.shaders.skybox import panorama_to_cube_fragment_shader
from splendor.frame_buffer import FrameBufferWrapper
from splendor.contexts import egl
import splendor.camera as camera
import splendor.image as image

def cube_to_strip(cube):
    strip = numpy.concatenate([
            cube['px'],
            cube['nx'],
            cube['py'],
            cube['ny'],
            cube['pz'],
            cube['nz']], axis=1)
    
    return strip

def panorama_to_strip(*args, **kwargs):
    cube = panorama_to_cube(*args, **kwargs)
    strip = cube_to_strip(cube)
    return strip

def panorama_to_cube(
        panorama_image,
        cube_width,
        panorama_filter='linear',
        device=None):

    # initialize egl
    egl.initialize_plugin()
    egl.initialize_device(device)

    frame_buffer = FrameBufferWrapper(cube_width, cube_width)
    frame_buffer.enable()

    # compile the shaders
    vertex_shader = shaders.compileShader(
            background_vertex_shader, GL.GL_VERTEX_SHADER)
    fragment_shader = shaders.compileShader(
            panorama_to_cube_fragment_shader, GL.GL_FRAGMENT_SHADER)
    program = shaders.compileProgram(vertex_shader, fragment_shader)

    GL.glUseProgram(program)

    # get shader variable locations
    projection_location = GL.glGetUniformLocation(program, 'projection_matrix')
    camera_location = GL.glGetUniformLocation(program, 'view_matrix')
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
    GL.glTexImage2D(
            GL.GL_TEXTURE_2D, 0, GL.GL_RGB,
            panorama_image.shape[1], panorama_image.shape[0], 0,
            GL.GL_RGB, GL.GL_UNSIGNED_BYTE, panorama_image)

    if panorama_filter == 'linear':
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D,
            GL.GL_TEXTURE_MAG_FILTER,
            GL.GL_LINEAR,
        )
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER,
                GL.GL_LINEAR_MIPMAP_LINEAR)
        GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
    elif panorama_filter == 'nearest':
        GL.glTexParameteri(
                GL.GL_TEXTURE_2D,
                GL.GL_TEXTURE_MAG_FILTER,
                GL.GL_NEAREST,
        )
        GL.glTexParameteri(
                GL.GL_TEXTURE_2D,
                GL.GL_TEXTURE_MIN_FILTER,
                GL.GL_NEAREST,
        )
    else:
        raise ValueError('Unknown filter: %s'%panorama_filter)

    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
    GL.glTexParameteri(
            GL.GL_TEXTURE_2D,
            GL.GL_TEXTURE_WRAP_T,
            GL.GL_MIRRORED_REPEAT,
    )

    # cameras
    projection_matrix = camera.projection_matrix(
            math.radians(90), 1.0, 0.01, 1.0)

    view_matrices = {
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

    GL.glUniformMatrix4fv(
            projection_location,
            1,
            GL.GL_TRUE,
            projection_matrix,
    )

    output_images = {}

    for cube_face in view_matrices:
        view_matrix = view_matrices[cube_face]
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glUniformMatrix4fv(camera_location, 1, GL.GL_TRUE, view_matrix)
        GL.glDrawElements(GL.GL_TRIANGLES, 2*3, GL.GL_UNSIGNED_INT, None)
        output_images[cube_face] = frame_buffer.read_pixels()

    vertex_buffer.unbind()
    face_buffer.unbind()
    GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
    GL.glDeleteTextures(texture_buffer)

    return output_images
