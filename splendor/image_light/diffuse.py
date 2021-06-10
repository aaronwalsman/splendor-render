import random
import math

# opengl
from OpenGL import GL
from OpenGL.GL import shaders
from OpenGL.arrays import vbo

# numpy
import numpy

# splendor
from splendor.shaders import image_light, background
from splendor.shader_library import ShaderLibrary
from splendor.contexts import egl
from splendor import camera
from splendor.frame_buffer import FrameBufferWrapper
from splendor.image import save_image

MAX_SAMPLES_PER_STEP = 512

def reflect_to_diffuse(
        diffuse_width,
        reflect_strip,
        intensity_strip,
        num_samples=512,
        debug=False,
        device=None):
    
    # initialize egl and frame_buffer
    egl.initialize_plugin()
    new = egl.initialize_device(device)
    frame_buffer = FrameBufferWrapper(
            diffuse_width, diffuse_width, color_format=GL.GL_RGBA32F)
    frame_buffer.enable()
    
    # compile the shaders
    shader_samples = min(num_samples, MAX_SAMPLES_PER_STEP)
    shader_library = ShaderLibrary({'reflect_to_diffuse':(
            background.background_vertex_shader,
            image_light.reflect_to_diffuse_fragment_shader(shader_samples))})
    shader_library.use_program('reflect_to_diffuse')
    locations = shader_library.get_shader_locations('reflect_to_diffuse')
    
    # process the inputs
    if len(intensity_strip.shape) == 2:
        intensity_strip = numpy.expand_dims(intensity_strip, -1)
    
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
    
    # create the samples
    importance_strip = intensity_strip
    reflect_width, strip_width = importance_strip.shape[:2]
    assert strip_width == reflect_width * 6
    pixel_ids = numpy.arange(reflect_width * strip_width)
    importance = importance_strip.reshape(-1)
    importance = importance / numpy.sum(importance)
    samples = numpy.random.choice(pixel_ids, num_samples, p=importance)
    y, strip_x = numpy.unravel_index(
            samples, (reflect_width, strip_width))
    
    if debug:
        debug_strip = numpy.repeat(intensity_strip, 3, 2).astype(numpy.uint8)
        debug_strip[y,strip_x] = (255,0,0)
        save_image(debug_strip, debug)
    
    x = strip_x % reflect_width
    face = strip_x // reflect_width
    
    view_matrices = numpy.array([
            # px
            [[ 0, 0,-1, 0],
             [ 0,-1, 0, 0],
             [-1, 0, 0, 0],
             [ 0, 0, 0, 1]],
            # nx
            [[ 0, 0, 1, 0],
             [ 0,-1, 0, 0],
             [ 1, 0, 0, 0],
             [ 0, 0, 0, 1]],
            # py
            [[ 1, 0, 0, 0],
             [ 0, 0, 1, 0],
             [ 0,-1, 0, 0],
             [ 0, 0, 0, 1]],
            # ny
            [[ 1, 0, 0, 0],
             [ 0, 0,-1, 0],
             [ 0, 1, 0, 0],
             [ 0, 0, 0, 1]],
            # pz
            [[ 1, 0, 0, 0],
             [ 0,-1, 0, 0],
             [ 0, 0,-1, 0],
             [ 0, 0, 0, 1]],
            # nz
            [[-1, 0, 0, 0],
             [ 0,-1, 0, 0],
             [ 0, 0, 1, 0],
             [ 0, 0, 0, 1]]])
    
    sample_poses = view_matrices[face]
    half_reflect_width = reflect_width / 2
    x = x - half_reflect_width
    y = y - half_reflect_width
    d = (x**2 + y**2 + half_reflect_width**2)**0.5
    sample_direction_importance = numpy.zeros((num_samples, 4, 1))
    sample_direction_importance[:,0,0] = x / d
    sample_direction_importance[:,1,0] = y / d
    sample_direction_importance[:,2,0] = -half_reflect_width / d
    sample_direction_importance = numpy.matmul(
            sample_poses, sample_direction_importance)
    
    # importance ratio
    sample_importance_ratio = 1. / (importance[samples] * importance.size)
    sample_direction_importance[:,3,0] = sample_importance_ratio
    
    # textures
    GL.glUniform1i(locations['reflect_sampler'], 0)
    reflect_intensity_strip = numpy.concatenate(
            (reflect_strip, intensity_strip), axis=-1)
    texture_buffer = GL.glGenTextures(1)
    GL.glActiveTexture(GL.GL_TEXTURE0)
    GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, texture_buffer)
    for i in range(6):
        strip_start = i * reflect_width
        strip_end = (i+1) * reflect_width
        cube_face = reflect_intensity_strip[:,strip_start:strip_end]
        GL.glTexImage2D(
                GL.GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                0, GL.GL_RGBA, cube_face.shape[1], cube_face.shape[0],
                0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, cube_face)
    
    GL.glTexParameteri(
            GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(
            GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(
            GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
    GL.glTexParameteri(
            GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
    GL.glTexParameteri(
            GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_WRAP_R, GL.GL_CLAMP_TO_EDGE)
    
    # cameras
    projection = camera.projection_matrix(math.radians(90.), 1.0, 0.01, 1.0)
    GL.glUniformMatrix4fv(
            locations['projection_matrix'], 1, GL.GL_TRUE, projection)
    
    # render
    diffuse_strip = numpy.zeros((diffuse_width, diffuse_width*6, 3))
    num_steps = math.ceil(num_samples / shader_samples)
    for i in range(num_steps):
        step_direction_importance = sample_direction_importance[
                i*shader_samples:(i+1)*shader_samples]
        step_samples = step_direction_importance.shape[0]
        if step_samples != shader_samples:
            step_direction_importance = numpy.concatenate((
                    step_direction_importance,
                    numpy.zeros((shader_samples - step_samples, 4, 1))), axis=0)
        GL.glUniform4fv(
                locations['sample_direction_importance_ratio'],
                num_samples,
                step_direction_importance[...,0])
    
        for i, view_matrix in enumerate(view_matrices):
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            GL.glUniformMatrix4fv(
                    locations['view_matrix'], 1, GL.GL_TRUE, view_matrix)
            GL.glDrawElements(GL.GL_TRIANGLES, 2*3, GL.GL_UNSIGNED_INT, None)
            strip_start = i * diffuse_width
            strip_end = (i+1) * diffuse_width
            diffuse_strip[:,strip_start:strip_end] += frame_buffer.read_pixels()
        
    diffuse_max = numpy.max(diffuse_strip)
    diffuse_strip /= diffuse_max
    diffuse_strip *= 255
    diffuse_strip = diffuse_strip.astype(numpy.uint8)
    
    return diffuse_strip
