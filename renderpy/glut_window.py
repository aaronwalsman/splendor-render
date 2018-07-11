#!/usr/bin/env python

# system
import math

# opengl
from OpenGL.GL import *
from OpenGL.GLX import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

# numpy/scipy
import scipy.misc
import numpy

# local
import renderpy.core as core
import renderpy.example_scenes as example_scenes

window_size = 128

class GlutWindow:
    def __init__(self, dimensions, window_name = 'RENDERPY'):
        
        self.dimensions = dimensions
        
        glutInit([])
        # GLUT_DOUBLE maxes out at 60fps
        #glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH)
        glutInitWindowSize(window_size, window_size)
        self.window_id = glutCreateWindow(window_name)
        self.set_active()
        
        # I think this is only necessary if I'm using the main loop, but I'm not
        #glutSetOption(
        #        GLUT_ACTION_ON_WINDOW_CLOSE,
        #        GLUT_ACTION_CONTINUE_EXECUTION)
        
        glutHideWindow(self.window_id)
        
        # generate off-screen framebuffer/renderbuffer objects
        self.frame_buffer_data = {}
        for frame_name in dimensions:
            
            width, height = dimensions[frame_name]
            
            # frame buffer
            frame_buffer = glGenFramebuffers(1)
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, frame_buffer)
            
            # color renderbuffer
            render_buffer = glGenRenderbuffers(1)
            glBindRenderbuffer(GL_RENDERBUFFER, render_buffer)
            glRenderbufferStorage(
                    GL_RENDERBUFFER, GL_RGBA8, width, height)
            glFramebufferRenderbuffer(
                    GL_FRAMEBUFFER,
                    GL_COLOR_ATTACHMENT0,
                    GL_RENDERBUFFER,
                    render_buffer)
            
            # depth renderbuffer
            depth_buffer = glGenRenderbuffers(1)
            glBindRenderbuffer(GL_RENDERBUFFER, depth_buffer)
            glRenderbufferStorage(
                    GL_RENDERBUFFER, GL_DEPTH_COMPONENT16, width, height)
            glFramebufferRenderbuffer(
                    GL_FRAMEBUFFER,
                    GL_DEPTH_ATTACHMENT,
                    GL_RENDERBUFFER,
                    depth_buffer)
            
            self.frame_buffer_data[frame_name] = {
                    'frame_buffer':frame_buffer,
                    'render_buffer':render_buffer,
                    'depth_buffer':depth_buffer}
        
        self.renderer = core.Renderpy()
    
    def set_active(self):
        glutSetWindow(self.window_id)
    
    def get_color(self, frame_name, *args, **kwargs):
        
        width, height = self.dimensions[frame_name]
        
        self.set_active()
        glBindFramebuffer(
                GL_FRAMEBUFFER,
                self.frame_buffer_data[frame_name]['frame_buffer'])
        glViewport(0,0, width, height)
        
        try:
            self.renderer.color_render(*args, **kwargs)
            glReadBuffer(GL_COLOR_ATTACHMENT0)
            pixels = glReadPixels(
                    0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
            image = numpy.frombuffer(pixels, dtype=numpy.uint8).reshape(
                    width, height, 3)
        finally:
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)
        
        return image
    
    def get_mask(self, frame_name, *args, **kwargs):
        
        width, height = self.dimensions[frame_name]
        
        self.set_active()
        glBindFramebuffer(
                GL_FRAMEBUFFER,
                self.frame_buffer_data[frame_name]['frame_buffer'])
        glViewport(0, 0, width, height)
        
        try:
            self.renderer.mask_render(*args, **kwargs)
            glReadBuffer(GL_COLOR_ATTACHMENT0)
            pixels = glReadPixels(
                    0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
            image = numpy.frombuffer(pixels, dtype=numpy.uint8).reshape(
                    width, height, 3)
        finally:
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)
        
        return image

if __name__ == '__main__':
    width = 256
    height = 256
    g = GlutWindow({'main':[width, height]})
    g.renderer.load_scene(example_scenes.second_test())
    
    #g2 = GlutWindow(width, height)
    #g2.renderer.load_scene(example_scenes.second_test())
    #g2.renderer.scene_description['instances']['cube1']['material_name'] = (
    #        'candy_color')
    
    theta = [0.0]
    translate = numpy.array([[1,0,0,0],[0,1,0,0],[0,0,1,6],[0,0,0,1]])
    e = math.radians(-20)
    elevate = numpy.array([
            [1, 0, 0, 0],
            [0, math.cos(e), -math.sin(e), 0],
            [0, math.sin(e), math.cos(e), 0],
            [0, 0, 0, 1]])
    
    import time
    t0 = time.time()
    rendered_frames = 0
    while True:
        rotate = numpy.array([
                [math.cos(theta[0]), 0, -math.sin(theta[0]), 0],
                [0, 1, 0, 0],
                [math.sin(theta[0]), 0,  math.cos(theta[0]), 0],
                [0, 0, 0, 1]])
    
        c = numpy.linalg.inv(
                numpy.dot(rotate, numpy.dot(elevate, translate)))
        g.renderer.move_camera(c)
        #g2.renderer.move_camera(c)
        
        theta[0] += 0.0001
        
        img = g.get_color('main')
        #img2 = g2.get_color()
        
        #scipy.misc.imsave('./test_img%i.png'%rendered_frames, img)
        
        rendered_frames +=1
        if rendered_frames % 100 == 0:
            print('hz: %.04f'%(rendered_frames / (time.time() - t0)))
