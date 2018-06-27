#!/usr/bin/env python

# system
import time
import math

# opengl
from OpenGL.GL import *
from OpenGL.GLX import *
from OpenGL.GLU import *

# glfw
import glfw

# numpy
import numpy

# local
import example_scenes

class GLFWWindow:
    def __init__(self, width, height, timer_freq = 0):
        glfw.init()
        glfw.window_hint(glfw.RESIZABLE, GL_FALSE)
        glfw.window_hint(glfw.DOUBLEBUFFER, GL_FALSE)
        self.window = glfw.create_window(width, height, 'RENDERPY', None, None)
        glfw.make_context_current(self.window)
        glfw.set_key_callback(self.window, self.key_callback)
        glViewport(0, 0, width, height)
        
        self.renderpy = None
        self.rendered_frames = 0
        
        self.timer_freq = timer_freq
        if self.timer_freq:
            self.start_time = time.time()
    
    def add_renderpy(self, renderpy):
        self.renderpy = renderpy
    
    def display(self):
        self.renderpy.color_render()
        glfw.poll_events()
        #glfw.swap_buffers(self.window)
        glFinish()
        self.rendered_frames += 1
        if self.timer_freq:
            if self.rendered_frames % self.timer_freq == 0:
                print('hz: %.04f'%(
                        self.rendered_frames/(time.time() - self.start_time)))
    
    def key_callback(self, win, key, scancode, action, mods):
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            self.shutdown()
    
    def shutdown(self):
        glfw.set_window_should_close(self.window, True)

if __name__ == '__main__':
    width = 256
    height = 256
    g = GLFWWindow(width, height, timer_freq = 1)
    r = example_scenes.first_test(width, height)
    g.add_renderpy(r)
    
    theta = 0.
    translate = numpy.array([[1,0,0,0],[0,1,0,0],[0,0,1,6],[0,0,0,1]])
    e = math.radians(-20)
    elevate = numpy.array([
            [1, 0, 0, 0],
            [0, math.cos(e), -math.sin(e), 0],
            [0, math.sin(e), math.cos(e), 0],
            [0,0,0,1]])
    
    try:
        while True:
            rotate = numpy.array([
                    [math.cos(theta), 0, -math.sin(theta), 0],
                    [0, 1, 0, 0],
                    [math.sin(theta), 0, math.cos(theta), 0],
                    [0, 0, 0, 1]])
            
            c = numpy.linalg.inv(
                numpy.dot(rotate, numpy.dot(elevate, translate)))
            r.move_camera(c)
            
            theta += 0.0001
            
            g.display()
    
    finally:
        g.shutdown()
