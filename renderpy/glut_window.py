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
import renderpy
import example_scenes

class GlutWindow:
    def __init__(self, width, height, timer_freq = 0):
        self.width = width
        self.height = height
        
        glutInit([])
        # GLUT_DOUBLE maxes out at 60fps
        #glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH)
        glutInitWindowSize(width, height)
        glutCreateWindow('RENDERPY')
        
        # I think this is only necessary if I'm using the main loop, but I'm not
        #glutSetOption(
        #        GLUT_ACTION_ON_WINDOW_CLOSE,
        #        GLUT_ACTION_CONTINUE_EXECUTION)
        
        #glutHideWindow()
        
        self.renderpy = None
    
    def add_renderpy(self, renderpy):
        self.renderpy = renderpy
    
    def get_color(self):
        self.renderpy.color_render()
        test = glReadPixels(
                0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
        img = numpy.frombuffer(test, dtype=numpy.uint8).reshape(width,height,3)
        return img

if __name__ == '__main__':
    width = 256
    height = 256
    g = GlutWindow(width, height, timer_freq = 100)
    #r = example_scenes.first_test(width, height)
    r = renderpy.Renderpy()
    r.load_scene(example_scenes.second_test())
    g.add_renderpy(r)
    
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
                [math.sin(theta[0]), 0, math.cos(theta[0]), 0],
                [0, 0, 0, 1]])
    
        c = numpy.linalg.inv(numpy.dot(rotate, numpy.dot(elevate, translate)))
        r.move_camera(c)
    
        theta[0] += 0.0001
        
        img = g.get_color()
        
        rendered_frames +=1
        if rendered_frames % 100 == 0:
            print('hz: %.04f'%(rendered_frames / (time.time() - t0)))
