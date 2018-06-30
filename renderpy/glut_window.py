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

class GlutWindow:
    def __init__(self, width, height):
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
        
        self.renderer = core.Renderpy(width, height)
    
    def get_color(self):
        self.renderer.color_render()
        pixels = glReadPixels(
                0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
        image = numpy.frombuffer(pixels, dtype=numpy.uint8).reshape(
                self.width, self.height, 3)
        return image
    
    def get_mask(self):
        self.renderer.mask_render()
        pixels = glReadPixels(
                0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
        image = numpy.frombuffer(pixels, dtype=numpy.uint8).reshape(
                self.width, self.height, 3)
        return image

if __name__ == '__main__':
    width = 256
    height = 256
    g = GlutWindow(width, height)
    #r = example_scenes.first_test(width, height)
    #r = renderer.Renderpy()
    #r.load_scene(example_scenes.second_test())
    #g.add_renderer(r)
    g.renderer.load_scene(example_scenes.second_test())
    
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
    
        c = numpy.linalg.inv(
                numpy.dot(rotate, numpy.dot(elevate, translate)))
        g.renderer.move_camera(c)
    
        theta[0] += 0.0001
        
        img = g.get_color()
        
        rendered_frames +=1
        if rendered_frames % 100 == 0:
            print('hz: %.04f'%(rendered_frames / (time.time() - t0)))
