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
    def __init__(self, width, height, renderer=None, window_name = 'RENDERPY'):
        self.width = width
        self.height = height
        
        glutInit([])
        # GLUT_DOUBLE maxes out at 60fps
        #glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH)
        glutInitWindowSize(width, height)
        self.window_id = glutCreateWindow(window_name)
        
        # I think this is only necessary if I'm using the main loop, but I'm not
        #glutSetOption(
        #        GLUT_ACTION_ON_WINDOW_CLOSE,
        #        GLUT_ACTION_CONTINUE_EXECUTION)
        
        #glutHideWindow()
        
        if renderer is None:
            self.renderer = core.Renderpy()
        else:
            self.renderer = renderer
    
    def get_color(self, *args, **kwargs):
        glutSetWindow(self.window_id)
        self.renderer.color_render(*args, **kwargs)
        pixels = glReadPixels(
                0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
        image = numpy.frombuffer(pixels, dtype=numpy.uint8).reshape(
                self.width, self.height, 3)
        return image
    
    def get_mask(self, *args, **kwargs):
        glutSetWindow(self.window_id)
        self.renderer.mask_render(*args, **kwargs)
        pixels = glReadPixels(
                0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
        image = numpy.frombuffer(pixels, dtype=numpy.uint8).reshape(
                self.width, self.height, 3)
        return image

if __name__ == '__main__':
    width = 256
    height = 256
    g = GlutWindow(width, height)
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
        
        img = g.get_color()
        #img2 = g2.get_color()
        
        rendered_frames +=1
        if rendered_frames % 100 == 0:
            print('hz: %.04f'%(rendered_frames / (time.time() - t0)))
