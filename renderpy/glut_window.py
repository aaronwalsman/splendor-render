#!/usr/bin/env python

# system
import time
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
        
        glutDisplayFunc(self.display)
        glutIdleFunc(self.idle)
        glutReshapeFunc(self.reshape)
        
        self.renderpy = None
        self.rendered_frames = 0
        
        self.timer_freq = timer_freq
        if self.timer_freq:
            self.start_time = time.time()
    
    def add_renderpy(self, renderpy):
        self.renderpy = renderpy
    
    def set_prerender(self, prerender):
        self.prerender = prerender
    
    def display(self):
        self.prerender()
        self.renderpy.color_render()
        #glutSwapBuffers()
        glFinish()
        self.rendered_frames += 1
        if self.timer_freq:
            if self.rendered_frames % self.timer_freq == 0:
                print('hz: %.04f'%(
                        self.rendered_frames/(time.time() - self.start_time)))
    
    def idle(self):
        self.display()

    def reshape(self, width, height):
        glViewport(0,0,width,height)
    
    def getImage(self):
        self.display()
        test = glReadPixels(
                0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
        img = numpy.frombuffer(test, dtype=numpy.uint8).reshape(width,height,3)
        return img
    
    def run(self):
        glutSetOption(
                GLUT_ACTION_ON_WINDOW_CLOSE,
                GLUT_ACTION_CONTINUE_EXECUTION)
        
        #glutMainLoop()
        while True:
            # no glut main loop, just display
            #glutMainLoopEvent()
            self.getImage()
        
if __name__ == '__main__':
    width = 256
    height = 256
    g = GlutWindow(width, height, timer_freq = 100)
    r = example_scenes.first_test(width, height)
    g.add_renderpy(r)
    
    theta = [0.0]
    translate = numpy.array([[1,0,0,0],[0,1,0,0],[0,0,1,6],[0,0,0,1]])
    e = math.radians(-20)
    elevate = numpy.array([
            [1, 0, 0, 0],
            [0, math.cos(e), -math.sin(e), 0],
            [0, math.sin(e), math.cos(e), 0],
            [0,0,0,1]])
    
    def spin():
        rotate = numpy.array([
                [math.cos(theta[0]), 0, -math.sin(theta[0]), 0],
                [0, 1, 0, 0],
                [math.sin(theta[0]), 0, math.cos(theta[0]), 0],
                [0, 0, 0, 1]])

        c = numpy.linalg.inv(numpy.dot(rotate, numpy.dot(elevate, translate)))
        r.move_camera(c)

        theta[0] += 0.0001
    
    g.set_prerender(spin)
    
    g.run()
