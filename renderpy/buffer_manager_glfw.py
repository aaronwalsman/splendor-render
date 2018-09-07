#!/usr/bin/env python

# system
import math

# opengl
from OpenGL.GL import *
from OpenGL.GLX import *
from OpenGL.GLU import *

# glfw
import glfw

# numpy/scipy
import scipy.misc
import numpy

# local
import renderpy.core as core
import renderpy.example_scenes as example_scenes

default_window_size = 128

shared_buffer_manager = []

def initialize_shared_buffer_manager(*args, **kwargs):
    if len(shared_buffer_manager):
        return shared_buffer_manager[0]
    else:
        shared_buffer_manager.append(BufferManager(*args, **kwargs))
        return shared_buffer_manager[0]

class BufferManager:
    def __init__(self, window_size = default_window_size):
        
        if not glfw.init():
            return
        
        CONTINUE HERE?
        
        glutInit([])
        # GLUT_DOUBLE maxes out at 60fps
        #glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        #glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_MULTISAMPLE)
        glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH)
        glutInitWindowSize(window_size, window_size)
        self.window_id = glutCreateWindow('RENDERPY')
        self.set_active()
        self.window_width = window_size
        self.window_height = window_size
        # does not work b/c glut
        #self.resize_window(window_size, window_size)
        self.hide_window()
        
        # I think this is only necessary if I'm using the main loop, but I'm not
        #glutSetOption(
        #        GLUT_ACTION_ON_WINDOW_CLOSE,
        #        GLUT_ACTION_CONTINUE_EXECUTION)
        
        # generate off-screen framebuffer/renderbuffer objects
        self.framebuffer_data = {}
    
    def hide_window(self):
        glutHideWindow(self.window_id)
    
    def show_window(self):
        glutShowWindow(self.window_id)
    
    def resize_window(self, width, height):
        if self.window_width != width or self.window_height != height:
            self.window_width = width
            self.window_height = height
            glutReshapeWindow(width, height)
    
    def add_frame(self, frame_name, width, height):
        
        if frame_name in self.framebuffer_data:
            raise Exception('The frame %s is already in use'%frame_name)
        
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
        
        self.framebuffer_data[frame_name] = {
                'width':width,
                'height':height,
                'framebuffer':frame_buffer,
                'renderbuffer':render_buffer,
                'depthbuffer':depth_buffer}
    
    def set_active(self):
        '''
        This is only necessary if you have multiple buffer managers, which
        you shouldn't do, because you will end up confusing yourself.
        '''
        glutSetWindow(self.window_id)
    
    def enable_window(self):
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glViewport(0, 0, self.window_width, self.window_height)
    
    def enable_frame(self, frame):
        width = self.framebuffer_data[frame]['width']
        height = self.framebuffer_data[frame]['height']
        glBindFramebuffer(
                GL_FRAMEBUFFER,
                self.framebuffer_data[frame]['framebuffer'])
        glViewport(0, 0, width, height)
    
    def read_pixels(self, frame):
        if frame is None:
            self.enable_window()
            width = self.window_width
            height = self.window_height
        else:
            self.enable_frame(frame)
            width = self.framebuffer_data[frame]['width']
            height = self.framebuffer_data[frame]['height']
        
        pixels = glReadPixels(
                0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
        image = numpy.frombuffer(pixels, dtype=numpy.uint8).reshape(
                width, height, 3)
        
        return image
    
    
    def finish(self):
        glFlush()
        glFinish()
        glutPostRedisplay()
        glutSwapBuffers()
        glutLeaveMainLoop()
    

if __name__ == '__main__':
    width = 512
    height = 512
    
    buffer_manager = initialize_shared_buffer_manager(width)
    buffer_manager.add_frame('A', width, height)
    #buffer_manager.add_frame('B', width*2, height*2)
    
    #glutSetOption(GLUT_MULTISAMPLE, 4)
    
    
    rendererA = core.Renderpy()
    rendererA.load_scene(example_scenes.fourth_test())
    
    #rendererB = core.Renderpy()
    #rendererB.load_scene(example_scenes.second_test())
    #rendererB.set_instance_material('cube1', 'candy_color')
    
    theta = [0.0]
    translate = numpy.array([[1,0,0,0],[0,1,0,0],[0,0,1,6],[0,0,0,1]])
    e = math.radians(-40)
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
        rendererA.set_camera_pose(c)
        #rendererB.set_camera_pose(c)
        
        theta[0] += 0.0001
        
        #buffer_manager.enable_frame('A')
        buffer_manager.show_window()
        buffer_manager.enable_window()
        rendererA.color_render(flip_y=False)
        #imgA = buffer_manager.read_pixels('A')
        
        #buffer_manager.enable_frame('B')
        #rendererB.color_render()
        #imgB = buffer_manager.read_pixels('B')
        
        #scipy.misc.imsave('./test_img_A_%i.png'%rendered_frames, imgA)
        #scipy.misc.imsave('./test_img_B_%i.png'%rendered_frames, imgB)
        
        rendered_frames +=1
        if rendered_frames % 100 == 0:
            print('hz: %.04f'%(rendered_frames / (time.time() - t0)))
