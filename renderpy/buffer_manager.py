# system
import os
import math
import os

# opengl
from OpenGL.GL import *
from OpenGL.GLX import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

# numpy/scipy
#import scipy.misc
import PIL.Image as Image
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

class FrameExistsError(Exception):
    pass

class BufferManager:
    def __init__(self,
            window_size = default_window_size,
            anti_aliasing = True,
            x_authority = None,
            display = None):
        
        if x_authority is not None:
            os.environ['XAUTHORITY'] = x_authority
            os.environ['DISPLAY'] = display
        
        glutInit([])
        # GLUT_DOUBLE maxes out at 60fps
        #glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        if anti_aliasing:
            glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_MULTISAMPLE)
            glEnable(GL_MULTISAMPLE)
        else:
            glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH)
        glutInitWindowSize(window_size, window_size)
        self.window_id = glutCreateWindow('RENDERPY')
        self.set_active()
        self.window_width = window_size
        self.window_height = window_size
        self.anti_aliasing = anti_aliasing
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
            raise FrameExistsError('The frame %s is already in use'%frame_name)
        
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
        
        if self.anti_aliasing:
            # multi-sample frame buffer
            frame_buffer_multi = glGenFramebuffers(1)
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, frame_buffer_multi)
            
            # color multi-sample renderbuffer
            render_buffer_multi = glGenRenderbuffers(1)
            glBindRenderbuffer(GL_RENDERBUFFER, render_buffer_multi)
            glRenderbufferStorageMultisample(
                    GL_RENDERBUFFER, 8, GL_RGBA8, width, height)
            glFramebufferRenderbuffer(
                    GL_FRAMEBUFFER,
                    GL_COLOR_ATTACHMENT0,
                    GL_RENDERBUFFER,
                    render_buffer_multi)
            
            # depth multi-sample renderbuffer
            depth_buffer_multi = glGenRenderbuffers(1)
            glBindRenderbuffer(GL_RENDERBUFFER, depth_buffer_multi)
            glRenderbufferStorageMultisample(
                    GL_RENDERBUFFER, 8, GL_DEPTH_COMPONENT16, width, height)
            glFramebufferRenderbuffer(
                    GL_FRAMEBUFFER,
                    GL_DEPTH_ATTACHMENT,
                    GL_RENDERBUFFER,
                    depth_buffer_multi)
            
            self.framebuffer_data[frame_name]['framebuffermulti'] = (
                    frame_buffer_multi)
            self.framebuffer_data[frame_name]['depthbuffermulti'] = (
                    depth_buffer_multi)
            self.framebuffer_data[frame_name]['renderbuffermulti'] = (
                    render_buffer_multi)
    
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
        if self.anti_aliasing:
            glBindFramebuffer(
                    GL_FRAMEBUFFER,
                    self.framebuffer_data[frame]['framebuffermulti'])
        else:
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
            width = self.framebuffer_data[frame]['width']
            height = self.framebuffer_data[frame]['height']
            if self.anti_aliasing:
                glBindFramebuffer(
                        GL_READ_FRAMEBUFFER,
                        self.framebuffer_data[frame]['framebuffermulti'])
                glBindFramebuffer(
                        GL_DRAW_FRAMEBUFFER,
                        self.framebuffer_data[frame]['framebuffer'])
                glBlitFramebuffer(
                        0, 0, width, height,
                        0, 0, width, height,
                        GL_COLOR_BUFFER_BIT, GL_NEAREST)
                glBindFramebuffer(
                        GL_FRAMEBUFFER,
                        self.framebuffer_data[frame]['framebuffer'])
            else:
                self.enable_frame(frame)
        
        pixels = glReadPixels(
                0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
        image = numpy.frombuffer(pixels, dtype=numpy.uint8).reshape(
                height, width, 3)
        
        # re-enable the multibuffer for future drawing
        if self.anti_aliasing:
            glBindFramebuffer(
                    GL_FRAMEBUFFER,
                    self.framebuffer_data[frame]['framebuffermulti'])
        
        return image
    
    
    def finish(self):
        glFlush()
        glFinish()
        glutPostRedisplay()
        glutSwapBuffers()
        glutLeaveMainLoop()
