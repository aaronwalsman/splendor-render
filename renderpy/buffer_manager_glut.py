# system
import os

# opengl
from OpenGL.GL import *
#from OpenGL.GLX import *
#from OpenGL.GLU import *
from OpenGL.GLUT import *

# numpy
import numpy

default_window_size = 128

shared_buffer_manager = []

os.environ['XAUTHORITY'] = "/home/awalsman/.Xauthority"
os.environ['DISPLAY'] = ':0'

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
            width = default_window_size,
            height = None,
            anti_aliasing = True,
            x_authority = None,
            display = None):
        
        if height is None:
            height = width
        
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
        glutInitWindowSize(width, height)
        self.window_id = glutCreateWindow('RENDERPY')
        self.set_active()
        self.window_width = width
        self.window_height = height
        self.anti_aliasing = anti_aliasing
        self.hide_window()
        
        # storage for off-screen framebuffer/renderbuffer objects
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
    
    '''
    def add_depth_frame(self, frame_name, width, height):
        
        if frame_name in self.framebuffer_data:
            raise FrameExistsError('The frame %s is already in use'%frame_name)
        
        # frame buffer
        frame_buffer = glGenFrameBuffers(1)
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, frame_buffer)
        
        # depth renderbuffer
        render_buffer = glGenRenderbuffers(1)
        glBindFramebuffer(GL_RENDER_BUFFER, render_buffer)
        glRenderbufferStorage(
                GL_RENDERBUFFER, GL_DEPTH_COMPONENT32F, width, height)
        glFramebufferRenderBuffer(
                GL_FRAMEBUFFER,
                GL_DEPTH_ATTACHMENT,
                GL_RENDERBUFFER,
                render_buffer)
        
    '''        
    
    def add_frame(self, frame_name, width, height, anti_aliasing=True):
        
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
                'depthbuffer':depth_buffer,
                'anti_aliasing':anti_aliasing}
        
        if anti_aliasing:
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
        if self.anti_aliasing:
            glEnable(GL_MULTISAMPLE)
        else:
            glDisable(GL_MULTISAMPLE)
    
    def enable_frame(self, frame):
        width = self.framebuffer_data[frame]['width']
        height = self.framebuffer_data[frame]['height']
        anti_aliasing = self.framebuffer_data[frame]['anti_aliasing']
        if anti_aliasing:
            glBindFramebuffer(
                    GL_FRAMEBUFFER,
                    self.framebuffer_data[frame]['framebuffermulti'])
            glEnable(GL_MULTISAMPLE)
        else:
            glBindFramebuffer(
                    GL_FRAMEBUFFER,
                    self.framebuffer_data[frame]['framebuffer'])
            glDisable(GL_MULTISAMPLE)
        glViewport(0, 0, width, height)
    
    def read_pixels(self, frame, read_depth=False, near=0.05, far=50.0):
        if frame is None:
            self.enable_window()
            width = self.window_width
            height = self.window_height
            anti_aliasing = self.anti_aliasing
        else:
            width = self.framebuffer_data[frame]['width']
            height = self.framebuffer_data[frame]['height']
            anti_aliasing = self.framebuffer_data[frame]['anti_aliasing']
            if anti_aliasing:
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
        
        if read_depth:
            pixels = glReadPixels(
                    0, 0, width, height, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT)
            image = numpy.frombuffer(pixels, dtype=numpy.ushort).reshape(
                    height, width, 1)
            image = image.astype(numpy.float) / (2**16-1)
            image = 2.0 * image - 1.0
            image = 2.0 * near * far / (far + near - image * (far - near))
            #image[mask] = -1.
            print(numpy.min(image))
            print(numpy.max(image))
        else:
            pixels = glReadPixels(
                    0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
            image = numpy.frombuffer(pixels, dtype=numpy.uint8).reshape(
                    height, width, 3)
        
        # re-enable the multibuffer for future drawing
        if anti_aliasing and frame is not None:
            glBindFramebuffer(
                    GL_FRAMEBUFFER,
                    self.framebuffer_data[frame]['framebuffermulti'])
            glEnable(GL_MULTISAMPLE)
        glViewport(0, 0, width, height)
        
        return image
    
    def finish(self):
        glFlush()
        glFinish()
        glutPostRedisplay()
        glutSwapBuffers()
        glutLeaveMainLoop()
