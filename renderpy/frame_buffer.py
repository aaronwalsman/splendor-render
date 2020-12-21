import numpy

from OpenGL.GL import *

class FrameBufferWrapper:
    def __init__(self, width, height, anti_alias=True):
        
        self.width = width
        self.height = height
        self.anti_alias = anti_alias
        
        # frame buffer
        self.frame_buffer = glGenFramebuffers(1)
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self.frame_buffer)
        
        # color renderbuffer
        self.render_buffer = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, self.render_buffer)
        glRenderbufferStorage(
                GL_RENDERBUFFER, GL_RGBA8, width, height)
        glFramebufferRenderbuffer(
                GL_FRAMEBUFFER,
                GL_COLOR_ATTACHMENT0,
                GL_RENDERBUFFER,
                self.render_buffer)
        
        # depth renderbuffer
        self.depth_buffer = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, self.depth_buffer)
        glRenderbufferStorage(
                GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, self.width, self.height)
        glFramebufferRenderbuffer(
                GL_FRAMEBUFFER,
                GL_DEPTH_ATTACHMENT,
                GL_RENDERBUFFER,
                self.depth_buffer)
        
        if self.anti_alias:
            # multi-sample frame buffer
            self.frame_buffer_multi = glGenFramebuffers(1)
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self.frame_buffer_multi)
            
            # color multi-sample renderbuffer
            self.render_buffer_multi = glGenRenderbuffers(1)
            glBindRenderbuffer(GL_RENDERBUFFER, self.render_buffer_multi)
            glRenderbufferStorageMultisample(
                    GL_RENDERBUFFER, 8, GL_RGBA8, self.width, self.height)
            glFramebufferRenderbuffer(
                    GL_FRAMEBUFFER,
                    GL_COLOR_ATTACHMENT0,
                    GL_RENDERBUFFER,
                    self.render_buffer_multi)
            
            # depth multi-sample renderbuffer
            self.depth_buffer_multi = glGenRenderbuffers(1)
            glBindRenderbuffer(GL_RENDERBUFFER, self.depth_buffer_multi)
            glRenderbufferStorageMultisample(
                    GL_RENDERBUFFER, 8, GL_DEPTH_COMPONENT16, width, height)
            glFramebufferRenderbuffer(
                    GL_FRAMEBUFFER,
                    GL_DEPTH_ATTACHMENT,
                    GL_RENDERBUFFER,
                    self.depth_buffer_multi)
    
    def enable(self):
        if self.anti_alias:
            glBindFramebuffer(GL_FRAMEBUFFER, self.frame_buffer_multi)
            glEnable(GL_MULTISAMPLE)
        else:
            glBindFramebuffer(GL_FRAMEBUFFER, self.frame_buffer)
            glDisable(GL_MULTISAMPLE)
        glViewport(0, 0, self.width, self.height)
    
    def read_pixels(self,
            read_depth = False,
            near = 0.05,
            far = 50.0):
        
        if self.anti_alias:
            glBindFramebuffer(GL_READ_FRAMEBUFFER, self.frame_buffer_multi)
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self.frame_buffer)
            glBlitFramebuffer(
                    0, 0, self.width, self.height,
                    0, 0, self.width, self.height,
                    GL_COLOR_BUFFER_BIT, GL_NEAREST)
            glBindFramebuffer(GL_FRAMEBUFFER, self.frame_buffer)
        else:
            self.enable()
        
        if read_depth:
            pixels = glReadPixels(
                    0, 0, self.width, self.height,
                    GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT)
            image = numpy.frombuffer(pixels, dtype=numpy.ushort).reshape(
                    self.height, self.width, 1)
            image = image.astype(numpy.float) / (2**16-1)
            image = 2.0 * image - 1.0
            image = 2.0 * near * far / (far + near - image * (far - near))
        else:
            pixels = glReadPixels(
                    0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
            image = numpy.frombuffer(pixels, dtype=numpy.uint8).reshape(
                    self.height, self.width, 3)
        
        # re-enable the multibuffer for future drawing
        if self.anti_alias:
            glBindFramebuffer(
                    GL_FRAMEBUFFER,
                    self.frame_buffer_multi)
            glEnable(GL_MULTISAMPLE)
        glViewport(0, 0, self.width, self.height)
        
        return image

