import os

from OpenGL import GL
import glfw

import numpy

import splendor.camera as camera
from splendor.contexts.initialization import (
    initialization_state, register_context)

_glfw_state = {
    'initialized' : False,
    'windows' : [],
}

def initialize(x_authority=None, display=None):
    if not glfw.init():
        raise Exception('GLFW cannot be initialized')
    new_context = register_context('glfw')
    if new_context:
        if x_authority is not None:
            os.environ['XAUTHORITY'] = x_authority
            os.environ['DISPLAY'] = display
        
        _glfw_state['initialized'] = True

def terminate():
    glfw.terminate()

class GLFWWindowWrapper:
    def __init__(self,
        name='SPLENDOR',
        width=128,
        height=128,
        anti_alias=False,
        anti_alias_samples=8,
    ):
        #initialized, mode = initialization_state()
        #assert initialized and mode == 'glfw'
        #assert _glfw_state['initialized']
        
        self.name = name
        self.anti_alias = anti_alias
        self.anti_alias_samples = anti_alias_samples
        
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
        #glfw.window_hint(glfw.GLFW_RED_BITS, 8)
        #glfw.window_hint(glfw.GLFW_GREEN_BITS, 8)
        #glfw.window_hint(glfw.GLFW_BLUE_BITS, 8)
        #glfw.window_hint(glfw.STENCIL_BITS, 8)
        #glfw.window_hint(glfw.DEPTH_BITS, 24)
        self.glfw_window = glfw.create_window(width, height, name, None, None)
        if self.glfw_window is None:
            glfw.terminate()
            raise Exception('GLFW window cannot be created')
        
        self.set_active()
        
    def set_active(self):
        glfw.make_context_current(self.glfw_window)
    
    def show_window(self):
        pass
    
    def should_close(self):
        return glfw.window_should_close(self.glfw_window)
    
    def poll_events(self):
        glfw.poll_events()
    
    def swap_buffers(self):
        glfw.swap_buffers(self.glfw_window)
    
    def enable_window(self):
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        fbw, fbh = glfw.get_framebuffer_size(self.glfw_window)
        GL.glViewport(0, 0, fbw, fbh)
        GL.glScissor(0, 0, fbw, fbh)
        if self.anti_alias:
            GL.glEnable(GL.GL_MULTISAMPLE)
        else:
            GL.glDisable(GL.GL_MULTISAMPLE)
    
    def read_pixels(self, read_depth=False, projection=None):
        self.enable_window()
        fbw, fbh = glfw.get_framebuffer_size(self.glfw_window)
        width = fbw
        height = fbh
        anti_alias = self.anti_alias

        if read_depth:
            if projection is None:
                raise ValueError('Must specify a projection when reading depth')
            near, far = camera.clip_from_projection(projection)
            pixels = GL.glReadPixels(
                0,
                0,
                width,
                height,
                GL.GL_DEPTH_COMPONENT,
                GL.GL_UNSIGNED_SHORT,
            )
            image = numpy.frombuffer(pixels, dtype=numpy.ushort).reshape(
                    height, width, 1)
            image = image.astype(float) / (2**16-1)
            image = 2.0 * image - 1.0
            image = 2.0 * near * far / (far + near - image * (far - near))
        else:
            pixels = GL.glReadPixels(
                    0, 0, width, height, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
            image = numpy.frombuffer(pixels, dtype=numpy.uint8).reshape(
                    height, width, 4)

        GL.glViewport(0, 0, width, height)
        return image
    
    def set_mouse_button_callback(self, fn):
        glfw.set_mouse_button_callback(self.glfw_window, fn)
    
    def set_cursor_pos_callback(self, fn):
        glfw.set_cursor_pos_callback(self.glfw_window, fn)
    
    def set_key_callback(self, fn):
        glfw.set_key_callback(self.glfw_window, fn)
    
    def set_scroll_callback(self, fn):
        glfw.set_scroll_callback(self.glfw_window, fn)
    
    def framebuffer_size(self):
        return glfw.get_framebuffer_size(self.glfw_window)

#if __name__ == '__main__':
#    initialize()
#    window = GLFWWindowWrapper(name='glfw_test', width=128, height=128)
