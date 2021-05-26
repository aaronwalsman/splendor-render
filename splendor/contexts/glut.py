'''
Manages and creates a glut window/context.
Rendering with GLUT requires an active window (currently only one is supported)
but also allows for offscreen rendering using a FrameBufferWrapper.

Use this module if you want to use splendor-render to draw to an active window
and one or more offscreen frame buffers.  On the other hand, if all you want is
offscreen rendering, see egl.py instead.

This module requires a physical display to be connected to the rendering
machine, if no display is present, use egl.py instead.
'''
import os

from OpenGL import GL
import OpenGL.GLUT as GLUT

import numpy

import splendor.camera as camera
from splendor.contexts.initialization import (
        initialization_state, register_context)

_glut_state = {
    'initialized' : False,
    'window' : None,
}

def initialize(x_authority = None, display=None, repeat_keys=False):
    '''
    Set glut to be the active context manager for this rendering session.
    Once initialized, other context managers (egl) may not be used.
    
    x_authority : may be used to set the x_authority for remote rendering on
        machines with physical displays attached.
    display : may be used to set the display for remote rendering on machines
        with phyiscal displays attached.
    '''
    new_context = register_context('glut')
    if new_context:
        if x_authority is not None:
            os.environ['XAUTHORITY'] = x_authority
            os.environ['DISPLAY'] = display
        
        GLUT.glutInit([])
        #GLUT.glutInitContextVersion(4,6)
        _glut_state['initialized'] = True
        
        if not repeat_keys:
            GLUT.glutSetKeyRepeat(GLUT.GLUT_KEY_REPEAT_OFF)

class GlutWindowWrapper:
    '''
    Wraps a single GLUT window.
    '''
    def __init__(self,
            name = 'RENDERPY',
            width = 128,
            height = 128,
            anti_alias = True,
            anti_alias_samples = 8):
        
        initialized, mode = initialization_state()
        assert initialized and mode == 'glut'
        
        # multiple windows not supported
        assert _glut_state['initialized'] and _glut_state['window'] is None
        
        self.name = name
        self.width = width
        self.height = height
        self.anti_alias = anti_alias
        self.anti_alias_samples = anti_alias_samples
        
        if self.anti_alias:
            GLUT.glutInitDisplayMode(
                    GLUT.GLUT_RGBA |
                    GLUT.GLUT_DEPTH |
                    GLUT.GLUT_MULTISAMPLE)
            GLUT.glutSetOption(GLUT.GLUT_MULTISAMPLE, self.anti_alias_samples)
        else:
            GLUT.glutInitDisplayMode(GLUT.GLUT_RGBA | GLUT.GLUT_DEPTH)
        GLUT.glutInitWindowSize(self.width, self.height)
        self.window_id = GLUT.glutCreateWindow(name)
        self.set_active()
        
        _glut_state['window'] = self.window_id

    def hide_window(self):
        GLUT.glutHideWindow(self.window_id)

    def show_window(self):
        GLUT.glutShowWindow(self.window_id)

    def resize_window(self, width, height):
        if self.width != width or self.height != height:
            self.width = width
            self.height = height
            GLUT.glutReshapeWindow(width, height)

    def set_active(self):
        '''
        Sets this window active
        '''
        GLUT.glutSetWindow(self.window_id)

    def enable_window(self):
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        GL.glViewport(0, 0, self.width, self.height)
        if self.anti_alias:
            GL.glEnable(GL.GL_MULTISAMPLE)
        else:
            GL.glDisable(GL.GL_MULTISAMPLE)

    def read_pixels(self, read_depth = False, projection=None):
        self.enable_window()
        width = self.width
        height = self.height
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
            image = image.astype(numpy.float) / (2**16-1)
            image = 2.0 * image - 1.0
            image = 2.0 * near * far / (far + near - image * (far - near))
        else:
            pixels = GL.glReadPixels(
                    0, 0, width, height, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
            image = numpy.frombuffer(pixels, dtype=numpy.uint8).reshape(
                    height, width, 3)

        GL.glViewport(0, 0, width, height)
        return image
    
    def register_callbacks(self, **callbacks):
        self.set_active()
        for callback_name, callback_function in callbacks.items():
            getattr(GLUT, callback_name)(callback_function)

def start_main_loop():
    GLUT.glutMainLoop()
