import os

from renderpy.opengl_wrapper import (
    glFlush,
    glFinish,
    glBindFramebuffer,
    GL_FRAMEBUFFER,
    glViewport,
    glEnable,
    glDisable,
    GL_MULTISAMPLE,
    glReadPixels,
    GLUT,
)

import numpy

from renderpy.opengl_wrapper import GL, GLUT
import renderpy.camera as camera

glut_state = {
    'initialized' : False
}

def initialize_glut(display=None, x_authority=None):
    if glut_state['initialized'] == False:
        if x_authority is not None:
            os.environ['XAUTHORITY'] = x_authority
            os.environ['DISPLAY'] = display

        GLUT.glutInit([])
        glut_state['initialized'] = True

def finish():
    GL.glFlush()
    GL.glFinish()
    GLUT.glutPostRedisplay()
    GLUT.glutSwapBuffers()
    GLUT.glutLeaveMainLoop()
    glut_state['initializer'] = None

class GlutWindowWrapper:
    def __init__(self,
            name = 'RENDERPY',
            width = 128,
            height = 128,
            anti_alias = True,
            anti_alias_samples = 8):

        self.name = name
        self.width = width
        self.height = height
        self.anti_alias = anti_alias
        self.anti_alias_samples = anti_alias_samples

        GLUT.glutInit([])
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
                    0, 0, width, height, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT)
            image = numpy.frombuffer(pixels, dtype=numpy.ushort).reshape(
                    height, width, 1)
            image = image.astype(numpy.float) / (2**16-1)
            image = 2.0 * image - 1.0
            image = 2.0 * near * far / (far + near - image * (far - near))
        else:
            pixels = GL.glReadPixels(
                    0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
            image = numpy.frombuffer(pixels, dtype=numpy.uint8).reshape(
                    height, width, 3)

        GL.glViewport(0, 0, width, height)
        return image

    def start_main_loop(self, **callbacks):
        self.set_active()
        for callback_name, callback_function in callbacks.items():
            getattr(GLUT, callback_name)(callback_function)
        GLUT.glutMainLoop()
