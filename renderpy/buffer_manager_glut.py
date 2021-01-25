# system
import os

# opengl
from OpenGL import GL
import OpenGL.GLUT as GLUT

# numpy
import numpy

# renderpy
import renderpy.camera as camera

default_window_size = 128

glut_state = {
    'buffer_manager' : None
}

def initialize_shared_buffer_manager(*args, **kwargs):
    if glut_state['buffer_manager'] is None:
        glut_state['buffer_manager'] = BufferManagerGLUT(*args, **kwargs)
    return glut_state['buffer_manager']

class BufferManagerGLUT:
    def __init__(self,
            width = default_window_size,
            height = default_window_size,
            anti_alias = True,
            anti_alias_samples = 8,
            hide_window = False,
            x_authority = None,
            display = None):

        self.width = width
        self.height = height
        self.anti_alias = anti_alias
        self.anti_alias_samples = anti_alias_samples

        if x_authority is not None:
            os.environ['XAUTHORITY'] = x_authority
            os.environ['DISPLAY'] = display

        GLUT.glutInit([])
        if self.anti_alias:
            GLUT.glutInitDisplayMode(
                    GLUT.GLUT_RGBA |
                    GLUT.GLUT_DEPTH |
                    GLUT.GLUT_MULTISAMPLE)
            GL.glEnable(GL.GL_MULTISAMPLE)
            GLUT.glutSetOption(GLUT.GLUT_MULTISAMPLE, self.anti_alias_samples)
        else:
            GLUT.glutInitDisplayMode(GLUT.GLUT_RGBA | GLUT.GLUT_DEPTH)
        GLUT.glutInitWindowSize(self.width, self.height)
        self.window_id = GLUT.glutCreateWindow('RENDERPY')
        self.set_active()

        if hide_window:
            self.hide_window()

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

    def read_pixels(self,
            read_depth = False,
            projection = None):

        #if frame is None:
        self.enable_window()
        width = self.width
        height = self.height
        anti_alias = self.anti_alias
        '''
        else:
            width = self.framebuffer_data[frame]['width']
            height = self.framebuffer_data[frame]['height']
            anti_alias = self.framebuffer_data[frame]['anti_alias']
            if anti_alias:
                GL.glBindFramebuffer(
                        GL.GL_READ_FRAMEBUFFER,
                        self.framebuffer_data[frame]['framebuffermulti'])
                GL.glBindFramebuffer(
                        GL.GL_DRAW_FRAMEBUFFER,
                        self.framebuffer_data[frame]['framebuffer'])
                GL.glBlitFramebuffer(
                        0, 0, width, height,
                        0, 0, width, height,
                        GL.GL_COLOR_BUFFER_BIT, GL.GL_NEAREST)
                GL.glBindFramebuffer(
                        GL.GL_FRAMEBUFFER,
                        self.framebuffer_data[frame]['framebuffer'])
            else:
                self.enable_frame(frame)
        '''
        if read_depth:
            near, far = camera.clip_from_projection(projection)
            pixels = GL.glReadPixels(
                    0, 0, width, height, GL.GL_DEPTH_COMPONENT, GL.GL_UNSIGNED_SHORT)
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

        '''
        # re-enable the multibuffer for future drawing
        if anti_alias and frame is not None:
            glBindFramebuffer(
                    GL_FRAMEBUFFER,
                    self.framebuffer_data[frame]['framebuffermulti'])
            GL.glEnable(GL.GL_MULTISAMPLE)
        '''
        GL.glViewport(0, 0, width, height)
        return image

    def start_main_loop(self, **callbacks):
        for callback_name, callback_function in callbacks.items():
            getattr(GLUT, callback_name)(callback_function)
        GLUT.glutMainLoop()

    def finish(self):
        GL.glFlush()
        GL.glFinish()
        GLUT.glutPostRedisplay()
        GLUT.glutSwapBuffers()
        GLUT.glutLeaveMainLoop()
