import numpy

from OpenGL import GL

import renderpy.camera as camera

class FrameBufferWrapper:
    def __init__(self, width, height, anti_alias=True, anti_alias_samples=8):

        self.width = width
        self.height = height
        self.anti_alias = anti_alias
        self.anti_alias_samples = anti_alias_samples

        # frame buffer
        self.frame_buffer = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.frame_buffer)

        # color renderbuffer
        self.render_buffer = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self.render_buffer)
        GL.glRenderbufferStorage(
                GL.GL_RENDERBUFFER, GL.GL_RGBA8, width, height)
        GL.glFramebufferRenderbuffer(
                GL.GL_FRAMEBUFFER,
                GL.GL_COLOR_ATTACHMENT0,
                GL.GL_RENDERBUFFER,
                self.render_buffer)

        # depth renderbuffer
        self.depth_buffer = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self.depth_buffer)
        GL.glRenderbufferStorage(
                GL.GL_RENDERBUFFER,
                GL.GL_DEPTH_COMPONENT24,
                self.width,
                self.height,
        )
        GL.glFramebufferRenderbuffer(
                GL.GL_FRAMEBUFFER,
                GL.GL_DEPTH_ATTACHMENT,
                GL.GL_RENDERBUFFER,
                self.depth_buffer)

        if self.anti_alias:
            # multi-sample frame buffer
            self.frame_buffer_multi = GL.glGenFramebuffers(1)
            GL.glBindFramebuffer(
                    GL.GL_DRAW_FRAMEBUFFER,
                    self.frame_buffer_multi,
            )

            # color multi-sample renderbuffer
            self.render_buffer_multi = GL.glGenRenderbuffers(1)
            GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self.render_buffer_multi)
            GL.glRenderbufferStorageMultisample(
                    GL.GL_RENDERBUFFER,
                    self.anti_alias_samples,
                    GL.GL_RGBA8,
                    self.width,
                    self.height)
            GL.glFramebufferRenderbuffer(
                    GL.GL_FRAMEBUFFER,
                    GL.GL_COLOR_ATTACHMENT0,
                    GL.GL_RENDERBUFFER,
                    self.render_buffer_multi)

            # depth multi-sample renderbuffer
            self.depth_buffer_multi = GL.glGenRenderbuffers(1)
            GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self.depth_buffer_multi)
            GL.glRenderbufferStorageMultisample(
                    GL.GL_RENDERBUFFER,
                    self.anti_alias_samples,
                    GL.GL_DEPTH_COMPONENT16,
                    width,
                    height)
            GL.glFramebufferRenderbuffer(
                    GL.GL_FRAMEBUFFER,
                    GL.GL_DEPTH_ATTACHMENT,
                    GL.GL_RENDERBUFFER,
                    self.depth_buffer_multi)

    def enable(self):
        if self.anti_alias:
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.frame_buffer_multi)
            GL.glEnable(GL.GL_MULTISAMPLE)
        else:
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.frame_buffer)
            GL.glDisable(GL.GL_MULTISAMPLE)
        GL.glViewport(0, 0, self.width, self.height)

    def read_pixels(self,
            read_depth = False,
            projection = None):

        if self.anti_alias:
            GL.glBindFramebuffer(
                    GL.GL_READ_FRAMEBUFFER,
                    self.frame_buffer_multi,
            )
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.frame_buffer)
            GL.glBlitFramebuffer(
                    0, 0, self.width, self.height,
                    0, 0, self.width, self.height,
                    GL.GL_COLOR_BUFFER_BIT, GL.GL_NEAREST)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.frame_buffer)
        else:
            self.enable()

        if read_depth:
            near, far = camera.clip_from_projection(projection)
            pixels = GL.glReadPixels(
                    0, 0, self.width, self.height,
                    GL.GL_DEPTH_COMPONENT, GL.GL_UNSIGNED_SHORT)
            image = numpy.frombuffer(pixels, dtype=numpy.ushort).reshape(
                    self.height, self.width, 1)
            image = image.astype(numpy.float) / (2**16-1)
            image = 2.0 * image - 1.0
            image = 2.0 * near * far / (far + near - image * (far - near))
        else:
            pixels = GL.glReadPixels(
                    0,
                    0,
                    self.width,
                    self.height,
                    GL.GL_RGB,
                    GL.GL_UNSIGNED_BYTE
            )
            image = numpy.frombuffer(pixels, dtype=numpy.uint8).reshape(
                    self.height, self.width, 3)

        # re-enable the multibuffer for future drawing
        if self.anti_alias:
            GL.glBindFramebuffer(
                    GL.GL_FRAMEBUFFER,
                    self.frame_buffer_multi)
            GL.glEnable(GL.GL_MULTISAMPLE)
        GL.glViewport(0, 0, self.width, self.height)

        return image

