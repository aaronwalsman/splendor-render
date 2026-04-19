import numpy

from OpenGL import GL

import splendor.camera as camera

class FrameBufferWrapper:
    def __init__(self,
        width,
        height,
        anti_alias=True,
        anti_alias_samples=8,
        color_format=GL.GL_RGBA8,
        texture_output=False,
        depth_only=False,
    ):
        """
        Parameters
        ----------
        texture_output : bool, default=False
            If True, attach a GL_TEXTURE_2D for color instead of a renderbuffer.
            Required when this FBO's output will be sampled by a shader (e.g.
            the radial warp pass).  When combined with anti_alias=True, the FBO
            uses an MSAA renderbuffer for rendering and a single-sample texture
            as the resolve target.  Call resolve() to blit MSAA → texture before
            sampling.
        depth_only : bool, default=False
            If True, no color attachment is created.  Only the depth buffer is
            allocated, and it is always a sampleable texture.  Useful for shadow
            maps and other depth-only passes.  Incompatible with texture_output.
        """
        self.width = width
        self.height = height
        self.anti_alias = anti_alias
        self.anti_alias_samples = anti_alias_samples
        self.color_format = color_format
        self.texture_output = texture_output
        self.depth_only = depth_only

        assert not (depth_only and texture_output)
        assert self.color_format in (GL.GL_RGBA8, GL.GL_RGBA32F)

        # resolve / single-sample frame buffer
        self.frame_buffer = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.frame_buffer)

        if depth_only:
            # No color attachment — tell OpenGL explicitly
            GL.glDrawBuffer(GL.GL_NONE)
            GL.glReadBuffer(GL.GL_NONE)
        elif texture_output:
            # color texture (sampleable by shaders)
            self.texture = GL.glGenTextures(1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture)
            internal_fmt = (GL.GL_RGBA8
                            if color_format == GL.GL_RGBA8 else GL.GL_RGBA32F)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, internal_fmt,
                            width, height, 0,
                            GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
            GL.glTexParameteri(
                GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(
                GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(
                GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
            GL.glTexParameteri(
                GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
            GL.glFramebufferTexture2D(
                GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0,
                GL.GL_TEXTURE_2D, self.texture, 0)
        else:
            # color renderbuffer
            self.render_buffer = GL.glGenRenderbuffers(1)
            GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self.render_buffer)
            GL.glRenderbufferStorage(
                GL.GL_RENDERBUFFER, self.color_format, width, height)
            GL.glFramebufferRenderbuffer(
                GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0,
                GL.GL_RENDERBUFFER, self.render_buffer)

        # depth attachment: texture when texture_output or depth_only (sampleable),
        # renderbuffer otherwise
        if depth_only or texture_output:
            self.depth_texture = GL.glGenTextures(1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.depth_texture)
            GL.glTexImage2D(
                GL.GL_TEXTURE_2D, 0, GL.GL_DEPTH_COMPONENT24,
                width, height, 0,
                GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT, None)
            GL.glTexParameteri(
                GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
            GL.glTexParameteri(
                GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
            # Out-of-frustum PCF samples should return max depth (no shadow)
            GL.glTexParameteri(
                GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S,
                GL.GL_CLAMP_TO_BORDER)
            GL.glTexParameteri(
                GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T,
                GL.GL_CLAMP_TO_BORDER)
            GL.glTexParameterfv(
                GL.GL_TEXTURE_2D, GL.GL_TEXTURE_BORDER_COLOR,
                [1.0, 1.0, 1.0, 1.0])
            GL.glFramebufferTexture2D(
                GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT,
                GL.GL_TEXTURE_2D, self.depth_texture, 0)
        else:
            self.depth_buffer = GL.glGenRenderbuffers(1)
            GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self.depth_buffer)
            GL.glRenderbufferStorage(
                GL.GL_RENDERBUFFER, GL.GL_DEPTH_COMPONENT24, width, height)
            GL.glFramebufferRenderbuffer(
                GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT,
                GL.GL_RENDERBUFFER, self.depth_buffer)

        if self.anti_alias:
            # multi-sample frame buffer
            self.frame_buffer_multi = GL.glGenFramebuffers(1)
            GL.glBindFramebuffer(
                GL.GL_DRAW_FRAMEBUFFER, self.frame_buffer_multi)

            if depth_only:
                GL.glDrawBuffer(GL.GL_NONE)
                GL.glReadBuffer(GL.GL_NONE)
            else:
                # color multi-sample renderbuffer
                self.render_buffer_multi = GL.glGenRenderbuffers(1)
                GL.glBindRenderbuffer(
                    GL.GL_RENDERBUFFER, self.render_buffer_multi)
                GL.glRenderbufferStorageMultisample(
                    GL.GL_RENDERBUFFER, self.anti_alias_samples,
                    self.color_format, self.width, self.height)
                GL.glFramebufferRenderbuffer(
                    GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0,
                    GL.GL_RENDERBUFFER, self.render_buffer_multi)

            # depth multi-sample renderbuffer
            self.depth_buffer_multi = GL.glGenRenderbuffers(1)
            GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self.depth_buffer_multi)
            GL.glRenderbufferStorageMultisample(
                GL.GL_RENDERBUFFER, self.anti_alias_samples,
                GL.GL_DEPTH_COMPONENT24, width, height)
            GL.glFramebufferRenderbuffer(
                GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT,
                GL.GL_RENDERBUFFER, self.depth_buffer_multi)

    def enable(self):
        if self.anti_alias:
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.frame_buffer_multi)
            GL.glEnable(GL.GL_MULTISAMPLE)
        else:
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.frame_buffer)
            GL.glDisable(GL.GL_MULTISAMPLE)
        GL.glViewport(0, 0, self.width, self.height)
        GL.glScissor(0, 0, self.width, self.height)

    def resolve(self):
        """
        Blit the MSAA renderbuffer into the texture FBO.

        Only meaningful when both anti_alias=True and texture_output=True.
        Call this before sampling the FBO's textures in a shader.
        """
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.frame_buffer_multi)
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.frame_buffer)
        GL.glBlitFramebuffer(
            0, 0, self.width, self.height,
            0, 0, self.width, self.height,
            GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT,
            GL.GL_NEAREST)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.frame_buffer_multi)

    def read_pixels(self,
        read_alpha=False,
        read_depth=False,
        projection=None,
    ):
        if self.anti_alias:
            GL.glBindFramebuffer(
                GL.GL_READ_FRAMEBUFFER, self.frame_buffer_multi)
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.frame_buffer)
            blit_mask = GL.GL_COLOR_BUFFER_BIT
            if read_depth:
                blit_mask |= GL.GL_DEPTH_BUFFER_BIT
            GL.glBlitFramebuffer(
                0, 0, self.width, self.height,
                0, 0, self.width, self.height,
                blit_mask, GL.GL_NEAREST)
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
            image = image.astype(numpy.float32) / (2**16 - 1)
            if numpy.all(projection[3, :3] == [0, 0, 0]):
                image = image * (far - near) + near
            else:
                image = 2.0 * image - 1.0
                image = 2.0 * near * far / (far + near - image * (far - near))
        else:
            if read_alpha:
                channel_format = GL.GL_RGBA
                num_channels = 4
            else:
                channel_format = GL.GL_RGB
                num_channels = 3
            if self.color_format == GL.GL_RGBA8:
                gl_dtype = GL.GL_UNSIGNED_BYTE
                numpy_dtype = numpy.uint8
            elif self.color_format == GL.GL_RGBA32F:
                gl_dtype = GL.GL_FLOAT
                numpy_dtype = numpy.float32
            pixels = GL.glReadPixels(
                0, 0, self.width, self.height, channel_format, gl_dtype)
            image = numpy.frombuffer(pixels, dtype=numpy_dtype).reshape(
                self.height, self.width, num_channels)

        # re-enable the multibuffer for future drawing
        if self.anti_alias:
            GL.glBindFramebuffer(
                GL.GL_FRAMEBUFFER, self.frame_buffer_multi)
            GL.glEnable(GL.GL_MULTISAMPLE)
        GL.glViewport(0, 0, self.width, self.height)

        return image
