"""Interactive OpenGL context via GLFW with window and input handling."""
import os

import numpy
from OpenGL import GL
import glfw as _glfw

import splendor.camera as camera
from splendor.contexts import _register_context


class GLFWContext:
    """
    Interactive OpenGL context backed by a GLFW window.

    The window surface acts as the default framebuffer.  Additional
    FrameBufferWrappers can be created for offscreen passes (masks,
    depth, etc.) and are context-agnostic.  Supports use as a context
    manager.

    Parameters
    ----------
    width, height : int
        Window dimensions in screen coordinates.
    title : str
        Window title bar text.
    anti_alias : bool, default=False
        Enable MSAA anti-aliasing on the window surface.
    anti_alias_samples : int, default=8
        Number of MSAA samples (only used when anti_alias=True).
    x_authority : str, optional
        XAUTHORITY path, for remote display connections.
    display : str, optional
        DISPLAY variable, for remote display connections.

    Examples
    --------
    ::

        with GLFWContext(512, 512, title='My Viewer') as ctx:
            renderer = SplendorRender()
            while not ctx.should_close():
                ctx.poll_events()
                ctx.enable()
                renderer.color_render()
                ctx.swap_buffers()
    """

    def __init__(self,
        width,
        height,
        title='Splendor',
        anti_alias=False,
        anti_alias_samples=8,
        x_authority=None,
        display=None,
    ):
        _register_context('glfw')

        if x_authority is not None:
            os.environ['XAUTHORITY'] = x_authority
            os.environ['DISPLAY'] = display

        if not _glfw.init():
            raise RuntimeError('GLFW cannot be initialized.')

        self.width = width
        self.height = height
        self.anti_alias = anti_alias
        self.anti_alias_samples = anti_alias_samples

        _glfw.window_hint(_glfw.CONTEXT_VERSION_MAJOR, 3)
        _glfw.window_hint(_glfw.CONTEXT_VERSION_MINOR, 3)
        _glfw.window_hint(_glfw.OPENGL_PROFILE, _glfw.OPENGL_CORE_PROFILE)
        _glfw.window_hint(_glfw.OPENGL_FORWARD_COMPAT, True)
        if anti_alias:
            _glfw.window_hint(_glfw.SAMPLES, anti_alias_samples)

        self._window = _glfw.create_window(width, height, title, None, None)
        if self._window is None:
            _glfw.terminate()
            raise RuntimeError('GLFW window cannot be created.')

        _glfw.make_context_current(self._window)

    # -- window surface --------------------------------------------------------

    def enable(self):
        """
        Bind the window's default framebuffer and set the viewport.

        Call this before rendering a pass that should appear in the window,
        or after switching back from a FrameBufferWrapper.
        """
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        fbw, fbh = _glfw.get_framebuffer_size(self._window)
        GL.glViewport(0, 0, fbw, fbh)
        GL.glScissor(0, 0, fbw, fbh)
        if self.anti_alias:
            GL.glEnable(GL.GL_MULTISAMPLE)
        else:
            GL.glDisable(GL.GL_MULTISAMPLE)

    def framebuffer_size(self):
        """Return (width, height) of the window framebuffer in pixels.

        On HiDPI displays this may differ from the screen-coordinate size
        passed to the constructor.
        """
        return _glfw.get_framebuffer_size(self._window)

    def read_pixels(self, read_depth=False, projection=None):
        """
        Read pixels from the window framebuffer.

        Parameters
        ----------
        read_depth : bool, default=False
            Read the depth buffer instead of color.  Returned as a float
            array of linear eye-space depth values.
        projection : 4x4 array, optional
            Required when read_depth=True for linearization.

        Returns
        -------
        numpy array, shape (H, W, 4) uint8 for color or (H, W, 1) float32
        for depth.
        """
        self.enable()
        fbw, fbh = _glfw.get_framebuffer_size(self._window)
        if read_depth:
            if projection is None:
                raise ValueError(
                    'Must specify a projection when reading depth.')
            near, far = camera.clip_from_projection(projection)
            pixels = GL.glReadPixels(
                0, 0, fbw, fbh, GL.GL_DEPTH_COMPONENT, GL.GL_UNSIGNED_SHORT)
            image = numpy.frombuffer(pixels, dtype=numpy.uint16).reshape(
                fbh, fbw, 1)
            image = image.astype(numpy.float32) / (2**16 - 1)
            image = 2.0 * image - 1.0
            image = 2.0 * near * far / (far + near - image * (far - near))
        else:
            pixels = GL.glReadPixels(
                0, 0, fbw, fbh, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
            image = numpy.frombuffer(pixels, dtype=numpy.uint8).reshape(
                fbh, fbw, 4)
        return image

    # -- event loop ------------------------------------------------------------

    def should_close(self):
        """Return True if the user has requested the window to close."""
        return _glfw.window_should_close(self._window)

    def poll_events(self):
        """Process pending window and input events."""
        _glfw.poll_events()

    def swap_buffers(self):
        """Swap front and back buffers to display the rendered frame."""
        _glfw.swap_buffers(self._window)

    # -- input callbacks -------------------------------------------------------

    def set_mouse_button_callback(self, fn):
        _glfw.set_mouse_button_callback(self._window, fn)

    def set_cursor_pos_callback(self, fn):
        _glfw.set_cursor_pos_callback(self._window, fn)

    def set_key_callback(self, fn):
        _glfw.set_key_callback(self._window, fn)

    def set_scroll_callback(self, fn):
        _glfw.set_scroll_callback(self._window, fn)

    # -- lifecycle -------------------------------------------------------------

    def close(self):
        """Destroy the window and terminate GLFW."""
        _glfw.terminate()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
