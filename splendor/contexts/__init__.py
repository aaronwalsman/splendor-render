"""OpenGL context management — EGL (headless) and GLFW (interactive)."""
from splendor.exceptions import SplendorContextException

_context_state = {
    'initialized': False,
    'mode': None,
}

def _register_context(mode):
    """
    Internal: claim a context type for this process.

    EGL installs itself into OpenGL.platform at import time, so mixing
    EGL and GLFW in a single process is not possible.  This raises a
    clear error rather than letting things fail mysteriously later.
    """
    if not _context_state['initialized']:
        _context_state['initialized'] = True
        _context_state['mode'] = mode
    elif _context_state['mode'] != mode:
        raise SplendorContextException(
            f'Cannot create a {mode!r} context: a {_context_state["mode"]!r} '
            f'context is already active in this process.')

from splendor.contexts.egl import EGLContext, query_devices
from splendor.contexts.glfw import GLFWContext

def context(interactive=False, **kwargs):
    """
    Create an OpenGL rendering context.

    Parameters
    ----------
    interactive : bool, default=False
        If True, creates a visible GLFW window (GLFWContext).
        If False, creates a headless EGL context (EGLContext).
    **kwargs
        Passed through to GLFWContext or EGLContext.

    Returns
    -------
    EGLContext or GLFWContext
        Supports use as a context manager (with statement).

    Examples
    --------
    Headless rendering::

        with splendor.context() as ctx:
            fb = FrameBufferWrapper(512, 512)
            renderer = SplendorRender()
            fb.enable()
            renderer.color_render()
            image = fb.read_pixels()

    Interactive window::

        with splendor.context(interactive=True, width=512, height=512) as ctx:
            while not ctx.should_close():
                ctx.poll_events()
                ctx.enable()
                renderer.color_render()
                ctx.swap_buffers()
    """
    if interactive:
        return GLFWContext(**kwargs)
    else:
        return EGLContext(**kwargs)
