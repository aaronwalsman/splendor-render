"""Headless OpenGL context via EGL for offscreen rendering."""
import os
import ctypes

from OpenGL import GL
import OpenGL.platform

from splendor.contexts import _register_context

EGL_PLATFORM_DEVICE_EXT = 0x313F
EGL_DRM_DEVICE_FILE_EXT = 0x3233

_egl_state = {
    'plugin_initialized': False,
    'module': None,
    'functions': {},
    'structs': {},
}

def _initialize_plugin():
    if _egl_state['plugin_initialized']:
        return
    _register_context('egl')
    plugin = OpenGL.platform.PlatformPlugin.by_name('egl')
    if plugin is None:
        raise RuntimeError('Cannot find EGL plugin.')
    plugin_class = plugin.load()
    plugin.loaded = True
    plugin = plugin_class()
    plugin.install(vars(OpenGL.platform))

    from OpenGL import EGL
    _egl_state['module'] = EGL
    _egl_state['structs']['EGLDeviceEXT'] = _get_egl_struct('EGLDeviceEXT')
    _egl_state['functions']['eglGetPlatformDisplayEXT'] = _get_egl_function(
        'eglGetPlatformDisplayEXT', EGL.EGLDisplay)
    _egl_state['functions']['eglQueryDevicesEXT'] = _get_egl_function(
        'eglQueryDevicesEXT', EGL.EGLBoolean)
    _egl_state['functions']['eglQueryDeviceStringEXT'] = _get_egl_function(
        'eglQueryDeviceStringEXT', ctypes.c_char_p)
    _egl_state['plugin_initialized'] = True

def _get_egl_function(function_name, return_type, *argtypes):
    address = _egl_state['module'].eglGetProcAddress(function_name)
    if address is None:
        return None
    proto = ctypes.CFUNCTYPE(return_type)
    proto.argtypes = argtypes
    return proto(address)

def _get_egl_struct(struct_name):
    from OpenGL._opaque import opaque_pointer_cls
    return opaque_pointer_cls(struct_name)


class EGLDevice:
    """Represents a single EGL-capable GPU device."""

    def __init__(self, display=None):
        self.display = display

    def get_display(self):
        EGL = _egl_state['module']
        if self.display is None:
            return EGL.eglGetDisplay(EGL.EGL_DEFAULT_DISPLAY)
        return _egl_state['functions']['eglGetPlatformDisplayEXT'](
            EGL_PLATFORM_DEVICE_EXT, self.display, None)

    @property
    def name(self):
        if self.display is None:
            return 'default'
        name = _egl_state['functions']['eglQueryDeviceStringEXT'](
            self.display, EGL_DRM_DEVICE_FILE_EXT)
        if name is None:
            return None
        return name.decode('ascii')

    def __eq__(self, other):
        return self.name == other.name

    def __repr__(self):
        return f'<EGLDevice(name={self.name})>'


def query_devices():
    """
    Return a list of available EGL devices.

    EGLContext must have been created before calling this.

    Returns
    -------
    list of EGLDevice
    """
    if not _egl_state['plugin_initialized']:
        raise RuntimeError(
            'query_devices() requires an EGLContext to be created first.')
    if _egl_state['functions'].get('eglQueryDevicesEXT') is None:
        raise RuntimeError('EGL device query extension not available.')
    EGL = _egl_state['module']
    num_devices = EGL.EGLint()
    success = _egl_state['functions']['eglQueryDevicesEXT'](
        0, None, ctypes.pointer(num_devices))
    if not success or num_devices.value < 0:
        return []
    devices = (_egl_state['structs']['EGLDeviceEXT'] * num_devices.value)()
    success = _egl_state['functions']['eglQueryDevicesEXT'](
        num_devices.value, devices, ctypes.pointer(num_devices))
    if not success or num_devices.value < 1:
        return []
    return [EGLDevice(devices[i]) for i in range(num_devices.value)]


class EGLContext:
    """
    Headless OpenGL context backed by EGL.

    Does not create a framebuffer — attach one or more FrameBufferWrappers
    to render into.

    The context is made current immediately on construction.  Use as a context
    manager to temporarily acquire/release the context around a block of GL
    work — useful when multiple rendering systems share a thread:

        ctx = EGLContext()
        try:
            with ctx:               # make_current() on enter
                renderer.color_render()
            # release() called on exit — other systems can now take the thread
        finally:
            ctx.close()             # destroy when completely done

    For single-owner use where nothing else needs the thread, the simpler
    pattern still works — just call close() explicitly when done:

        ctx = EGLContext()
        renderer = SplendorRender()
        renderer.color_render()
        ctx.close()

    Parameters
    ----------
    device : int, EGLDevice, or None
        Which GPU to use.  None selects the default device.  An integer
        indexes into the list returned by query_devices().
    """

    def __init__(self, device=None):
        _initialize_plugin()

        # resolve device
        if device is None:
            if _egl_state['functions'].get('eglQueryDevicesEXT') is None:
                device = EGLDevice(None)
            else:
                device = query_devices()[0]
        elif isinstance(device, int):
            device = query_devices()[device]

        self._device = device
        self._display = None
        self._context = None
        self._init_context()

    def _init_context(self):
        from OpenGL.EGL import (
            EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
            EGL_BLUE_SIZE, EGL_RED_SIZE, EGL_GREEN_SIZE, EGL_DEPTH_SIZE,
            EGL_COLOR_BUFFER_TYPE, EGL_RGB_BUFFER,
            EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT, EGL_CONFORMANT,
            EGL_NONE, EGL_NO_CONTEXT,
            EGL_OPENGL_API, EGL_CONTEXT_MAJOR_VERSION,
            EGL_CONTEXT_MINOR_VERSION,
            EGL_CONTEXT_OPENGL_PROFILE_MASK,
            EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
            eglInitialize, eglChooseConfig,
            eglBindAPI, eglCreateContext, eglMakeCurrent,
            EGL_NO_SURFACE, EGLConfig,
        )

        config_attributes = GL.arrays.GLintArray.asArray([
            EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
            EGL_BLUE_SIZE, 8,
            EGL_RED_SIZE, 8,
            EGL_GREEN_SIZE, 8,
            EGL_DEPTH_SIZE, 24,
            EGL_COLOR_BUFFER_TYPE, EGL_RGB_BUFFER,
            EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
            EGL_CONFORMANT, EGL_OPENGL_BIT,
            EGL_NONE,
        ])
        context_attributes = GL.arrays.GLintArray.asArray([
            EGL_CONTEXT_MAJOR_VERSION, 3,
            EGL_CONTEXT_MINOR_VERSION, 3,
            EGL_CONTEXT_OPENGL_PROFILE_MASK,
            EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
            EGL_NONE,
        ])

        major = ctypes.c_long()
        minor = ctypes.c_long()
        num_configs = ctypes.c_long()
        configs = (EGLConfig * 1)()

        # temporarily remove DISPLAY so EGL doesn't try to connect to X11
        original_display = os.environ.pop('DISPLAY', None)
        self._display = self._device.get_display()
        if original_display is not None:
            os.environ['DISPLAY'] = original_display

        assert eglInitialize(self._display, major, minor)
        assert eglChooseConfig(
            self._display, config_attributes, configs, 1, num_configs)
        assert eglBindAPI(EGL_OPENGL_API)
        self._context = eglCreateContext(
            self._display, configs[0], EGL_NO_CONTEXT, context_attributes)
        assert eglMakeCurrent(
            self._display, EGL_NO_SURFACE, EGL_NO_SURFACE, self._context)

        GL.glEnable(GL.GL_MULTISAMPLE)

    def make_current(self):
        """Make this EGL context current on the calling thread."""
        from OpenGL.EGL import eglMakeCurrent, EGL_NO_SURFACE
        assert eglMakeCurrent(
            self._display, EGL_NO_SURFACE, EGL_NO_SURFACE, self._context)

    def release(self):
        """
        Release this context from the calling thread without destroying it.

        After this call the thread has no current GL context, allowing another
        rendering system (e.g. Isaac) to make its own context current.  Call
        make_current() to reacquire.
        """
        from OpenGL.EGL import eglMakeCurrent, EGL_NO_SURFACE, EGL_NO_CONTEXT
        assert eglMakeCurrent(
            self._display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT)

    def close(self):
        """Release and destroy the EGL context and display."""
        from OpenGL.EGL import eglDestroyContext, eglTerminate
        if self._display is not None:
            if self._context is not None:
                eglDestroyContext(self._display, self._context)
                self._context = None
            eglTerminate(self._display)
            self._display = None

    def __enter__(self):
        self.make_current()
        return self

    def __exit__(self, *args):
        self.release()

