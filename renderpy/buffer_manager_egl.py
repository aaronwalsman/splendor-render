# parts of this file written using pyrender as a reference
import os
import ctypes

import numpy

from OpenGL import GL
import OpenGL.platform

EGL_PLATFORM_DEVICE_EXT = 0x313F
EGL_DRM_DEVICE_FILE_EXT = 0x3233

def load_egl():
    plugin = OpenGL.platform.PlatformPlugin.by_name('egl')
    if plugin is None:
        raise RuntimeError('Cannot find EGL plugin.')
    plugin_class = plugin.load()
    plugin.loaded = True
    plugin = plugin_class()

    plugin.install(vars(OpenGL.platform))

load_egl()
from OpenGL import EGL

def get_egl_function(function_name, return_type, *argtypes):
    address = EGL.eglGetProcAddress(function_name)
    if address is None:
        return None

    proto = ctypes.CFUNCTYPE(return_type)
    proto.argtypes = argtypes
    function = proto(address)
    return function

def get_egl_struct(struct_name):
    from OpenGL._opaque import opaque_pointer_cls
    return opaque_pointer_cls(struct_name)

EGLDeviceEXT = get_egl_struct('EGLDeviceEXT')
eglGetPlatformDisplayEXT = get_egl_function(
        'eglGetPlatformDisplayEXT', EGL.EGLDisplay)
eglQueryDevicesEXT = get_egl_function('eglQueryDevicesEXT', EGL.EGLBoolean)
eglQueryDeviceStringEXT = get_egl_function(
        'eglQueryDeviceStringEXT', ctypes.c_char_p)

egl_state = {
    'buffer_manager' : None
}
def initialize_shared_buffer_manager(*args, **kwargs):
    if egl_state['buffer_manager'] is None:
        egl_state['buffer_manager'] = BufferManagerEGL(*args, **kwargs)
    return egl_state['buffer_manager']

class EGLDevice:
    def __init__(self, display=None):
        self.display = display

    def get_display(self):
        if self.display is None:
            return EGL.eglGetDisplay(EGL.EGL_DEFAULT_DISPLAY)

        return eglGetPlatformDisplayEXT(
                EGL_PLATFORM_DEVICE_EXT, self.display, None)

    @property
    def name(self):
        if self.display is None:
            return 'default'

        name = eglQueryDeviceStringEXT(self.display, EGL_DRM_DEVICE_FILE_EXT)
        if name is None:
            return None

        return name.decode('ascii')

    def __repr__(self):
        return '<EGLDevice(name={})>'.format(self.name)

def query_devices():
    if eglQueryDevicesEXT is None:
        raise RuntimeError('EGL query extension not available')

    num_devices = EGL.EGLint()
    success = eglQueryDevicesEXT(0, None, ctypes.pointer(num_devices))
    if not success or num_devices.value < 0:
        return []

    devices = (EGLDeviceEXT * num_devices.value)()
    success = eglQueryDevicesEXT(
            num_devices.value, devices, ctypes.pointer(num_devices))
    if not success or num_devices.value < 1:
        return []

    return [EGLDevice(devices[i]) for i in range(num_devices.value)]

def get_default_device():
    if eglQueryDevicesEXT is None:
        return EGLDevice(None)

    return query_devices()[0]

class BufferManagerEGL:
    def __init__(self, device = None):

        if device is None:
            device = get_default_device()

        self.egl_device = device

        from OpenGL.EGL import (
                EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
                EGL_BLUE_SIZE, EGL_RED_SIZE, EGL_GREEN_SIZE, EGL_DEPTH_SIZE,
                EGL_COLOR_BUFFER_TYPE, EGL_RGB_BUFFER,
                EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT, EGL_CONFORMANT,
                EGL_NONE, EGL_DEFAULT_DISPLAY, EGL_NO_CONTEXT,
                EGL_OPENGL_API, EGL_CONTEXT_MAJOR_VERSION,
                EGL_CONTEXT_MINOR_VERSION,
                EGL_CONTEXT_OPENGL_PROFILE_MASK,
                EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
                eglGetDisplay, eglInitialize, eglChooseConfig,
                eglBindAPI, eglCreateContext, EGLConfig)

        config_attributes = GL.arrays.GLintArray.asArray([
                EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
                EGL_BLUE_SIZE, 8,
                EGL_RED_SIZE, 8,
                EGL_GREEN_SIZE, 8,
                EGL_DEPTH_SIZE, 24,
                EGL_COLOR_BUFFER_TYPE, EGL_RGB_BUFFER,
                EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
                EGL_CONFORMANT, EGL_OPENGL_BIT,
                EGL_NONE])

        context_attributes = GL.arrays.GLintArray.asArray([
                EGL_CONTEXT_MAJOR_VERSION, 3,
                EGL_CONTEXT_MINOR_VERSION, 1,
                EGL_CONTEXT_OPENGL_PROFILE_MASK,
                EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
                EGL_NONE])

        major = ctypes.c_long()
        minor = ctypes.c_long()
        num_configs = ctypes.c_long()
        configs = (EGLConfig * 1)()

        original_display = None
        if 'DISPLAY' in os.environ:
            original_display = os.environ['DISPLAY']
            del os.environ['DISPLAY']

        self.egl_display = self.egl_device.get_display()
        if original_display is not None:
            os.environ['DISPLAY'] = original_display

        assert eglInitialize(self.egl_display, major, minor)
        assert eglChooseConfig(
                self.egl_display, config_attributes, configs, 1, num_configs)

        assert eglBindAPI(EGL_OPENGL_API)

        self.egl_context = eglCreateContext(
            self.egl_display, configs[0], EGL_NO_CONTEXT, context_attributes)

        from OpenGL.EGL import eglMakeCurrent, EGL_NO_SURFACE
        assert eglMakeCurrent(
                self.egl_display,
                EGL_NO_SURFACE,
                EGL_NO_SURFACE,
                self.egl_context)

        GL.glEnable(GL.GL_MULTISAMPLE)

    def delete_context(self):
        from OpenGL.EGL import eglDestroyContext, eglTerminate
        if self.egl_display is not None:
            if self.egl_context is not None:
                eglDestroyContext(self.egl_display, self.egl_context)
                self.egl_context = None
            eglTerminate(self.egl_display)
            self.egl_display = None

    def finish():
        GL.glFlush()
        GL.glFinish()
