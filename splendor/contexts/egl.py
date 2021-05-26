# parts of this file written using pyrender as a reference
import os
import ctypes

import numpy

from OpenGL import GL
import OpenGL.platform

from splendor.contexts.initialization import (
        initialization_state, register_context)

EGL_PLATFORM_DEVICE_EXT = 0x313F
EGL_DRM_DEVICE_FILE_EXT = 0x3233

_egl_state = {
    'initialized' : False,
    'module' : None,
    'functions' : {},
    'structs' : {},
    'device' : None,
    'display' : None,
    'context' : None,
}

def initialize_plugin():
    new_context = register_context('egl')
    if new_context:
        plugin = OpenGL.platform.PlatformPlugin.by_name('egl')
        if plugin is None:
            raise RuntimeError('Cannot find EGL plugin.')
        plugin_class = plugin.load()
        plugin.loaded = True
        plugin = plugin_class()
        plugin.install(vars(OpenGL.platform))
    
    from OpenGL import EGL
    _egl_state['module'] = EGL
    
    _egl_state['structs']['EGLDeviceEXT'] = get_egl_struct('EGLDeviceEXT')
    _egl_state['functions']['eglGetPlatformDisplayEXT'] = get_egl_function(
            'eglGetPlatformDisplayEXT', _egl_state['module'].EGLDisplay)
    _egl_state['functions']['eglQueryDevicesEXT'] = get_egl_function(
            'eglQueryDevicesEXT', _egl_state['module'].EGLBoolean)
    _egl_state['functions']['eglQueryDeviceStringEXT'] = get_egl_function(
            'eglQueryDeviceStringEXT', ctypes.c_char_p)
    
def initialize_device(device=None, force=False):
    plugin_initialized, mode = initialization_state()
    assert plugin_initialized and mode == 'egl'
    
    if device is None:
        device = get_default_device()
    elif isinstance(device, int):
        all_devices = query_devices()
        device = all_devices[device]
    
    if _egl_state['initialized']:
        if device == _egl_state['device'] and not force:
            return False
        else:
            delete_context()
    
    _egl_state['initialized'] = True
    _egl_state['device'] = device
    
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
    
    _egl_state['display'] = _egl_state['device'].get_display()
    if original_display is not None:
        os.environ['DISPLAY'] = original_display
    
    assert eglInitialize(_egl_state['display'], major, minor)
    assert eglChooseConfig(
            _egl_state['display'], config_attributes, configs, 1, num_configs)
    assert eglBindAPI(EGL_OPENGL_API)
    
    _egl_state['context'] = eglCreateContext(
            _egl_state['display'],
            configs[0],
            EGL_NO_CONTEXT,
            context_attributes)
    
    from OpenGL.EGL import eglMakeCurrent, EGL_NO_SURFACE
    assert eglMakeCurrent(
            _egl_state['display'],
            EGL_NO_SURFACE,
            EGL_NO_SURFACE,
            _egl_state['context'])
    
    GL.glEnable(GL.GL_MULTISAMPLE)
    
    return True

def delete_context():
    plugin_initialized, mode = initialization_state()
    assert plugin_initialized and mode == 'egl'
    
    from OpenGL.EGL import eglDestroyContext, eglTerminate
    if _egl_state['display'] is not None:
        if _egl_state['context'] is not None:
            eglDestroyContext(_egl_state['display'], _egl_state['context'])
            _egl_state['context'] = None
        eglTerminate(_egl_state['display'])
        _egl_state['display'] = None
    
    _egl_state['device'] = None

def get_egl_function(function_name, return_type, *argtypes):
    plugin_initialized, mode = initialization_state()
    assert plugin_initialized and mode == 'egl'
    
    address = _egl_state['module'].eglGetProcAddress(function_name)
    if address is None:
        return None
    
    proto = ctypes.CFUNCTYPE(return_type)
    proto.argtypes = argtypes
    function = proto(address)
    return function

def get_egl_struct(struct_name):
    plugin_initialized, mode = initialization_state()
    assert plugin_initialized and mode == 'egl'
    
    from OpenGL._opaque import opaque_pointer_cls
    return opaque_pointer_cls(struct_name)

class EGLDevice:
    def __init__(self, display=None):
        plugin_initialized, mode = initialization_state()
        assert plugin_initialized and mode == 'egl'
        
        self.display = display
    
    def get_display(self):
        if self.display is None:
            return _egl_state['module'].eglGetDisplay(EGL.EGL_DEFAULT_DISPLAY)
        
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
        return '<EGLDevice(name={})>'.format(self.name)

def query_devices():
    plugin_initialized, mode = initialization_state()
    assert plugin_initialized and mode == 'egl'
    
    if _egl_state['functions']['eglQueryDevicesEXT'] is None:
        raise RuntimeError('EGL query extension not available')
    
    num_devices = _egl_state['module'].EGLint()
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

def get_default_device():
    plugin_initialized, mode = initialization_state()
    assert plugin_initialized and mode == 'egl'
    
    if _egl_state['functions']['eglQueryDevicesEXT'] is None:
        return EGLDevice(None)
    
    return query_devices()[0]

'''
def finish():
    GL.glFlush()
    GL.glFinish()
'''
