import os
import ctypes

import OpenGL.GL as GL
import OpenGL.platform as platform
from OpenGL._opaque import opaque_pointer_cls

EGL = None
#EGLDeviceEXT = None
#eglGetPlatformDisplayEXT = None
#eglQueryDevicesEXT = None
#eglQueryDeviceStringEXT = None

EGL_PLATFORM_DEVICE_EXT = 0x313F
EGL_DRM_DEVICE_FILE_EXT = 0x3233

def egl_function(function_name, return_type, *argtypes):
    address = EGL.eglGetProcAddress(function_name)
    if address is None:
        return None
    
    proto = ctypes.CFUNCTYPE(return_type)
    proto.argtypes = argtypes
    function = proto(address)
    return function

def egl_struct(struct_name):
    return opaque_pointer_cls(struct_name)

def load_EGL_plugin():
    global EGL
    global EGLDeviceEXT
    global eglGetPlatformDisplayEXT
    global eglQueryDevicesEXT
    global eglQueryDeviceStringEXT
    global EGL_SURFACE_TYPE
    global EGL_PBUFFER_BIT
    global EGL_BLUE_SIZE
    global EGL_RED_SIZE
    global EGL_GREEN_SIZE
    global EGL_DEPTH_SIZE
    global EGL_COLOR_BUFFER_TYPE
    global EGL_RGB_BUFFER
    global EGL_RENDERABLE_TYPE
    global EGL_OPENGL_BIT
    global EGL_CONFORMANT
    global EGL_NONE
    global EGL_DEFAULT_DISPLAY
    global EGL_NO_CONTEXT
    global EGL_OPENGL_API
    global EGL_CONTEXT_MAJOR_VERSION
    global EGL_CONTEXT_MINOR_VERSION
    global EGL_CONTEXT_OPENGL_PROFILE_MASK
    global EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT
    global eglGetDisplay
    global eglInitialize
    global eglChooseConfig
    global eglBindAPI
    global eglCreateContext
    global EGLConfig
    global eglMakeCurrent
    global EGL_NO_SURFACE
    global eglDestroyContext
    global eglTerminate
    
    if EGL is None:
        
        # load the plugin
        plugin = platform.PlatformPlugin.by_name('egl')
        assert plugin is not None, 'Cannot find EGL plugin.'
        plugin_class = plugin.load()
        plugin.loaded = True
        plugin = plugin_class()
        plugin.install(vars(platform))
        
        # import EGL
        from OpenGL import EGL
        from OpenGL.EGL import (
            EGL_SURFACE_TYPE,
            EGL_PBUFFER_BIT,
            EGL_BLUE_SIZE,
            EGL_RED_SIZE,
            EGL_GREEN_SIZE,
            EGL_DEPTH_SIZE,
            EGL_COLOR_BUFFER_TYPE,
            EGL_RGB_BUFFER,
            EGL_RENDERABLE_TYPE,
            EGL_OPENGL_BIT,
            EGL_CONFORMANT,
            EGL_NONE,
            EGL_DEFAULT_DISPLAY,
            EGL_NO_CONTEXT,
            EGL_OPENGL_API,
            EGL_CONTEXT_MAJOR_VERSION,
            EGL_CONTEXT_MINOR_VERSION,
            EGL_CONTEXT_OPENGL_PROFILE_MASK,
            EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
            eglGetDisplay,
            eglInitialize,
            eglChooseConfig,
            eglBindAPI,
            eglCreateContext,
            EGLConfig,
            eglMakeCurrent,
            EGL_NO_SURFACE,
            eglDestroyContext,
            eglTerminate,
        )
        
        # get EGL structs and functions
        EGLDeviceEXT = egl_struct('EGLDeviceEXT')
        eglGetPlatformDisplayEXT = egl_function(
            'eglGetPlatformDisplayEXT', EGL.EGLDisplay)
        eglQueryDevicesEXT = egl_function(
            'eglQueryDevicesEXT', EGL.EGLBoolean)
        eglQueryDeviceStringEXT = egl_function(
            'eglQueryDeviceStringEXT', ctypes.c_char_p)

class EGLContext:
    def __init__(self, device=None):
        load_EGL_plugin()
        
        if device is None:
            device = EGLDevice.default_device()
        elif not isinstance(device, EGLDevice):
            device = self.available_devices()[device]
        self._device = device
        self._create_context()
    
    def _create_context(self):
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
        
        assert eglInitialize(self._device.display, major, minor)
        assert eglChooseConfig(
            self._device.display, config_attributes, configs, 1, num_configs)
        assert eglBindAPI(EGL_OPENGL_API)
        
        self._context = eglCreateContext(
            self._device.display,
            configs[0],
            EGL_NO_CONTEXT,
            context_attributes,
        )
    
    def _destroy_context(self):
        if self._device.display is not None:
            eglDestroyContext(self._device.display, self._context)
            self._context = None
        eglTerminate(self._device.display)
    
    def activate(self):
        assert eglMakeCurrent(
            self._device.display,
            EGL_NO_SURFACE,
            EGL_NO_SURFACE,
            self._context,
        )
        
        GL.glEnable(GL.GL_MULTISAMPLE)
    
    def deactivate(self):
        pass
    
    '''
    def __enter__(self):
        self.enable()
    
    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.disable()
    '''
    
    def __del__(self):
        self._destroy_context()

class EGLDevice:
    def __init__(self, display=None):
        
        original_display = None
        if 'DISPLAY' in os.environ:
            original_display = os.environ['DISPLAY']
            del os.environ['DISPLAY']
        
        if display is None:
            self._display = EGL.eglGetDisplay(EGL.EGL_DEFAULT_DISPLAY)
        else:
            self._display = eglGetPlatformDisplayEXT(
                EGL_PLATFORM_DEVICE_EXT, display, None)
        
        if original_display is not None:
            os.environ['DISPLAY'] = original_display
    
    @property
    def display(self):
        return self._display
    
    @property
    def name(self):
        if self.display is None:
            return 'default'
        
        name = eglQueryDeviceStringEXT(
            self.display, EGL_DRM_DEVICE_FILE_EXT)
        if name is None:
            return None
        
        return name.decode('ascii')
    
    def __eq__(self, other):
        return self.name == other.name
    
    def __repr__(self):
        return f'<EGLDevice(name={self.name})>'

    @staticmethod
    def available_devices():
        if eglQueryDevicesEXT is None:
            raise RuntimeError('EGL query extension not available')
        
        num_devices = EGL.EGLint()
        success = eglQueryDevicesEXT(0, None, ctypes.pointer(num_devices))
        if not success or num_devices.value < 1:
            return []
        
        devices = (EGLDeviceEXT * num_devices.value)()
        success = eglQueryDevicesEXT(
            num_devices.value, devices, ctypes.pointer(num_devices))
        if not success or num_devices.value < 1:
            return []
        
        return [EGLDevice(devices[i]) for i in range(num_devices.value)]
    
    @staticmethod
    def default_device():
        return EGLDevice.available_devices()[0]
