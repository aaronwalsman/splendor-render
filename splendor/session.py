import OpenGL.GL as GL

from splendor.context.egl import EGLContext

from splendor.exception import SplendorException
from splendor.shader_library import ShaderLibrary

class SessionException(SplendorException):
    pass

class NoActiveSessionException(SessionException):
    pass

def active_render_session():
    return RenderSession._active_sessions[-1]

class RenderSession:
    '''
    Manages the render context and tracks all content for serialization purposes
    '''
    
    _active_sessions = []
    
    def __init__(self, context_type='EGL'):
        #self._context = initialize_context(context_type)
        self._context = EGLContext()
        #self._content = Content()
    
    @property
    def context(self):
        return self._context
    
    @property
    def shader_library(self):
        return self._shader_library
    
    @property
    def content(self):
        return self._content
    
    @property
    def active_session(self):
        if len(self._active_sessions) == 0:
            raise NoActiveSessionException('no active session')
        return self._active_sessions[-1]
    
    def add_content(self, content):
        if content.CONTENT_CATEGORY not in self._session_content:
            self._session_content[content.CONTENT_CATEGORY] = {}
        
        if content.name in self._session_content[content.CONTENT_CATEGORY]:
            raise NameException(
                f'{content.CONTENT_CATEGORY} named {name} already exists')
        self._session_content[content.CONTENT_CATEGORY][content.name] = content
    
    #def list_content(self, content_type):
    #    if 
    #    return list(self._session_content
    
    def activate(self):
        self._context.activate()
        self._active_sessions.append(self)
        self._shader_library = ShaderLibrary()
        
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_SCISSOR_TEST)
        GL.glEnable(GL.GL_TEXTURE_CUBE_MAP_SEAMLESS)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glDepthFunc(GL.GL_LESS)
        GL.glDepthRange(0., 1.)
        
        GL.glClearColor(0.,0.,0.,0.)
    
    def deactivate(self):
        self._context.deactivate()
        if self._active_sessions[-1] is not self:
            raise SessionException('this session is not active')
        self._active_sessions.pop()
    
    def __enter__(self):
        self.activate()
    
    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.deactivate()

class SessionContent:
    
    def __init__(self, name=None):
        session = RenderSession.active_session
        name = session
