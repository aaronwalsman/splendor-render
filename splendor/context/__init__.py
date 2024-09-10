from splendor.exception import SplendorException
from splendor.context.egl import EGLContext

class ContextException(SplendorException):
    pass

def initailize_context(context_type):
    if context_type == 'EGL':
        return EGLContext()
    else:
        raise ContextException(f'unknown context type: {context_type}')
