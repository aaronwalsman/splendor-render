"""Exception classes for splendor-render."""
class SplendorException(Exception):
    """Base exception for splendor-render."""
    pass

class SplendorAssetException(SplendorException):
    """Raised when an asset cannot be found."""
    pass

class SplendorContextException(SplendorException):
    """Raised on OpenGL context errors (e.g. mixing EGL and GLFW)."""
    pass

class SplendorEmptyMeshException(SplendorException):
    """Raised when loading a mesh with no vertices."""
    pass
