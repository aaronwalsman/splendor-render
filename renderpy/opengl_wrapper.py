# This is a hacky workaround to use PyOpenGL on Mac, see here:
# https://stackoverflow.com/questions/63475461/unable-to-import-opengl-gl-in-python-on-macos
try:
    from OpenGL.GL import (
        glFlush,
        glFinish,
        glBindFramebuffer,
        GL_FRAMEBUFFER,
        glViewport,
        glEnable,
        glDisable,
        GL_MULTISAMPLE,
        glReadPixels,
    )# this fails in <=2020 versions of Python on OS X 11.x
except ImportError:
    print('Patching OpenGL for Newer Mac OS Versions')
    from ctypes import util
    orig_util_find_library = util.find_library
    def new_util_find_library( name ):
        res = orig_util_find_library( name )
        if res:
            return res
        return '/System/Library/Frameworks/'+name+'.framework/'+name
    util.find_library = new_util_find_library
    from OpenGL.GL import (
        glFlush,
        glFinish,
        glBindFramebuffer,
        GL_FRAMEBUFFER,
        glViewport,
        glEnable,
        glDisable,
        GL_MULTISAMPLE,
        glReadPixels,
    )# this fails in <=2020 versions of Python on OS X 11.x

import OpenGL.GLUT as GLUT
import OpenGL.platform as platform
