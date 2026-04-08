import pyglet
from splendor.contexts.initialization import (
    initialization_state, register_context
)

def initialize():
    new_context = register_context('pyglet')
    if new_context:
        if x_authority is not None:
            os.environ['XAUTHORITY'] = x_authority
            os.environ['DISPLAY'] = display

class PygletPlatform:
    def __init__(self):
        pyglet.options['shadow_window'] = False
        try:
            pyglet.ib.x11.xlib.XInitThreads()
        except:
            pass
        
        self._window = PygletWindow(
            width=1,
            height=1,
            anti_alias=False,
            visible=False,
        )
    
    def activate(self):
        pass
    
    def deactivate(self):
        pass

class PygletWindow:
    def __init__(self,
        width,
        height,
        anti_alias=False,
        anti_alias_samples=8,
        visible=True,
    ):
        self._window_config = pyglet.gl.Config(
            sample_buffers=anti_alias,
            samples=anti_alias_samples,
            depth_size=24,
            double_buffer=True,
            major_version=3,
            minor_version=3,
        )
        self._window = pyglet.window.Window(
            config=self._window_config,
            visible=visible,
            resizable=False,
            width=width,
            height=height,
        )
    
    def activate(self):
        self._window.switch_to()
    
    def show_window(self):
        pass
    
    def hide_window(self):
        pass
    
    def set_active(self):
        pass
