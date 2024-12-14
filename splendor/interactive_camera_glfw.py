import glfw

class InteractiveCameraGLFW:
    def __init__(self, window, renderer):
        self.window = window
        self.renderer = renderer
        
    def mouse_callback(self, window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
            x, y = glfw.get_cursor_pos(window)
            w, h = glfw.get_window_size(window)
            fbw, fbh = glfw.get_framebuffer_size(window)
            dx, dy = round(x*(fbw/w)), round(y*(fbh/h))
            #dx, dy = round(x), round(y)
            
            depth = self.window.read_pixels(
                read_depth=True,
                projection=self.renderer.get_projection(),
            )
            z = depth[self.window.height-dy, dx]
            
            color = self.window.read_pixels()
            r,g,b,a = color[self.window.height-dy, dx]
            
            print(f'click {x}, {y}, {z}, {r} {g} {b} {a}')
