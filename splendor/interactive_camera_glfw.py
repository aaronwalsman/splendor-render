import glfw

import numpy as np

import splendor.camera as camera

class InteractiveCameraGLFW:
    def __init__(self, window, renderer):
        self.window = window
        self.renderer = renderer
        self.mouse_down_button = None
        self.mouse_position = (0,0)
        self.mouse_click_depth = None
        self.shift_down = False
    
    def get_mouse_pixel_position(self, window, raw_xy=None):
        w, h = glfw.get_window_size(window)
        fbw, fbh = glfw.get_framebuffer_size(window)
        if raw_xy is None:
            raw_x, raw_y = glfw.get_cursor_pos(window)
        else:
            raw_x, raw_y = raw_xy
        x, y = round(raw_x*(fbw/w)), round(raw_y*(fbh/h))
        
        return x, y
    
    def mouse_callback(self, window, button, action, mods):
        if button in (
            glfw.MOUSE_BUTTON_LEFT,
            glfw.MOUSE_BUTTON_RIGHT,
        ) and action == glfw.PRESS:
            x, y = self.get_mouse_pixel_position(window)
            depth = self.window.read_pixels(
                read_depth=True,
                projection=self.renderer.get_projection(),
            )
            z = depth[self.window.height-y, x]
            
            color = self.window.read_pixels()
            r,g,b,a = color[self.window.height-y, x]
            
            self.mouse_down_button = glfw.MOUSE_BUTTON_LEFT
            self.mouse_position = (x,y)
            self.mouse_click_depth = z
        
        if action == glfw.RELEASE:
            self.mouse_down_button = None
    
    def mouse_move(self, window, raw_x, raw_y):
        x, y = self.get_mouse_pixel_position(window, (raw_x, raw_y))
        w, h = glfw.get_framebuffer_size(window)
        mx, my = self.mouse_position
        dx = (x - mx) / w
        dy = (y - my) / h
        view_matrix = self.renderer.get_view_matrix()
        camera_pose = np.linalg.inv(view_matrix)
        
        orbit = (
            self.mouse_down_button == glfw.MOUSE_BUTTON_LEFT and
            not self.shift_down
        )
        pan = (
            self.mouse_down_button == glfw.MOUSE_BUTTON_RIGHT or (
            self.mouse_down_button == glfw.MOUSE_BUTTON_LEFT and
            self.shift_down)
        )
        
        if orbit:
            inverse_pivot = np.eye(4)
            inverse_pivot[2,3] = self.mouse_click_depth
            pivot = np.eye(4)
            pivot[2,3] = -self.mouse_click_depth

            parameters = [dx*2, dy*2, 0, 0, 0, 0]
            pose_offset = camera.azimuthal_parameters_to_matrix(*parameters)
            pose_offset = pivot @ np.linalg.inv(pose_offset) @ inverse_pivot
            camera_pose = np.dot(camera_pose, pose_offset)
            view_matrix = np.linalg.inv(camera_pose)
            self.renderer.set_view_matrix(view_matrix)
    
        if pan:
            x_direction = view_matrix[0,0:3]
            y_direction = view_matrix[1,0:3]
            x_offset = -x_direction * dx * self.mouse_click_depth
            y_offset = y_direction * dy * self.mouse_click_depth
            camera_pose[0:3,3] += x_offset + y_offset
            view_matrix = np.linalg.inv(camera_pose)
            self.renderer.set_view_matrix(view_matrix)

        self.mouse_position = (x,y)
    
    def key_callback(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            if key == glfw.KEY_LEFT_SHIFT or key == glfw.KEY_RIGHT_SHIFT:
                self.shift_down = True
        elif action == glfw.RELEASE:
            if key == glfw.KEY_LEFT_SHIFT or key == glfw.KEY_RIGHT_SHIFT:
                self.shift_down = False
    
    def scroll_callback(self, window, x_offset, y_offset):
        x, y = self.get_mouse_pixel_position(window)
        depth = self.window.read_pixels(
            read_depth=True,
            projection=self.renderer.get_projection(),
        )
        z = depth[self.window.height-y, x]
        self.mouse_click_depth = z
        
        view_matrix = self.renderer.get_view_matrix()
        z_direction = view_matrix[2,:3]
        distance = y_offset * -0.1 * self.mouse_click_depth
        z_offset = z_direction * distance
        
        camera_pose = np.linalg.inv(view_matrix)
        camera_pose[:3,3] += z_offset
        view_matrix = np.linalg.inv(camera_pose)
        self.renderer.set_view_matrix(view_matrix)
