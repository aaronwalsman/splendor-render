"""GLFW mouse/keyboard orbit camera controller."""
import glfw

import numpy as np

import splendor.camera as camera

class InteractiveCameraGLFW:
    """Orbit camera controller for GLFW windows.

    Left-drag orbits, right-drag (or shift+left-drag) pans, scroll zooms.
    Orbits around the depth under the cursor.

    Parameters
    ----------
    up : array-like (3,), optional
        World up vector.  When given, orbiting becomes a turntable: the
        camera's spherical coordinates (azimuth/elevation) about this axis
        are updated directly, so the horizon stays level and no roll can
        accumulate.  Default None preserves the free trackball orbit.
    """
    def __init__(self, window, renderer, camera_name, up=None):
        self.window = window
        self.renderer = renderer
        self.camera_name = camera_name
        self.up = None
        if up is not None:
            self.up = np.array(up, dtype=np.float64)
            self.up /= np.linalg.norm(self.up)
        self.mouse_down_button = None
        self.mouse_position = (0,0)
        self.mouse_click_depth = None
        self.shift_down = False
    
    def get_raw_mouse_pixel_position(self, window):
        w, h = glfw.get_window_size(window)
        raw_x, raw_y = glfw.get_cursor_pos(window)
        return round(raw_x), round(raw_y)
    
    def get_mouse_pixel_position(self, window, raw_xy=None):
        w, h = glfw.get_window_size(window)
        #fbw, fbh = glfw.get_framebuffer_size(window)
        fbw, fbh = self.window.framebuffer_size()
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
                projection=self.renderer.get_camera_projection(self.camera_name),
            )
            fbw, fbh = glfw.get_framebuffer_size(window)
            row = min(max(fbh - 1 - y, 0), fbh - 1)
            col = min(max(x, 0), fbw - 1)
            z = float(depth[row, col])   # depth image is (H, W, 1)

            color = self.window.read_pixels()
            r,g,b,a = color[row, col]

            self.mouse_down_button = button
            self.mouse_position = (x,y)
            self.mouse_click_depth = z
        
        if action == glfw.RELEASE:
            self.mouse_down_button = None
    
    def mouse_move(self, window, raw_x, raw_y):
        x, y = self.get_mouse_pixel_position(window, (raw_x, raw_y))
        if self.mouse_down_button is not None and self.mouse_click_depth is None:
            self.mouse_position = (x, y)
            return
        w, h = glfw.get_framebuffer_size(window)
        mx, my = self.mouse_position
        dx = (x - mx) / w
        dy = (y - my) / h
        view_matrix = self.renderer.get_camera_view_matrix(self.camera_name)
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
        
        if orbit and self.up is not None:
            # Turntable: update the camera's spherical coordinates about the
            # fixed up axis, pivoting on the depth under the click.  The
            # camera always re-aims at the pivot with a level horizon, so no
            # roll can ever accumulate.
            u = self.up
            pivot_point = (camera_pose @ np.array(
                [0., 0., -self.mouse_click_depth, 1.]))[:3]
            offset = camera_pose[:3,3] - pivot_point
            r = np.linalg.norm(offset)
            # In-plane azimuth basis (any fixed pair orthogonal to up).
            ref = np.array([1.,0.,0.]) if abs(u[0]) < 0.9 else np.array([0.,1.,0.])
            b1 = ref - (ref @ u) * u
            b1 /= np.linalg.norm(b1)
            b2 = np.cross(u, b1)
            az = np.arctan2(offset @ b2, offset @ b1)
            el = np.arcsin(np.clip((offset @ u) / r, -1., 1.))
            az -= dx*2
            el = np.clip(el + dy*2, -np.pi/2 + 0.02, np.pi/2 - 0.02)
            offset = r * (np.cos(el) * (np.cos(az)*b1 + np.sin(az)*b2)
                          + np.sin(el) * u)
            position = pivot_point + offset
            z_cam = offset / r                      # camera looks down -Z
            x_cam = np.cross(u, z_cam)
            x_cam /= np.linalg.norm(x_cam)
            y_cam = np.cross(z_cam, x_cam)
            camera_pose = np.eye(4)
            camera_pose[:3,0] = x_cam
            camera_pose[:3,1] = y_cam
            camera_pose[:3,2] = z_cam
            camera_pose[:3,3] = position
            view_matrix = np.linalg.inv(camera_pose)
            self.renderer.set_camera_view_matrix(self.camera_name, view_matrix)

        elif orbit:
            inverse_pivot = np.eye(4)
            inverse_pivot[2,3] = self.mouse_click_depth
            pivot = np.eye(4)
            pivot[2,3] = -self.mouse_click_depth

            parameters = [dx*2, dy*2, 0, 0, 0, 0]
            pose_offset = camera.azimuthal_parameters_to_matrix(*parameters)
            pose_offset = pivot @ np.linalg.inv(pose_offset) @ inverse_pivot
            camera_pose = np.dot(camera_pose, pose_offset)
            view_matrix = np.linalg.inv(camera_pose)
            self.renderer.set_camera_view_matrix(self.camera_name, view_matrix)

        if pan:
            x_direction = view_matrix[0,0:3]
            y_direction = view_matrix[1,0:3]
            x_offset = -x_direction * dx * self.mouse_click_depth
            y_offset = y_direction * dy * self.mouse_click_depth
            camera_pose[0:3,3] += x_offset + y_offset
            view_matrix = np.linalg.inv(camera_pose)
            self.renderer.set_camera_view_matrix(self.camera_name, view_matrix)

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
            projection=self.renderer.get_camera_projection(self.camera_name),
        )
        fbw, fbh = glfw.get_framebuffer_size(window)
        row = min(max(fbh - 1 - y, 0), fbh - 1)
        col = min(max(x, 0), fbw - 1)
        z = float(depth[row, col])   # depth image is (H, W, 1)
        self.mouse_click_depth = z

        view_matrix = self.renderer.get_camera_view_matrix(self.camera_name)
        z_direction = view_matrix[2,:3]
        distance = y_offset * -0.1 * self.mouse_click_depth
        z_offset = z_direction * distance

        camera_pose = np.linalg.inv(view_matrix)
        camera_pose[:3,3] += z_offset
        view_matrix = np.linalg.inv(camera_pose)
        self.renderer.set_camera_view_matrix(self.camera_name, view_matrix)
