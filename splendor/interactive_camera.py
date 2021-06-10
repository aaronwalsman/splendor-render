import numpy
from splendor import camera

class InteractiveCamera(object):
    def __init__(self, window, renderer):
        self.window = window
        self.renderer = renderer
        self.button = -1
        self.mouse_click_position = (0,0)
        self.mouse_click_depth = 0
    
    def mouse_button(self, button, button_state, x, y):
        if button in (0,2,3,4):
            if button_state == 0:
                depth = self.window.read_pixels(
                        read_depth = True,
                        projection = self.renderer.get_projection())
                self.button = button
                self.mouse_click_position = (x,y)
                self.mouse_click_depth = depth[self.window.height-y,x]
                
                near, far = camera.clip_from_projection(
                    self.renderer.get_projection())
                min_depth = numpy.min(depth)
                self.mouse_click_depth = min(
                    min_depth*5, self.mouse_click_depth)
                
                if button in (3,4):
                    view_matrix = self.renderer.get_view_matrix()
                    z_direction = view_matrix[2,0:3]
                    
                    distance = 0.1 * self.mouse_click_depth
                    z_offset = z_direction * distance
                    if button == 3:
                        z_offset *= -1.
                    
                    camera_pose = numpy.linalg.inv(view_matrix)
                    camera_pose[0:3,3] += z_offset
                    view_matrix = numpy.linalg.inv(camera_pose)
                    self.renderer.set_view_matrix(view_matrix)
            
            else:
                self.button = -1
    
    def mouse_move(self, x, y):
        if self.button == 0:
            # orbit
            delta_x = (
                    x - self.mouse_click_position[0])/self.window.width
            delta_y = (
                    y - self.mouse_click_position[1])/self.window.height
            view_matrix = self.renderer.get_view_matrix()
            camera_pose = numpy.linalg.inv(view_matrix)
            
            inverse_pivot = numpy.eye(4)
            inverse_pivot[2,3] = self.mouse_click_depth
            pivot = numpy.eye(4)
            pivot[2,3] = -self.mouse_click_depth
            
            parameters = [delta_x*2, delta_y*2, 0, 0, 0, 0]
            pose_offset = camera.azimuthal_parameters_to_matrix(*parameters)
            pose_offset = pivot @ numpy.linalg.inv(pose_offset) @ inverse_pivot
            camera_pose = numpy.dot(camera_pose, pose_offset)
            view_matrix = numpy.linalg.inv(camera_pose)
            self.renderer.set_view_matrix(view_matrix)
            
            self.mouse_click_position = (x,y)
        
        if self.button == 2:
            # pan
            delta_x = (
                    x - self.mouse_click_position[0])/self.window.width
            delta_y = (
                    y - self.mouse_click_position[1])/self.window.height
            view_matrix = self.renderer.get_view_matrix()
            x_direction = view_matrix[0,0:3]
            y_direction = view_matrix[1,0:3]
            
            x_offset = -x_direction * delta_x * self.mouse_click_depth
            y_offset = y_direction * delta_y * self.mouse_click_depth
            
            camera_pose = numpy.linalg.inv(view_matrix)
            camera_pose[0:3,3] += x_offset + y_offset
            view_matrix = numpy.linalg.inv(camera_pose)
            self.renderer.set_view_matrix(view_matrix)
            
            self.mouse_click_position = (x,y)

