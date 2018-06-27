import math
import numpy

def projection_matrix(
        horizontal_field_of_view,
        aspect_ratio,
        near_clip = 0.1,
        far_clip = 100):
    
    x_limit = near_clip * math.tan(horizontal_field_of_view * 0.5)
    y_limit = x_limit / aspect_ratio
    
    return numpy.array([
            [near_clip/x_limit, 0, 0, 0],
            [0, near_clip/y_limit, 0, 0],
            [0, 0, -(far_clip + near_clip) / (far_clip - near_clip),
                -2 * far_clip * near_clip / (far_clip - near_clip)],
            [0, 0, -1, 0]])
