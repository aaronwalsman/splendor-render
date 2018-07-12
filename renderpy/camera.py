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

def turntable_pose(
        distance, orientation, elevation, spin):
    
    d = numpy.array([
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,distance],
            [0,0,0,1]])
    
    ce = math.cos(elevation)
    se = math.sin(elevation)
    e = numpy.array([
            [1, 0, 0, 0],
            [0, ce, -se, 0],
            [0, se, ce, 0],
            [0, 0, 0, 1]])
    
    co = math.cos(orientation)
    so = math.sin(orientation)
    o = numpy.array([
            [co, 0, -so, 0],
            [0, 1, 0, 0],
            [so, 0, co, 0],
            [0, 0, 0, 1]])
    
    cs = math.cos(spin)
    ss = math.sin(spin)
    s = numpy.array([
            [cs, -ss, 0, 0],
            [ss, cs, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]])
    
    return numpy.linalg.inv(
            numpy.dot(o, numpy.dot(e, numpy.dot(d,s))))

