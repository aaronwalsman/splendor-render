import math
import numpy

def rodrigues(p, axis, angle):
    c = math.cos(angle)
    s = math.sin(angle)
    return (p * c +
            numpy.cross(p, axis) * s +
            axis * numpy.dot(axis, p) * (1-c))

def rodrigues_matrix(axis, angle):
    x = rodrigues(numpy.array([1.,0.,0.]), axis, angle)
    y = rodrigues(numpy.array([0.,1.,0.]), axis, angle)
    z = rodrigues(numpy.array([0.,0.,1.]), axis, angle)
    m = numpy.eye(4)
    m[:3,0] = x
    m[:3,1] = y
    m[:3,2] = z
    return m

def euler_x_matrix(angle):
    c = math.cos(angle)
    s = math.sin(angle)
    m = numpy.eye(4)
    m[1,1] = c
    m[1,2] = -s
    m[2,1] = s
    m[2,2] = c
    return m

def euler_y_matrix(angle):
    c = math.cos(angle)
    s = math.sin(angle)
    m = numpy.eye(4)
    m[0,0] = c
    m[0,2] = s
    m[2,0] = -s
    m[2,2] = c
    return m

def euler_z_matrix(angle):
    c = math.cos(angle)
    s = math.sin(angle)
    m = numpy.eye(4)
    m[0,0] = c
    m[0,1] = -s
    m[1,0] = s
    m[1,1] = c
    return m

def translate_matrix(translate):
    m = numpy.eye(4)
    if len(translate) == 3:
        m[:3,3] = translate
    elif len(translate) == 4:
        m[:,3] = translate
    return m

def scale_matrix(sx, sy, sz):
    m = numpy.eye(4)
    m[0,0] = sx
    m[1,1] = sy
    m[2,2] = sz
    return m
