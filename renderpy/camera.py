import math
import random
import numpy

def projection_matrix(
        horizontal_field_of_view,
        aspect_ratio,
        near_clip = 0.05,
        far_clip = 50):
    
    x_limit = near_clip * math.tan(horizontal_field_of_view * 0.5)
    y_limit = x_limit / aspect_ratio
    
    return numpy.array([
            [near_clip/x_limit, 0, 0, 0],
            [0, near_clip/y_limit, 0, 0],
            [0, 0, -(far_clip + near_clip) / (far_clip - near_clip),
                -2 * far_clip * near_clip / (far_clip - near_clip)],
            [0, 0, -1, 0]])


def projection_matrix_from_intrinsics(
        intrinsics,
        image_resolution,
        near_clip = 0.05,
        far_clip = 50):
    
    '''
    OpenGlMatrixSpec ProjectionMatrixRDF_TopLeft(
            int w, int h,
            GLprecision fu, GLprecision fv, GLprecision u0, GLprecision v0,
            GLprecision zNear, GLprecision zFar )
    '''
    
    fu = intrinsics[0,0]
    fv = intrinsics[1,1]
    u0 = intrinsics[0,2]
    v0 = intrinsics[1,2]
    w = image_resolution[1]
    h = image_resolution[0]
    
    L = -(u0) * near_clip / fu
    R = +(w-u0) * near_clip / fu
    T = -(v0) * near_clip / fv
    B = +(h-v0) * near_clip / fv
    
    P = numpy.zeros((4,4))
    
    P[0,0] = 2 * near_clip / (R-L)
    P[1,1] = 2 * near_clip / (T-B)
    
    P[0,2] = (R+L)/(L-R)
    P[1,2] = (T+B)/(B-T)
    P[2,2] = (far_clip + near_clip) / (far_clip - near_clip)
    P[3,2] = 1.0
    
    P[2,3] = (2 * far_clip * near_clip)/(near_clip - far_clip)
    
    return P


def turntable_pose(
        distance, orientation, elevation, spin, lift=0):
    
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
    
    l = numpy.array([
            [1, 0, 0, 0],
            [0, 1, 0, lift],
            [0, 0, 1, 0],
            [0, 0, 0, 1]])
    
    return numpy.linalg.inv(
            numpy.dot(l, numpy.dot(o, numpy.dot(e, numpy.dot(d,s)))))


def sample_turntable(
        distance_extents,
        num_poses,
        initial_orientation_extents,
        orientation_spacing_range,
        elevation_extents,
        spin_extents,
        lift_extents):
    
    theta_step = math.pi * 2. / num_poses
    theta = random.random() * (
            initial_orientation_extents[1] - initial_orientation_extents[0]) + (
            initial_orientation_extents[0])
    
    poses = []
    for _ in range(num_poses):
        distance_range = distance_extents[1] - distance_extents[0]
        distance = random.random() * distance_range + distance_extents[0]
        
        theta += theta_step
        orientation = (theta + random.random() * orientation_spacing_range -
                orientation_spacing_range * 0.5)
        
        elevation_range = elevation_extents[1] - elevation_extents[0]
        elevation = random.random() * elevation_range + elevation_extents[0]
        
        spin_range = spin_extents[1] - spin_extents[0]
        spin = random.random() * spin_range + spin_extents[0]
        
        lift_range = lift_extents[1] - lift_extents[0]
        lift = random.random() * lift_range + lift_extents[0]
        
        poses.append(turntable_pose(distance, theta, elevation, spin, lift))
    
    return poses

def position_to_pixels(
        position,
        projection_matrix,
        camera_matrix,
        screen_resolution,
        flip_y = True):
    
    projected = numpy.dot(numpy.dot(projection_matrix, camera_matrix), position)
    projected_x = projected[0] / projected[3]
    projected_y = projected[1] / projected[3]
    
    if flip_y:
        projected_y *= -1
    
    x = int(round((projected_x + 1.) * 0.5 * screen_resolution[0]))
    y = int(round((projected_y + 1.) * 0.5 * screen_resolution[1]))
    
    return x, y
