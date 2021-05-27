import math
import numpy
import splendor.pose as pose

#===============================================================================
# projection
#===============================================================================
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

def orthographic_matrix(
        l = -1,
        r = 1,
        b = -1,
        t = 1,
        n = 0.05,
        f = 50):
    
    return numpy.array([
            [2./(r-l),        0,        0, -(r+l)/(r-l)],
            [       0, 2./(t-b),        0, -(t+b)/(t-b)],
            [       0,        0, -2/(f-n), -(f+n)/(f-n)],
            [       0,        0,        0,            1]])

def projection_matrix_from_intrinsics(
        intrinsics,
        image_resolution,   
        near_clip = 0.05,
        far_clip = 50,
        offset_x = 0,
        offset_y = 0):
    
    fu = intrinsics[0,0]
    fv = intrinsics[1,1]
    u0 = image_resolution[1] - intrinsics[0,2] + offset_x
    v0 = image_resolution[0] - intrinsics[1,2] + offset_y
    w = image_resolution[1]
    h = image_resolution[0]
    
    L = -(u0) * near_clip / fu
    R = +(w-u0) * near_clip / fu
    B = -(v0) * near_clip / fv
    T = +(h-v0) * near_clip / fv
    
    P = numpy.zeros((4,4))
    
    P[0,0] = 2 * near_clip / (R-L)
    P[1,1] = 2 * near_clip / (T-B)
    
    P[0,2] = (R+L)/(L-R)
    P[1,2] = (T+B)/(B-T)
    P[2,2] = -(far_clip + near_clip) / (far_clip - near_clip)
    P[3,2] = -1.0
    
    P[2,3] = -(2 * far_clip * near_clip)/(far_clip - near_clip)
    
    return P

def clip_from_projection(projection):
    near =-((-projection[3,3] - projection[2,3]) /
            ( projection[2,2] + projection[3,2]))
    far = -(( projection[3,3] - projection[2,3]) /
            ( projection[2,2] - projection[3,2]))
    return near, far

def change_projection_aspect_ratio(
        projection_matrix,
        old_resolution,
        new_resolution):
    y_scale = old_resolution[0] / new_resolution[0]
    x_scale = old_resolution[1] / new_resolution[1]
    scaled_projection_matrix = numpy.copy(projection_matrix)
    scaled_projection_matrix[0] *= x_scale
    scaled_projection_matrix[1] *= y_scale
    
    return scaled_projection_matrix

def crop_projection_matrix(box, resolution, projection):
    box_width = box[3] - box[1]
    box_height = box[2] - box[0]
    cropped_projection = projection.copy()
    
    x_scale = resolution[1] / box_width
    y_scale = resolution[0] / box_height
    cropped_projection[0,0] *= x_scale
    cropped_projection[1,1] *= y_scale
    
    box_center_x = box_width * 0.5 + box[1]
    box_center_y = box_height * 0.5 + box[0]
    x_offset = (box_center_x - resolution[1]/2.) * 2 / box_width
    # this -1 scale is due to the inverse relationship between pixels
    # and normalized device coordinates
    y_offset = -(box_center_y - resolution[0]/2.) * 2 / box_height
    
    cropped_projection[0,2] = (
            cropped_projection[0,2] * x_scale + x_offset)
    cropped_projection[1,2] = (
            cropped_projection[1,2] * y_scale + y_offset)
    
    return cropped_projection

def project(
        position,
        projection_matrix,
        camera_matrix,
        screen_resolution,
        flip_y = True,
        round_to_pixel=True):
    
    if len(position) == 3:
        position = numpy.append(position, [1])
    
    rx = screen_resolution[1] - 1
    ry = screen_resolution[0] - 1
    
    projected = numpy.dot(numpy.dot(projection_matrix, camera_matrix), position)
    projected_x = projected[0] / projected[3]
    projected_y = projected[1] / projected[3]
    
    if flip_y:
        projected_y *= -1
    
    x = rx * (projected_x + 1.) * 0.5
    y = ry * (projected_y + 1.) * 0.5
    
    if round_to_pixel:
        x = round(x)
        y = round(y)
    
    return x, y


#===============================================================================
# camera matrix
#===============================================================================
def view_matrix(parameters):
    
    # dict input
    if isinstance(parameters, dict):
        return numpy.linalg.inv(azimuthal_parameters_to_matrix(**parameters))
    
    # matrix input
    if len(parameters) == 4 and len(parameters[0]) == 4:
        return numpy.array(parameters)
    
    # azimuth
    elif len(parameters) == 6 or len(parameters) == 9:
        return numpy.linalg.inv(azimuthal_parameters_to_matrix(*parameters))
    
    # unknown
    else:
        raise ValueError('camera parameters should be a 4x4 matrix '
            'or a dictionary with named azimuthal argments '
            'or 6 elements '
            '[azimuth, elevation, tilt, distance, shift_x, shift_y]'
            'or 9 elements '
            '[azimuth, elevation, tilt, distance, shift_x, shift_y, '
            'center_x, center_y, center_z]')
        
def azimuthal_parameters_to_matrix(
    azimuth=0,
    elevation=0,
    tilt=0,
    distance=0,
    shift_x=0,
    shift_y=0,
    center_x=0,
    center_y=0,
    center_z=0,
):
    
    azimuth = pose.euler_y_matrix(azimuth)
    elevation = pose.euler_x_matrix(elevation)
    tilt = pose.euler_z_matrix(tilt)
    
    translate = pose.translate_matrix([shift_x, shift_y, distance])
    
    center = pose.translate_matrix((center_x, center_y, center_z))
    
    #m = numpy.linalg.inv(
    #        numpy.dot(c,
    #        numpy.dot(a,
    #        numpy.dot(e,
    #        numpy.dot(t, translate)))))
    matrix = center @ azimuth @ elevation @ tilt @ translate
    
    return matrix

def framing_distance_for_bbox(bbox, projection_matrix, multiplier):
    diagonal = numpy.array(bbox[1]) - numpy.array(bbox[0])
    radius = numpy.linalg.norm(diagonal) * 0.5
    return projection_matrix[0,0] * radius * 0.5 * multiplier

def frame_bbox(bbox, projection_matrix, multiplier,
        azimuth=0, elevation=0, tilt=0, shift_x=0, shift_y=0):
    diagonal = numpy.array(bbox[1]) - numpy.array(bbox[0])
    centroid = bbox[0] + diagonal * 0.5
    distance = framing_distance_for_bbox(bbox, projection_matrix, multiplier)
    return numpy.linalg.inv(azimuthal_parameters_to_matrix(
            azimuth, elevation, tilt, distance, shift_x, shift_y, *centroid))

'''
# moved to camera_utils.py in detection_utils
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


def get_framing_distance_for_mesh(
        mesh,
        mesh_transform,
        projection_matrix):
    
    v_min = numpy.min(mesh['vertices'], axis=0)
    v_max = numpy.max(mesh['vertices'], axis=0)
    v_offset = v_max - v_min
    v_centroid = v_offset * 0.5 + v_min
    v_centroid = numpy.dot(mesh_transform, numpy.append(v_centroid, 1.0))[:3]
    radius_3d = numpy.linalg.norm(v_offset) * 0.5

    size_based_distance = projection_matrix[0,0] * radius_3d
    
    return size_based_distance, v_centroid


def sample_uniform_mesh_viewing_angles(
        mesh,
        mesh_transform,
        projection_matrix,
        num_poses,
        distance_scale_extents):
    
    size_based_distance, v_centroid = get_framing_distance_for_mesh(
            mesh, mesh_transform, projection_matrix)
    
    distance_extents = (
            size_based_distance * distance_scale_extents[0],
            size_based_distance * distance_scale_extents[1])
    
    centroid_offset_a = numpy.eye(4)
    centroid_offset_a[:3,3] -= v_centroid
    
    centroid_offset_b = numpy.eye(4)
    centroid_offset_b[:3,3] += v_centroid


def sample_mesh_turntable(
        mesh,
        mesh_transform,
        projection_matrix,
        num_poses,
        distance_scale_extents,
        initial_orientation_extents,
        orientation_spacing_range,
        elevation_extents,
        spin_extents,
        lift_extents):
    
    size_based_distance, v_centroid = get_framing_distance_for_mesh(
            mesh, mesh_transform, projection_matrix)
    
    distance_extents = (
            size_based_distance * distance_scale_extents[0],
            size_based_distance * distance_scale_extents[1])

    centroid_offset = numpy.eye(4)
    centroid_offset[:3,3] -= v_centroid
    model_camera_matrices = sample_turntable(
            distance_extents = distance_extents,
            num_poses = num_poses,
            initial_orientation_extents = initial_orientation_extents,
            orientation_spacing_range =
                math.pi * 2. / (2 * num_poses),
            elevation_extents = elevation_extents,
            spin_extents = spin_extents,
            lift_extents = lift_extents)

    model_camera_matrices = [numpy.dot(camera_matrix, centroid_offset)
            for camera_matrix in model_camera_matrices]
    
    return model_camera_matrices


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
        #distance_range = distance_extents[1] - distance_extents[0]
        #distance = random.random() * distance_range + distance_extents[0]
        distance = random.uniform(*distance_extents)
        
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
'''
