import math
import numpy
import renderpy.pose_utils as pose_utils

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

#===============================================================================
# camera pose
#===============================================================================
def camera_pose_to_matrix(pose):
    
    # matrix input
    if len(pose) == 4 and len(pose[0]) == 4:
        return numpy.array(pose)
    
    # azimuth, elevation, spin, distance, x, y
    elif len(pose) == 6:
        return azimuthal_pose_to_matrix(pose)
    
    # unknown
    else:
        raise ValueError('camera pose should be a 4x4 matrix or 6 elements '
                '[azimuth, elevation, tilt, distance, shift_x, shift_y]')
        
def azimuthal_pose_to_matrix(pose):
    azimuth = pose[0]
    elevation = pose[1]
    tilt = pose[2]
    distance = pose[3]
    shift_x = pose[4]
    shift_y = pose[5]
    
    a = pose_utils.euler_y_matrix(azimuth)
    e = pose_utils.euler_x_matrix(elevation)
    t = pose_utils.euler_z_matrix(tilt)
    
    translate = pose_utils.translate_matrix([shift_x, shift_y, distance])
    
    m = numpy.linalg.inv(
            numpy.dot(a,
            numpy.dot(e,
            numpy.dot(t, translate))))
    
    return m

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
    model_camera_poses = sample_turntable(
            distance_extents = distance_extents,
            num_poses = num_poses,
            initial_orientation_extents = initial_orientation_extents,
            orientation_spacing_range =
                math.pi * 2. / (2 * num_poses),
            elevation_extents = elevation_extents,
            spin_extents = spin_extents,
            lift_extents = lift_extents)

    model_camera_poses = [numpy.dot(camera_pose, centroid_offset)
            for camera_pose in model_camera_poses]
    
    return model_camera_poses


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

def crop_projection_matrix(box, resolution, projection, batch=False):
    box_width = box[3] - box[1]
    box_height = box[2] - box[0]
    try:
        cropped_projection = projection.clone()
    except:
        cropped_projection = projection.copy()
    x_scale = resolution[1] / box_width
    y_scale = resolution[0] / box_height
    if batch:
        cropped_projection[:,0,0] *= x_scale
        cropped_projection[:,1,1] *= y_scale
    else:
        cropped_projection[0,0] *= x_scale
        cropped_projection[1,1] *= y_scale
    
    box_center_x = box_width * 0.5 + box[1]
    box_center_y = box_height * 0.5 + box[0]
    x_offset = (box_center_x - resolution[1]/2.) * 2 / box_width
    # this -1 scale is due to the inverse relationship between pixels
    # and normalized device coordinates
    y_offset = -(box_center_y - resolution[0]/2.) * 2 / box_height
    
    if batch:
        cropped_projection[:,0,2] = (
                cropped_projection[:,0,2] * x_scale + x_offset)
        cropped_projection[:,1,2] = (
                cropped_projection[:,1,2] * y_scale + y_offset)
    else:
        cropped_projection[0,2] = (
                cropped_projection[0,2] * x_scale + x_offset)
        cropped_projection[1,2] = (
                cropped_projection[1,2] * y_scale + y_offset)
    
    return cropped_projection


def position_to_pixels(
        position,
        projection_matrix,
        camera_matrix,
        screen_resolution,
        flip_y = True,
        round_output=True):
    
    if len(position) == 3:
        position = numpy.append(position, [1])
    
    projected = numpy.dot(numpy.dot(projection_matrix, camera_matrix), position)
    projected_x = projected[0] / projected[3]
    projected_y = projected[1] / projected[3]
    
    if flip_y:
        projected_y *= -1
    
    if round_output:
        if isinstance(projected_x, numpy.ndarray):
            x = numpy.round((projected_x + 1.) * 0.5 * screen_resolution[1])
        else:
            x = int(round((projected_x + 1.) * 0.5 * screen_resolution[1]))
        
        if isinstance(projected_y, numpy.ndarray):
            y = numpy.round((projected_y + 1.) * 0.5 * screen_resolution[0])
        else:
            y = int(round((projected_y + 1.) * 0.5 * screen_resolution[0]))
    
    else:
        x = projected_x * screen_resolution[1]
        y = projected_y * screen_resolution[0]
    
    return x, y

