"""Camera and projection utilities — perspective, orthographic, intrinsics, and view matrices."""
import math
import numpy
import splendor.pose as pose

# projection matrix utilities

def projection_matrix(
    horizontal_field_of_view,
    aspect_ratio,
    near_clip = 0.05,
    far_clip = 50,
):
    """Build a perspective projection matrix.

    Parameters
    ----------
    horizontal_field_of_view : float
        Horizontal field of view in radians.
    aspect_ratio : float
        Width divided by height.
    near_clip : float, optional
        Near clipping plane distance (default 0.05).
    far_clip : float, optional
        Far clipping plane distance (default 50).

    Returns
    -------
    numpy.ndarray, shape (4, 4)
        Perspective projection matrix.
    """
    # aspect ratio = x/y
    
    x_limit = near_clip * math.tan(horizontal_field_of_view * 0.5)
    y_limit = x_limit / aspect_ratio
    
    return numpy.array([
        [near_clip/x_limit, 0, 0, 0],
        [0, near_clip/y_limit, 0, 0],
        [0, 0, -(far_clip + near_clip) / (far_clip - near_clip),
            -2 * far_clip * near_clip / (far_clip - near_clip)],
        [0, 0, -1, 0]
    ])

def orthographic_matrix(
    l = -1,
    r = 1,
    b = -1,
    t = 1,
    n = 0.05,
    f = 50,
):
    """Build an orthographic projection matrix.

    Parameters
    ----------
    l : float, optional
        Left clipping plane (default -1).
    r : float, optional
        Right clipping plane (default 1).
    b : float, optional
        Bottom clipping plane (default -1).
    t : float, optional
        Top clipping plane (default 1).
    n : float, optional
        Near clipping plane (default 0.05).
    f : float, optional
        Far clipping plane (default 50).

    Returns
    -------
    numpy.ndarray, shape (4, 4)
        Orthographic projection matrix.
    """
    return numpy.array([
        [2./(r-l),        0,        0, -(r+l)/(r-l)],
        [       0, 2./(t-b),        0, -(t+b)/(t-b)],
        [       0,        0, -2/(f-n), -(f+n)/(f-n)],
        [       0,        0,        0,            1],
    ])

def projection_matrix_from_intrinsics(
    intrinsics,
    image_resolution,
    near_clip = 0.05,
    far_clip = 50,
    offset_x = 0,
    offset_y = 0,
):
    """Build an OpenGL projection matrix from camera intrinsics.

    Converts a standard computer-vision 3x3 intrinsics matrix (with fu, fv,
    u0, v0) into a 4x4 OpenGL-style perspective projection matrix.

    Parameters
    ----------
    intrinsics : numpy.ndarray, shape (3, 3)
        Camera intrinsics matrix with focal lengths at (0,0) and (1,1) and
        principal point at (0,2) and (1,2).
    image_resolution : tuple of int
        (height, width) of the target image.
    near_clip : float, optional
        Near clipping plane distance (default 0.05).
    far_clip : float, optional
        Far clipping plane distance (default 50).
    offset_x : float, optional
        Horizontal principal-point offset in pixels (default 0).
    offset_y : float, optional
        Vertical principal-point offset in pixels (default 0).

    Returns
    -------
    numpy.ndarray, shape (4, 4)
        Perspective projection matrix.
    """
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
    """Extract near and far clip distances from a projection matrix.

    Parameters
    ----------
    projection : numpy.ndarray, shape (4, 4)
        Projection matrix (perspective or orthographic).

    Returns
    -------
    near : float
        Near clipping plane distance.
    far : float
        Far clipping plane distance.
    """
    near =-((-projection[3,3] - projection[2,3]) /
            ( projection[2,2] + projection[3,2]))
    far = -(( projection[3,3] - projection[2,3]) /
            ( projection[2,2] - projection[3,2]))
    return near, far

def change_projection_aspect_ratio(
        projection_matrix,
        old_resolution,
        new_resolution):
    """Rescale a projection matrix for a different image resolution.

    Parameters
    ----------
    projection_matrix : numpy.ndarray, shape (4, 4)
        Original projection matrix.
    old_resolution : tuple of int
        (height, width) the matrix was built for.
    new_resolution : tuple of int
        (height, width) to rescale to.

    Returns
    -------
    numpy.ndarray, shape (4, 4)
        Scaled projection matrix.
    """
    y_scale = old_resolution[0] / new_resolution[0]
    x_scale = old_resolution[1] / new_resolution[1]
    scaled_projection_matrix = numpy.copy(projection_matrix)
    scaled_projection_matrix[0] *= x_scale
    scaled_projection_matrix[1] *= y_scale
    
    return scaled_projection_matrix

def crop_projection_matrix(box, resolution, projection):
    """Modify a projection matrix to render only a cropped region.

    Parameters
    ----------
    box : tuple of int
        Crop region as (min_y, min_x, max_y, max_x) in pixels.
    resolution : tuple of int
        (height, width) of the full image.
    projection : numpy.ndarray, shape (4, 4)
        Original projection matrix.

    Returns
    -------
    numpy.ndarray, shape (4, 4)
        Projection matrix that renders only the cropped region.
    """
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
    round_to_pixel=True,
):
    """Project a 3D point to 2D pixel coordinates.

    Parameters
    ----------
    position : array-like, shape (3,) or (4,)
        World-space 3D point. If length 3, a homogeneous 1 is appended.
    projection_matrix : numpy.ndarray, shape (4, 4)
        Projection matrix.
    camera_matrix : numpy.ndarray, shape (4, 4)
        View (world-to-camera) matrix.
    screen_resolution : tuple of int
        (height, width) of the target image.
    flip_y : bool, optional
        Flip the y-axis to match image coordinates (default True).
    round_to_pixel : bool, optional
        Floor coordinates to integer pixel indices (default True).

    Returns
    -------
    x : int or float
        Horizontal pixel coordinate.
    y : int or float
        Vertical pixel coordinate.
    """
    if len(position) == 3:
        position = numpy.append(position, [1])
    
    rx = screen_resolution[1]# - 1
    ry = screen_resolution[0]# - 1
    
    projected = projection_matrix @ camera_matrix @ position
    projected_x = projected[0] / projected[3]
    projected_y = projected[1] / projected[3]
    
    if flip_y:
        projected_y *= -1
    
    x = rx * (projected_x + 1.) * 0.5
    y = ry * (projected_y + 1.) * 0.5
    
    if round_to_pixel:
        x = math.floor(x)
        y = math.floor(y)
    
    return x, y

# camera matrix utilities

def view_matrix(parameters):
    """Build a view matrix (world-to-camera transform) from various formats.

    Accepts a 4x4 matrix (passed through), a dict of azimuthal parameters,
    6 elements ``[azimuth, elevation, tilt, distance, shift_x, shift_y]``,
    or 9 elements (adding ``center_x, center_y, center_z``). Angles are in
    radians.

    Parameters
    ----------
    parameters : dict, array-like (4x4), or array-like (6,) or (9,)
        Camera parameters in one of the supported formats.

    Returns
    -------
    numpy.ndarray, shape (4, 4)
        View matrix (world-to-camera transform).

    Raises
    ------
    ValueError
        If *parameters* does not match any recognized format.
    """
    # dict input
    if isinstance(parameters, dict):
        return numpy.linalg.inv(azimuthal_parameters_to_matrix(**parameters))
    
    # matrix input
    elif len(parameters) == 4 and len(parameters[0]) == 4:
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
            'center_x, center_y, center_z]'
        )
        
def direction_light_pose(direction, position=(0., 0., 0.)):
    """
    Build a 4x4 pose matrix for a directional light.

    The pose encodes both the orientation of the light (which way it shines)
    and an anchor position (used when computing a shadow frustum).

    Parameters
    ----------
    direction : array-like, shape (3,)
        The direction the light rays travel (from source toward the scene).
        Does not need to be normalized.
    position : array-like, shape (3,), default (0, 0, 0)
        World-space anchor point for the light.  Used as the shadow camera
        position when rendering shadow maps.

    Returns
    -------
    numpy.ndarray, shape (4, 4)
        Pose matrix (camera-to-world transform) whose -Z axis points in
        ``direction``.
    """
    direction = numpy.array(direction, dtype=float)
    direction = direction / numpy.linalg.norm(direction)

    # Build an orthonormal frame with -Z aligned to direction
    forward = direction  # -Z of the pose
    # Pick an up vector that isn't parallel to forward
    up = numpy.array([0., 1., 0.])
    if abs(numpy.dot(forward, up)) > 0.99:
        up = numpy.array([1., 0., 0.])
    right = numpy.cross(up, forward)
    right = right / numpy.linalg.norm(right)
    up = numpy.cross(forward, right)

    pose = numpy.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = forward   # +Z = backward = direction of rays
    pose[:3, 3] = numpy.array(position, dtype=float)
    return pose


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
    """Build a camera-to-world pose matrix from azimuthal orbit parameters.

    The camera orbits around a center point at the given distance, oriented
    by azimuth, elevation, and tilt angles, with optional lateral shifts.

    Parameters
    ----------
    azimuth : float, optional
        Azimuth angle in radians (rotation about Y axis, default 0).
    elevation : float, optional
        Elevation angle in radians (rotation about X axis, default 0).
    tilt : float, optional
        Tilt angle in radians (rotation about Z / optical axis, default 0).
    distance : float, optional
        Distance from the center point (default 0).
    shift_x : float, optional
        Horizontal shift (default 0).
    shift_y : float, optional
        Vertical shift (default 0).
    center_x : float, optional
        X coordinate of the orbit center (default 0).
    center_y : float, optional
        Y coordinate of the orbit center (default 0).
    center_z : float, optional
        Z coordinate of the orbit center (default 0).

    Returns
    -------
    numpy.ndarray, shape (4, 4)
        Camera-to-world pose matrix.
    """
    azimuth = pose.euler_y_matrix(azimuth)
    elevation = pose.euler_x_matrix(elevation)
    tilt = pose.euler_z_matrix(tilt)
    
    translate = pose.translate_matrix([shift_x, shift_y, distance])
    
    center = pose.translate_matrix((center_x, center_y, center_z))
    
    matrix = center @ azimuth @ elevation @ tilt @ translate
    
    return matrix

def framing_distance_for_bbox(bbox, projection_matrix, multiplier):
    """Compute the camera distance needed to frame a bounding box.

    Parameters
    ----------
    bbox : array-like, shape (2, 3)
        Bounding box as [min_corner, max_corner].
    projection_matrix : numpy.ndarray, shape (4, 4)
        Projection matrix used for rendering.
    multiplier : float
        Scale factor applied to the computed distance (>1 zooms out).

    Returns
    -------
    float
        Camera distance that frames the bounding box.
    """
    diagonal = numpy.array(bbox[1]) - numpy.array(bbox[0])
    radius = numpy.linalg.norm(diagonal) * 0.5
    return projection_matrix[0,0] * radius * 0.5 * multiplier

def frame_bbox(bbox, projection_matrix, multiplier,
        azimuth=0, elevation=0, tilt=0, shift_x=0, shift_y=0):
    """Build a view matrix that frames a bounding box from the given angles.

    Computes the distance needed to fit the bounding box in frame, then
    constructs a view matrix using azimuthal orbit parameters centered on
    the bounding box centroid.

    Parameters
    ----------
    bbox : array-like, shape (2, 3)
        Bounding box as [min_corner, max_corner].
    projection_matrix : numpy.ndarray, shape (4, 4)
        Projection matrix used for rendering.
    multiplier : float
        Scale factor for the framing distance (>1 zooms out).
    azimuth : float, optional
        Azimuth angle in radians (default 0).
    elevation : float, optional
        Elevation angle in radians (default 0).
    tilt : float, optional
        Tilt angle in radians (default 0).
    shift_x : float, optional
        Horizontal shift (default 0).
    shift_y : float, optional
        Vertical shift (default 0).

    Returns
    -------
    numpy.ndarray, shape (4, 4)
        View matrix (world-to-camera transform) that frames the bounding box.
    """
    diagonal = numpy.array(bbox[1]) - numpy.array(bbox[0])
    centroid = bbox[0] + diagonal * 0.5
    distance = framing_distance_for_bbox(bbox, projection_matrix, multiplier)
    return numpy.linalg.inv(azimuthal_parameters_to_matrix(
            azimuth, elevation, tilt, distance, shift_x, shift_y, *centroid))
