import math
import numpy


def cubemap_strip_to_sh(strip):
    """
    Compute L0+L1+L2 spherical harmonic irradiance coefficients from a cubemap.

    Parameters
    ----------
    strip : (H, 6H, 3) uint8 array
        Cubemap horizontal strip, faces in order px,nx,py,ny,pz,nz.

    Returns
    -------
    coeffs : (9, 3) float32 array
        SH irradiance coefficients pre-multiplied by Lambertian zonal harmonics
        (A_0=pi, A_1=2pi/3, A_2=pi/4).  Evaluate irradiance at unit normal n:
            irr = sum_i coeffs[i] * Y_i(n)
        where Y_i are the real spherical harmonic basis functions.
    """
    H = strip.shape[0]
    assert strip.shape == (H, 6 * H, 3), (
        f'Expected ({H}, {6 * H}, 3), got {strip.shape}')

    # Camera-to-world rotation for each cubemap face (transposed view matrices).
    # Each matrix transforms a direction from face-local camera space to world.
    # Face-local: right=+x, up=+y, forward=-z (standard OpenGL camera convention).
    face_rotations = numpy.array([
        [[ 0,  0, -1], [ 0, -1,  0], [-1,  0,  0]],  # px
        [[ 0,  0,  1], [ 0, -1,  0], [ 1,  0,  0]],  # nx
        [[ 1,  0,  0], [ 0,  0, -1], [ 0,  1,  0]],  # py
        [[ 1,  0,  0], [ 0,  0,  1], [ 0, -1,  0]],  # ny
        [[ 1,  0,  0], [ 0, -1,  0], [ 0,  0, -1]],  # pz
        [[-1,  0,  0], [ 0, -1,  0], [ 0,  0,  1]],  # nz
    ], dtype=numpy.float32)  # (6, 3, 3)

    # Pixel centers in face-local space (camera looks along -z)
    idx = numpy.arange(H, dtype=numpy.float32)
    jj, ii = numpy.meshgrid(idx, idx)   # jj=col, ii=row, each (H, H)
    x_c = jj - H / 2.0 + 0.5           # (H, H)
    y_c = ii - H / 2.0 + 0.5           # (H, H)
    z_c = numpy.full_like(x_c, -H / 2.0)

    local_dirs = numpy.stack([x_c, y_c, z_c], axis=-1)  # (H, H, 3)

    # Rotate to world space: world_dirs[f,h,w] = face_rotations[f] @ local_dirs[h,w]
    world_dirs = numpy.einsum('fij,hwj->fhwi', face_rotations, local_dirs)  # (6,H,H,3)

    # Normalize
    norms = numpy.linalg.norm(world_dirs, axis=-1, keepdims=True)  # (6,H,H,1)
    dirs = world_dirs / norms  # (6, H, H, 3)

    # Solid angle per pixel: dΩ = (H/2) / |r|³  where |r| = norm of local_dir
    r_sq = x_c ** 2 + y_c ** 2 + (H / 2.0) ** 2   # (H, H)
    solid_angle = (H / 2.0) / r_sq ** 1.5            # (H, H)
    solid_angle = numpy.broadcast_to(solid_angle, (6, H, H))  # (6, H, H)

    # Colors from strip in [0, 1]: (6, H, H, 3)
    colors = numpy.stack([
        strip[:, f * H:(f + 1) * H, :].astype(numpy.float32) / 255.0
        for f in range(6)
    ], axis=0)

    # Real spherical harmonic basis functions (l=0,1,2) at each direction
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]  # (6, H, H) each
    sh_basis = numpy.stack([
        numpy.full_like(x, 0.282095),      # Y_00
        0.488603 * y,                       # Y_1-1
        0.488603 * z,                       # Y_10
        0.488603 * x,                       # Y_11
        1.092548 * x * y,                  # Y_2-2
        1.092548 * y * z,                  # Y_2-1
        0.315392 * (3.0 * z * z - 1.0),   # Y_20
        1.092548 * x * z,                  # Y_21
        0.546274 * (x * x - y * y),       # Y_22
    ], axis=-1)  # (6, H, H, 9)

    # Integrate: coeffs[b, c] = Σ_{f,h,w} L(ω)[c] * Y_b(ω) * dΩ
    coeffs = numpy.einsum(
        'fhwc,fhwb,fhw->bc', colors, sh_basis, solid_angle)  # (9, 3)

    # Pre-multiply by Lambertian zonal harmonics so the shader is just a dot product
    lambertian = numpy.array([
        math.pi,              # l=0: A_0 = pi
        2.0 * math.pi / 3.0, # l=1: A_1 = 2pi/3
        2.0 * math.pi / 3.0,
        2.0 * math.pi / 3.0,
        math.pi / 4.0,        # l=2: A_2 = pi/4
        math.pi / 4.0,
        math.pi / 4.0,
        math.pi / 4.0,
        math.pi / 4.0,
    ], dtype=numpy.float32)

    coeffs *= lambertian[:, numpy.newaxis]

    return coeffs.astype(numpy.float32)
