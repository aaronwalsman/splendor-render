"""Spherical harmonic diffuse irradiance — precompute, evaluate, and debug."""
import math
import numpy


# Camera-to-world rotation for each cubemap face (transposed view matrices).
# Face-local: right=+x, up=+y, forward=-z (standard OpenGL camera convention).
_FACE_ROTATIONS = numpy.array([
    [[ 0,  0, -1], [ 0, -1,  0], [-1,  0,  0]],  # px
    [[ 0,  0,  1], [ 0, -1,  0], [ 1,  0,  0]],  # nx
    [[ 1,  0,  0], [ 0,  0, -1], [ 0,  1,  0]],  # py
    [[ 1,  0,  0], [ 0,  0,  1], [ 0, -1,  0]],  # ny
    [[ 1,  0,  0], [ 0, -1,  0], [ 0,  0, -1]],  # pz
    [[-1,  0,  0], [ 0, -1,  0], [ 0,  0,  1]],  # nz
], dtype=numpy.float32)  # (6, 3, 3)

# SH basis normalization constants (baked into coefficients so shader is simple)
_SH_NORMS = numpy.array([
    0.282095,                                          # Y_00
    0.488603, 0.488603, 0.488603,                      # Y_1{-1,0,1}
    1.092548, 1.092548, 0.315392, 1.092548, 0.546274,  # Y_2{-2..2}
], dtype=numpy.float32)

# Lambertian zonal harmonics
_LAMBERTIAN = numpy.array([
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


def _cubemap_directions_and_solid_angles(H):
    """
    Compute world-space directions and solid angles for each pixel of an
    H x 6H cubemap strip.

    Returns (dirs, solid_angle): dirs is (6,H,H,3) unit vectors,
    solid_angle is (6,H,H).
    """
    idx = numpy.arange(H, dtype=numpy.float32)
    jj, ii = numpy.meshgrid(idx, idx)
    x_c = jj - H / 2.0 + 0.5
    y_c = ii - H / 2.0 + 0.5
    z_c = numpy.full_like(x_c, -H / 2.0)

    local_dirs = numpy.stack([x_c, y_c, z_c], axis=-1)  # (H, H, 3)
    world_dirs = numpy.einsum(
        'fij,hwj->fhwi', _FACE_ROTATIONS, local_dirs)   # (6,H,H,3)
    norms = numpy.linalg.norm(world_dirs, axis=-1, keepdims=True)
    dirs = world_dirs / norms  # (6, H, H, 3)

    r_sq = x_c ** 2 + y_c ** 2 + (H / 2.0) ** 2
    solid_angle = (H / 2.0) / r_sq ** 1.5
    solid_angle = numpy.broadcast_to(solid_angle, (6, H, H))

    return dirs, solid_angle


def _sh_basis(dirs):
    """Evaluate the 9 real SH basis functions at each direction. Returns (*, 9)."""
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    return numpy.stack([
        numpy.full_like(x, 0.282095),      # Y_00
        0.488603 * y,                       # Y_1-1
        0.488603 * z,                       # Y_10
        0.488603 * x,                       # Y_11
        1.092548 * x * y,                  # Y_2-2
        1.092548 * y * z,                  # Y_2-1
        0.315392 * (3.0 * z * z - 1.0),   # Y_20
        1.092548 * x * z,                  # Y_21
        0.546274 * (x * x - y * y),       # Y_22
    ], axis=-1)


def evaluate_sh(coeffs, directions):
    """
    Evaluate SH irradiance at the given unit directions.

    Parameters
    ----------
    coeffs : (9, 3) array
        SH irradiance coefficients (as returned by cubemap_strip_to_sh).
    directions : (..., 3) array
        Unit-length direction vectors.

    Returns
    -------
    (..., 3) array of irradiance values (clamped to >= 0).
    """
    coeffs = numpy.asarray(coeffs, dtype=numpy.float32)
    directions = numpy.asarray(directions, dtype=numpy.float32)
    nx, ny, nz = directions[..., 0], directions[..., 1], directions[..., 2]
    irr = (coeffs[0]
         + coeffs[1] * ny[..., None]
         + coeffs[2] * nz[..., None]
         + coeffs[3] * nx[..., None]
         + coeffs[4] * (nx * ny)[..., None]
         + coeffs[5] * (ny * nz)[..., None]
         + coeffs[6] * (3 * nz * nz - 1)[..., None]
         + coeffs[7] * (nx * nz)[..., None]
         + coeffs[8] * (nx * nx - ny * ny)[..., None])
    return numpy.maximum(irr, 0.0)


def sh_to_cubemap_strip(coeffs, H=64):
    """
    Reconstruct a diffuse cubemap strip from SH coefficients (for debugging).

    Parameters
    ----------
    coeffs : (9, 3) array
    H : int
        Face resolution.

    Returns
    -------
    (H, 6H, 3) uint8 array.
    """
    dirs, _ = _cubemap_directions_and_solid_angles(H)  # (6,H,H,3)
    irr = evaluate_sh(coeffs, dirs)  # (6, H, H, 3)
    irr = numpy.clip(irr, 0, 1)
    strip = numpy.concatenate(
        [(irr[f] * 255).astype(numpy.uint8) for f in range(6)], axis=1)
    return strip


def cubemap_strip_to_sh(strip, input_gamma=1.5, auto_normalize=True,
                        normalize_target=1.0):
    """
    Compute L0+L1+L2 spherical harmonic irradiance coefficients from a cubemap.

    Parameters
    ----------
    strip : (H, 6H, 3) uint8 array
        Cubemap horizontal strip, faces in order px,nx,py,ny,pz,nz.
    input_gamma : float, default=1.5
        Gamma applied to pixel values before integration.  Values > 1
        expand contrast between bright and dark areas, partially
        compensating for the dynamic range lost in LDR images.
        Set to 1.0 for linear (e.g. when input is already HDR/linear).
    auto_normalize : bool, default=True
        If True, scale the coefficients so that the peak irradiance across
        all directions and channels equals normalize_target.  This provides
        auto-exposure so results are in a display-friendly range regardless
        of input brightness.
    normalize_target : float, default=0.7
        Target for the peak irradiance when auto_normalize is True.
        0.7 leaves headroom so the brightest direction doesn't clip.

    Returns
    -------
    coeffs : (9, 3) float32 array
        SH irradiance coefficients.  Evaluate irradiance at unit normal n:
            irr = c[0] + c[1]*ny + c[2]*nz + c[3]*nx + c[4]*nx*ny + ...
    """
    H = strip.shape[0]
    assert strip.shape == (H, 6 * H, 3), (
        f'Expected ({H}, {6 * H}, 3), got {strip.shape}')

    dirs, solid_angle = _cubemap_directions_and_solid_angles(H)

    # Colors from strip in [0, 1], with input gamma for LDR contrast boost
    colors = numpy.stack([
        strip[:, f * H:(f + 1) * H, :].astype(numpy.float32) / 255.0
        for f in range(6)
    ], axis=0)  # (6, H, H, 3)
    if input_gamma != 1.0:
        colors = colors ** input_gamma

    # Evaluate SH basis at each direction
    basis = _sh_basis(dirs)  # (6, H, H, 9)

    # Integrate: coeffs[b, c] = Σ L(ω)[c] * Y_b(ω) * dΩ
    coeffs = numpy.einsum(
        'fhwc,fhwb,fhw->bc', colors, basis, solid_angle)  # (9, 3)

    # Pre-multiply by Lambertian zonal harmonics + SH normalization constants
    # so the shader polynomial needs no extra constants at evaluation time.
    coeffs *= _LAMBERTIAN[:, numpy.newaxis]
    coeffs *= _SH_NORMS[:, numpy.newaxis]

    # Normalize so uniform white environment → 1.0
    coeffs /= math.pi

    # Auto-normalize: scale so peak irradiance = normalize_target
    if auto_normalize:
        # Evaluate at a dense grid of directions to find peak irradiance
        probe_dirs, _ = _cubemap_directions_and_solid_angles(32)
        probe_irr = evaluate_sh(coeffs, probe_dirs)  # (6, 32, 32, 3)
        peak = probe_irr.max()
        if peak > 0:
            coeffs *= normalize_target / peak

    return coeffs.astype(numpy.float32)


def compute_shadow_color(coeffs):
    """
    Compute the global minimum irradiance across all directions — the
    darkest the environment gets.  Used as the shadow color for IBL shadows.

    Parameters
    ----------
    coeffs : (9, 3) array
        SH irradiance coefficients.

    Returns
    -------
    (3,) float32 array — RGB shadow color.
    """
    probe_dirs, _ = _cubemap_directions_and_solid_angles(32)
    probe_irr = evaluate_sh(coeffs, probe_dirs)  # (6, 32, 32, 3)
    # Find the direction with the lowest luminance
    luminance = (probe_irr[..., 0] * 0.2126
               + probe_irr[..., 1] * 0.7152
               + probe_irr[..., 2] * 0.0722)
    min_idx = numpy.unravel_index(luminance.argmin(), luminance.shape)
    return probe_irr[min_idx].astype(numpy.float32)
