"""Split-sum BRDF integration lookup table (DFG / environment BRDF).

The second half of the split-sum specular IBL approximation (Karis 2013):
for each (N dot V, perceptual roughness) pair, the scale and bias applied to
F0 so that `prefiltered * (F0 * scale + bias)` integrates the GGX BRDF over
the environment.  Environment-independent; computed once in numpy and cached
in the splendor home.
"""
import os

import numpy

from splendor.home import get_splendor_home

# Bump when the integration math changes; invalidates the cache.
DFG_VERSION = 1
DEFAULT_SIZE = 64
DEFAULT_SAMPLES = 1024

def _hammersley(n):
    """(n, 2) low-discrepancy sample positions in [0, 1)^2."""
    i = numpy.arange(n, dtype=numpy.uint32)
    bits = i.copy()
    bits = (bits << numpy.uint32(16)) | (bits >> numpy.uint32(16))
    bits = ((bits & numpy.uint32(0x55555555)) << numpy.uint32(1)) | \
           ((bits & numpy.uint32(0xAAAAAAAA)) >> numpy.uint32(1))
    bits = ((bits & numpy.uint32(0x33333333)) << numpy.uint32(2)) | \
           ((bits & numpy.uint32(0xCCCCCCCC)) >> numpy.uint32(2))
    bits = ((bits & numpy.uint32(0x0F0F0F0F)) << numpy.uint32(4)) | \
           ((bits & numpy.uint32(0xF0F0F0F0)) >> numpy.uint32(4))
    bits = ((bits & numpy.uint32(0x00FF00FF)) << numpy.uint32(8)) | \
           ((bits & numpy.uint32(0xFF00FF00)) >> numpy.uint32(8))
    return numpy.stack(
        [i / n, bits * numpy.float32(2.3283064365386963e-10)], axis=-1)

def dfg_lookup_table(size=DEFAULT_SIZE, samples=DEFAULT_SAMPLES):
    """Integrate the (size, size, 2) float32 DFG table.

    Axis 0 (rows, v texture coordinate) is perceptual roughness, axis 1
    (columns, u) is N dot V; channels are the F0 scale and bias.  Matches the
    k = alpha/2 image-based-light geometry term used in shaders/pbr.py.
    """
    texel = (numpy.arange(size, dtype=numpy.float32) + 0.5) / size
    n_dot_v = numpy.clip(texel, 1e-4, 1.0)[None, :, None]      # (1, size, 1)
    roughness = texel[:, None, None]                            # (size, 1, 1)
    alpha = roughness * roughness

    xi = _hammersley(samples).astype(numpy.float32)             # (samples, 2)
    phi = 2.0 * numpy.pi * xi[:, 0]
    # GGX half-vector importance sampling in tangent space (N = +z)
    cos_theta = numpy.sqrt(
        (1.0 - xi[None, None, :, 1])
        / (1.0 + (alpha * alpha - 1.0) * xi[None, None, :, 1]))
    sin_theta = numpy.sqrt(numpy.clip(1.0 - cos_theta ** 2, 0.0, 1.0))
    h_x = numpy.cos(phi)[None, None, :] * sin_theta
    h_y = numpy.sin(phi)[None, None, :] * sin_theta
    h_z = cos_theta

    v_x = numpy.sqrt(numpy.clip(1.0 - n_dot_v ** 2, 0.0, 1.0))
    v_z = n_dot_v
    v_dot_h = v_x * h_x + v_z * h_z
    l_z = 2.0 * v_dot_h * h_z - v_z

    n_dot_l = numpy.clip(l_z, 0.0, 1.0)
    n_dot_h = numpy.clip(h_z, 0.0, 1.0)
    v_dot_h = numpy.clip(v_dot_h, 0.0, 1.0)

    # Smith Schlick-Beckmann geometry with the IBL k = alpha / 2 convention
    k = alpha / 2.0
    g_v = v_z / (v_z * (1.0 - k) + k)
    g_l = n_dot_l / (n_dot_l * (1.0 - k) + k)
    g = g_v * g_l
    g_vis = numpy.where(
        n_dot_l > 0.0,
        g * v_dot_h / numpy.maximum(n_dot_h * v_z, 1e-6),
        0.0)
    fresnel = (1.0 - v_dot_h) ** 5

    scale = numpy.mean((1.0 - fresnel) * g_vis, axis=2)
    bias = numpy.mean(fresnel * g_vis, axis=2)
    return numpy.stack([scale, bias], axis=-1).astype(numpy.float32)

def cached_dfg_lookup_table(size=DEFAULT_SIZE, samples=DEFAULT_SAMPLES):
    """dfg_lookup_table with an npz cache in the splendor home."""
    cache_dir = os.path.join(get_splendor_home(), 'prefilter')
    cache_path = os.path.join(
        cache_dir, 'dfg_v%i_%i_%i.npz' % (DFG_VERSION, size, samples))
    if os.path.exists(cache_path):
        with numpy.load(cache_path) as data:
            return data['dfg'].astype(numpy.float32)
    table = dfg_lookup_table(size, samples)
    os.makedirs(cache_dir, exist_ok=True)
    numpy.savez_compressed(cache_path, dfg=table)
    return table
