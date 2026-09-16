"""GGX prefiltered environment cubemaps for roughness-based reflections.

Bakes a fixed ladder of roughness levels (split-sum approximation, Karis 2013)
into a cubemap mip chain: level 0 is the sharp environment, level n-1 is
roughness 1.  The renderer samples with explicit LOD `rough * (n-1)`, so
material roughness maps to a baked angular spread instead of a screen-footprint
mip bias.  Requires a current GL context.  Replaces the retired Gaussian
blurry_mipmaps approach.
"""
import hashlib
import math
import os

from OpenGL import GL
from OpenGL.GL import shaders

import numpy

from splendor.frame_buffer import FrameBufferWrapper
from splendor.home import get_splendor_home
from splendor.shaders.background import background_vertex_shader
from splendor.image_light.panorama import face_view_matrices, face_order
import splendor.camera as camera

# Bump when the bake math changes; invalidates every cache entry.
PREFILTER_VERSION = 1
MAX_PREFILTER_LEVELS = 6
DEFAULT_SAMPLES = 1024

prefilter_fragment_shader = '''#version 330 core
in vec3 fragment_direction;
out vec4 color;

uniform samplerCube source_sampler;
uniform float roughness;
uniform float source_size;
uniform int sample_count;

const float PI = 3.14159265359;

float radical_inverse_vdc(uint bits){
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10;
}

vec3 importance_sample_ggx(vec2 xi, vec3 n, float alpha){
    float phi = 2.0 * PI * xi.x;
    float cos_theta = sqrt((1.0 - xi.y) / (1.0 + (alpha*alpha - 1.0) * xi.y));
    float sin_theta = sqrt(1.0 - cos_theta*cos_theta);
    vec3 h = vec3(cos(phi)*sin_theta, sin(phi)*sin_theta, cos_theta);
    vec3 up = abs(n.z) < 0.999 ? vec3(0.0,0.0,1.0) : vec3(1.0,0.0,0.0);
    vec3 tangent = normalize(cross(up, n));
    vec3 bitangent = cross(n, tangent);
    return tangent*h.x + bitangent*h.y + n*h.z;
}

void main(){
    vec3 n = normalize(fragment_direction);
    if(roughness <= 0.0){
        color = vec4(textureLod(source_sampler, n, 0.0).rgb, 1.0);
        return;
    }
    // GGX microfacet alpha is perceptual roughness squared; the ladder is
    // indexed by perceptual roughness, so square exactly once here.
    float alpha = roughness * roughness;
    vec3 v = n;  // N = V = R simplification (Karis split-sum)
    vec3 acc = vec3(0.0);
    float weight = 0.0;
    float omega_texel = 4.0 * PI / (6.0 * source_size * source_size);
    for(uint i = 0u; i < uint(sample_count); ++i){
        vec2 xi = vec2(float(i) / float(sample_count), radical_inverse_vdc(i));
        vec3 h = importance_sample_ggx(xi, n, alpha);
        vec3 l = normalize(2.0 * dot(v, h) * h - v);
        float n_dot_l = dot(n, l);
        if(n_dot_l > 0.0){
            // pdf-matched source mip keeps bright pixels from turning into
            // fireflies at practical sample counts
            float a2 = alpha * alpha;
            float n_dot_h = max(dot(n, h), 0.0);
            float denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
            float d = a2 / (PI * denom * denom);
            float h_dot_v = max(dot(h, v), 1e-4);
            float pdf = d * n_dot_h / (4.0 * h_dot_v) + 1e-4;
            float omega_sample = 1.0 / (float(sample_count) * pdf);
            float lod = max(0.5 * log2(omega_sample / omega_texel), 0.0);
            acc += textureLod(source_sampler, l, lod).rgb * n_dot_l;
            weight += n_dot_l;
        }
    }
    color = vec4(acc / max(weight, 1e-6), 1.0);
}
'''

def num_prefilter_levels(base_size):
    """Roughness ladder length for a cubemap with this base face size.

    Fixed at MAX_PREFILTER_LEVELS so the roughness->LOD mapping does not
    depend on environment resolution; only clamped when the source is too
    small to hold that many meaningful levels.
    """
    return max(min(MAX_PREFILTER_LEVELS, int(math.log2(base_size)) + 1), 1)

def prefilter_cubemap(source_texture, source_size, samples=DEFAULT_SAMPLES):
    """GGX-prefilter a loaded cubemap texture into a roughness level ladder.

    Requires a current GL context.  source_texture must be a complete
    mip-mapped GL cubemap (its own format handles any sRGB decode).  Returns a
    list of float32 (h, 6h, 3) linear strips, level 0 sharp at source_size,
    each subsequent level half size, baked at roughness level/(n-1).
    """
    n_levels = num_prefilter_levels(source_size)
    program = shaders.compileProgram(
        shaders.compileShader(background_vertex_shader, GL.GL_VERTEX_SHADER),
        shaders.compileShader(prefilter_fragment_shader, GL.GL_FRAGMENT_SHADER))
    GL.glUseProgram(program)
    GL.glActiveTexture(GL.GL_TEXTURE0)
    GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, source_texture)
    GL.glUniform1i(GL.glGetUniformLocation(program, 'source_sampler'), 0)
    GL.glUniform1f(
        GL.glGetUniformLocation(program, 'source_size'), float(source_size))
    GL.glUniform1i(
        GL.glGetUniformLocation(program, 'sample_count'), int(samples))
    projection = camera.projection_matrix(math.radians(90), 1.0, 0.01, 1.0)
    GL.glUniformMatrix4fv(
        GL.glGetUniformLocation(program, 'projection_matrix'),
        1, GL.GL_TRUE, projection.astype(numpy.float32))
    view_location = GL.glGetUniformLocation(program, 'view_matrix')
    roughness_location = GL.glGetUniformLocation(program, 'roughness')

    empty_vao = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(empty_vao)
    depth_test = GL.glIsEnabled(GL.GL_DEPTH_TEST)
    GL.glDisable(GL.GL_DEPTH_TEST)

    levels = []
    try:
        for level in range(n_levels):
            size = source_size >> level
            roughness = level / (n_levels - 1) if n_levels > 1 else 0.
            GL.glUniform1f(roughness_location, roughness)
            frame_buffer = FrameBufferWrapper(
                size, size, anti_alias=False,
                color_format=GL.GL_RGBA32F)
            try:
                frame_buffer.enable()
                faces = []
                for face in face_order:
                    GL.glUniformMatrix4fv(
                        view_location, 1, GL.GL_TRUE,
                        face_view_matrices[face].astype(numpy.float32))
                    GL.glClear(GL.GL_COLOR_BUFFER_BIT)
                    GL.glDrawArrays(GL.GL_TRIANGLE_FAN, 0, 4)
                    faces.append(frame_buffer.read_pixels())
                levels.append(
                    numpy.concatenate(faces, axis=1).astype(numpy.float32))
            finally:
                frame_buffer.close()
    finally:
        if depth_test:
            GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glBindVertexArray(0)
        GL.glDeleteVertexArrays(1, [empty_vao])
        GL.glUseProgram(0)
        GL.glDeleteProgram(program)
    return levels

def prefilter_cache_key(source_strip, color_space, samples):
    """Content-addressed cache key: source bytes + bake parameters + version."""
    digest = hashlib.sha256()
    digest.update(numpy.ascontiguousarray(source_strip).tobytes())
    digest.update(repr((
        source_strip.shape, str(source_strip.dtype), color_space,
        int(samples), PREFILTER_VERSION)).encode())
    return digest.hexdigest()

def cached_prefilter_cubemap(
    source_strip,
    color_space,
    source_texture,
    samples=DEFAULT_SAMPLES,
):
    """prefilter_cubemap with a content-addressed cache in the splendor home.

    Levels are stored float16 on disk and returned float32.
    """
    key = prefilter_cache_key(source_strip, color_space, samples)
    cache_dir = os.path.join(get_splendor_home(), 'prefilter')
    cache_path = os.path.join(cache_dir, key + '.npz')
    if os.path.exists(cache_path):
        with numpy.load(cache_path) as data:
            return [data['level_%i' % i].astype(numpy.float32)
                    for i in range(int(data['n_levels']))]
    levels = prefilter_cubemap(
        source_texture, source_strip.shape[0], samples=samples)
    os.makedirs(cache_dir, exist_ok=True)
    numpy.savez_compressed(
        cache_path,
        n_levels=len(levels),
        **{'level_%i' % i: level.astype(numpy.float16)
           for i, level in enumerate(levels)})
    return levels
