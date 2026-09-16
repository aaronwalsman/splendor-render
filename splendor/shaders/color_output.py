"""Display transform for color rendering only. Inputs are linear HDR + depth."""
from splendor.shaders.radial_warp import radial_warp_vertex_shader

color_output_vertex_shader = radial_warp_vertex_shader
color_output_fragment_shader = """#version 330 core
in vec2 fragment_uv;
out vec4 color;
uniform sampler2D intermediate_sampler;
uniform sampler2D intermediate_depth_sampler;
uniform float exposure;
uniform bool tone_map;
uniform bool encode_srgb;

void main() {
    vec4 source = texture(intermediate_sampler, fragment_uv);
    vec3 rgb = max(source.rgb * exp2(exposure), vec3(0.0));
    if (tone_map) rgb = rgb / (vec3(1.0) + rgb);
    if (encode_srgb) {
        bvec3 low = lessThanEqual(rgb, vec3(0.0031308));
        rgb = mix(1.055 * pow(rgb, vec3(1.0 / 2.4)) - 0.055,
                  12.92 * rgb, low);
    }
    color = vec4(rgb, source.a);
    gl_FragDepth = texture(intermediate_depth_sampler, fragment_uv).r;
}
"""
