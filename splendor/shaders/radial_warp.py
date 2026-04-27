"""Radial distortion warp pass shader."""
radial_warp_vertex_shader = '''#version 330 core
out vec2 fragment_uv;

void main() {
    // Fullscreen quad via gl_VertexID, no VBO needed
    vec2 positions[4] = vec2[](
        vec2(-1, -1), vec2(-1, 1), vec2(1, 1), vec2(1, -1));
    vec2 uvs[4] = vec2[](
        vec2(0, 0), vec2(0, 1), vec2(1, 1), vec2(1, 0));
    gl_Position = vec4(positions[gl_VertexID], 0, 1);
    fragment_uv = uvs[gl_VertexID];
}
'''

radial_warp_fragment_shader = '''#version 330 core
in vec2 fragment_uv;
out vec4 color;

uniform sampler2D intermediate_sampler;
uniform sampler2D intermediate_depth_sampler;
uniform float radial_k1;
uniform float radial_k2;
// Focal lengths from the output camera projection matrix (proj[0][0], proj[1][1])
uniform float fx;
uniform float fy;
// Ratio of intermediate focal length to output focal length (< 1 = wider FOV)
uniform float fov_scale;

void main() {
    // Convert output pixel UV to NDC
    vec2 ndc = fragment_uv * 2.0 - 1.0;

    // Convert NDC to camera-space normalized coords using output focal lengths
    vec2 p = vec2(ndc.x / fx, ndc.y / fy);

    // Inverse distortion: given distorted p, find undistorted p_u
    // Forward model: p_d = p_u * (1 + k1*r_u^2 + k2*r_u^4)
    // First-order inverse approximation using distorted radius:
    float r2 = dot(p, p);
    float r4 = r2 * r2;
    float d = 1.0 + r2 * radial_k1 + r4 * radial_k2;
    vec2 p_u = p / d;

    // Convert undistorted camera-space coords back to intermediate NDC,
    // accounting for the intermediate render's (possibly wider) FOV
    float fx_inter = fx * fov_scale;
    float fy_inter = fy * fov_scale;
    vec2 ndc_u = vec2(p_u.x * fx_inter, p_u.y * fy_inter);

    // Convert to UV for texture sampling
    vec2 uv_u = ndc_u * 0.5 + 0.5;

    if (any(lessThan(uv_u, vec2(0.0))) || any(greaterThan(uv_u, vec2(1.0)))) {
        color = vec4(0.0, 0.0, 0.0, 1.0);
        gl_FragDepth = 1.0;
    } else {
        color = texture(intermediate_sampler, uv_u);
        gl_FragDepth = texture(intermediate_depth_sampler, uv_u).r;
    }
}
'''
