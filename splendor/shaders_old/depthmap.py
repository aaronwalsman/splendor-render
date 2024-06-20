textured_depthmap_vertex_shader = '''#version 460 core

layout(location=0) in float vertex_depth;

uniform mat4 projection_matrix;
uniform mat4 model_pose;
uniform mat4 view_matrix;
uniform float radial_k1;
uniform float radial_k2;

uniform vec2 focal_length;
uniform int width;
uniform int height;

out vec2 fragment_uv;

void main() {
    mat4 vm = view_matrix * model_pose;
    mat4 pvm = projection_matrix * vm;
    
    int u_pixel = gl_VertexID % width;
    int v_pixel = gl_VertexID / width;
    
    float u = float(u_pixel) / width;
    float v = float(v_pixel) / height;
    
    float height_ratio = float(height) / width;
    
    fragment_uv.x = u;
    fragment_uv.y = v * height_ratio;
    
    float x = (u * 2.0 - 1.0) * vertex_depth / focal_length.x;
    float y = ((1. - v) * 2.0 - 1.0) * height_ratio * vertex_depth /
            focal_length.y;

    gl_Position = pvm * vec4(x, y, -vertex_depth, 1);
    
    // radial distortion
    float xx = gl_Position.x/gl_Position.z;
    float yy = gl_Position.y/gl_Position.z;
    float r2 = xx*xx + yy*yy;
    float r4 = r2*r2;
    float d = max(1.0 + r2 * radial_k1 + r4 * radial_k2, 0.1);
    gl_Position.x = gl_Position.x + gl_Position.x / d;
    gl_Position.y = gl_Position.y + gl_Position.y / d;
}
'''

textured_depthmap_fragment_shader = '''#version 460 core
layout(binding=0) uniform sampler2D texture_sampler;
in vec2 fragment_uv;
out vec3 color;

void main() {
    color = texture(texture_sampler, fragment_uv).rgb;
}
'''
