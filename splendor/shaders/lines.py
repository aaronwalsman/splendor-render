lines_vertex_shader = """
#version 330 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;

uniform mat4 mvp_matrix;

out vec3 vertex_color;

void main() {
    gl_Position = mvp_matrix * vec4(position, 1.0);
    vertex_color = color;
}
"""

lines_fragment_shader = """
#version 330 core

in vec3 vertex_color;
out vec4 frag_color;

void main() {
    frag_color = vec4(vertex_color, 1.0);
}
"""
