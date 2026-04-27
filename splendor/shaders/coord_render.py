"""Coordinate rendering shader — outputs world/object-space coordinates."""
from splendor.shaders.mesh import mesh_vertex_shader

coord_vertex_shader = f'''#version 330 core
#define COMPILE_COORD
{mesh_vertex_shader}'''

coord_fragment_shader = '''#version 330 core
in vec3 coord;
out vec3 color;
void main(){
    color = coord;
}
'''
