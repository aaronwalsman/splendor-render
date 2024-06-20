from splendor.shaders.mesh import mesh_vertex_shader

coord_vertex_shader = f'''#version 460 core
#define COMPILE_COORD
{mesh_vertex_shader}'''

coord_fragment_shader = '''#version 460 core
in vec3 coord;
out vec3 color;
void main(){
    color = coord;
}
'''
