from renderpy.shaders.mesh import mesh_vertex_shader

mask_vertex_shader = '''#version 460 core
#define COMPILE_MASK''' + mesh_vertex_shader

mask_fragment_shader = '''#version 460 core
out vec3 color;
uniform vec3 mask_color;

void main(){
    color = mask_color;
}
'''
