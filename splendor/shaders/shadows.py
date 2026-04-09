from splendor.shaders.mesh import mesh_vertex_shader

depthmap_shadow_vertex_shader = f'''#version 330 core
#define COMPILE_SHADOW
{mesh_vertex_shader}
'''

depthmap_shadow_fragment_shader = '''#version 330 core

void main()
{}
'''
