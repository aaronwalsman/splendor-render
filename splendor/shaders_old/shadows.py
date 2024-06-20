from splendor.shaders.mesh import mesh_vertex_shader

depthmap_shadow_vertex_shader = f'''#version 460 core
#define COMPILE_SHADOW
{mesh_vertex_shader}
'''

depthmap_shadow_fragment_shader = '''#version 460 core

void main()
{}
'''
