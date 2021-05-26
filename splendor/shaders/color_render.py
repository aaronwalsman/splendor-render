from splendor.shaders.mesh import mesh_vertex_shader
from splendor.shaders.lighting_model import lighting_model_fragment_shader

# textured + textured material properties
textured_material_properties_vertex_shader = f'''#version 460 core
#define COMPILE_TEXTURE
#define COMPILE_TEXTURED_MATERIAL_PROPERTIES
{mesh_vertex_shader}'''
textured_material_properties_fragment_shader = f'''#version 460 core
#define COMPILE_TEXTURE
#define COMPILE_TEXTURED_MATERIAL_PROPERTIES
{lighting_model_fragment_shader}'''

# textured
textured_vertex_shader = f'''#version 460 core
#define COMPILE_TEXTURE
{mesh_vertex_shader}'''
textured_fragment_shader = f'''#version 460 core
#define COMPILE_TEXTURE
{lighting_model_fragment_shader}'''

# vertex_color
vertex_color_vertex_shader = f'''#version 460 core
#define COMPILE_VERTEX_COLORS
{mesh_vertex_shader}'''
vertex_color_fragment_shader = f'''#version 460 core
#define COMPILE_VERTEX_COLORS
{lighting_model_fragment_shader}'''

# flat_color
flat_color_vertex_shader = f'''#version 460 core
#define COMPILE_FLAT_COLOR
{mesh_vertex_shader}'''
flat_color_fragment_shader = f'''#version 460 core
#define COMPILE_FLAT_COLOR
{lighting_model_fragment_shader}'''
