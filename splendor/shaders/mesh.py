# when using mesh_vertex_shader, define one and only one of the following:
# COMPILE_TEXTURE
# COMPILE_VERTEX_COLORS
# COMPILE_FLAT_COLORS
# COMPILE_MASK
# COMPILE_COORD
mesh_vertex_shader = '''
layout(location=0) in vec3 vertex_position;

#if defined(COMPILE_TEXTURE) || \
    defined(COMPILE_VERTEX_COLORS) || \
    defined(COMPILE_FLAT_COLOR)
layout(location=1) in vec3 vertex_normal;
#endif

#ifdef COMPILE_TEXTURE
layout(location=2) in vec2 vertex_uv;
#endif

#ifdef COMPILE_VERTEX_COLORS
layout(location=2) in vec3 vertex_color;
#endif

#if defined(COMPILE_TEXTURE) || \
    defined(COMPILE_VERTEX_COLORS) || \
    defined(COMPILE_FLAT_COLOR)
out vec4 fragment_position;
out vec4 fragment_normal;
#endif

#ifdef COMPILE_TEXTURE
out vec2 fragment_uv;
#endif

#ifdef COMPILE_VERTEX_COLORS
out vec3 fragment_color;
#endif

#ifdef COMPILE_COORD
out vec3 coord;
#endif

uniform mat4 projection_matrix;
uniform mat4 model_pose;
uniform mat4 view_matrix;

#ifdef COMPILE_FLAT_COLOR
uniform vec3 box_min;
uniform vec3 box_max;
#endif

void main(){
    mat4 vm = view_matrix * model_pose;
    mat4 pvm = projection_matrix * vm;
    
    gl_Position = pvm * vec4(vertex_position,1);
    
    #if defined(COMPILE_TEXTURE) || \
        defined(COMPILE_VERTEX_COLORS) || \
        defined(COMPILE_FLAT_COLOR)
    fragment_position = vm * vec4(vertex_position,1);
    fragment_normal = vm * vec4(vertex_normal,0);
    #endif
    
    #ifdef COMPILE_TEXTURE
    fragment_uv.x = vertex_uv.x;
    fragment_uv.y =-vertex_uv.y;
    #endif
    
    #ifdef COMPILE_VERTEX_COLORS
    fragment_color = vertex_color;
    #endif
    
    #ifdef COMIPLE_COORD
    coord = (vertex_position - box_min) / (box_max - box_min);
    #endif
}
'''
