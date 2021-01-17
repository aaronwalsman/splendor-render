mesh_vertex_shader = '''
layout(location=0) in vec3 vertex_position;
layout(location=1) in vec3 vertex_normal;

#ifdef use_texture
layout(location=2) in vec2 vertex_uv;
#endif

#ifdef use_vertex_colors
layout(location=2) in vec3 vertex_color;
#endif

out vec4 fragment_position;
out vec4 fragment_normal;

#ifdef use_texture
out vec2 fragment_uv;
#endif

#ifdef use_vertex_colors
out vec3 fragment_color;
#endif

uniform mat4 projection_matrix;
uniform mat4 model_pose;
uniform mat4 camera_pose;

void main(){
    mat4 vm = camera_pose * model_pose;
    mat4 pvm = projection_matrix * vm;
    
    gl_Position = pvm * vec4(vertex_position,1);
    
    fragment_position = vm * vec4(vertex_position,1);
    fragment_normal = vm * vec4(vertex_normal,0);
    
    #ifdef use_texture
    fragment_uv.x = vertex_uv.x;
    fragment_uv.y =-vertex_uv.y;
    #endif
    
    #ifdef use_vertex_colors
    fragment_color = vertex_color;
    #endif
}
'''
