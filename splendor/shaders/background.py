from splendor.shaders.skybox import skybox_fn

background_vertex_shader = '''#version 330 core

#define FAR 1-(1e-4)

uniform mat4 projection_matrix;
uniform mat4 view_matrix;

out vec3 fragment_direction;
out vec2 fragment_uv;

void main(){
    mat4 inv_vp = inverse(projection_matrix * view_matrix);
    vec4 p0 = inverse(view_matrix) * vec4(0,0,0,1);

    if(gl_VertexID == 0){
        gl_Position = vec4(-1, -1, FAR, 1);
        fragment_uv = vec2(0, 0);
    }
    else if(gl_VertexID == 1){
        gl_Position = vec4(-1, 1, FAR, 1);
        fragment_uv = vec2(0, 1);
    }
    else if(gl_VertexID == 2){
        gl_Position = vec4(1, 1, FAR, 1);
        fragment_uv = vec2(1, 1);
    }
    else if(gl_VertexID == 3){
        gl_Position = vec4(1, -1, FAR, 1);
        fragment_uv = vec2(1, 0);
    }

    vec4 p1 = inv_vp * gl_Position;
    p1 /= p1.w;
    fragment_direction = vec3(p1 - p0);
}
'''

background_fragment_shader = '''#version 330 core
in vec3 fragment_direction;
out vec3 color;

uniform float blur;
uniform samplerCube cubemap_sampler;
uniform mat4 offset_matrix;''' + f'''
{skybox_fn}''' + '''
void main(){
    vec4 offset_direction = offset_matrix *
            vec4(fragment_direction, 0.);
    color = vec3(skybox_texture(cubemap_sampler, offset_direction, blur));
}
'''
