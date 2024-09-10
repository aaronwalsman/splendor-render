skybox_fn = '''
vec4 skybox_texture(samplerCube sampler, vec3 v){
    return texture(sampler, vec3(-v.x, v.y, v.z));
}

vec4 skybox_texture(samplerCube sampler, vec4 v){
    return texture(sampler, vec3(-v.x, v.y, v.z));
}

vec4 skybox_texture(samplerCube sampler, vec3 v, float level){
    return texture(sampler, vec3(-v.x, v.y, v.z), level);
}

vec4 skybox_texture(samplerCube sampler, vec4 v, float level){
    return texture(sampler, vec3(-v.x, v.y, v.z), level);
}
'''

panorama_to_cube_fragment_shader = '''#version 460 core
#define M_PI 3.1415926535897932384626433832795

in vec3 fragment_direction;
out vec3 color;

layout(binding=0) uniform sampler2D texture_sampler;

void main(){
    vec3 target_direction = vec3(0,0,1);
    vec2 uv;
    vec3 direction_n = normalize(vec3(fragment_direction));
    uv.y = (-asin(direction_n.y) + M_PI * 0.5) / M_PI;
    direction_n.y = 0;
    direction_n = normalize(direction_n);
    uv.x = atan(direction_n.x, direction_n.z) / (M_PI * 2);
    color = texture(texture_sampler, uv).rgb;
}
'''
