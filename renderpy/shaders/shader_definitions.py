from renderpy.shaders.utils import phong_fn, softish_step_fn, intensity_fn
from renderpy.shaders.skybox import skybox_fn, panorama_to_cube_fragment_shader
from renderpy.shaders.image_light import image_light_diffuse_fn
from renderpy.shaders.mesh import mesh_vertex_shader
from renderpy.shaders.color_render import color_fragment_shader
from renderpy.shaders.background import (
        background_verex_shader,
        background_2D_fragment_shader,
        background_fragment_shader)
from panorama_to_cube import panorama_to_cube_fragment_shader

textured_vertex_shader = '''#version 460 core
#define use_texture''' + mesh_vertex_shader

textured_fragment_shader = '''#version 460 core
#define use_texture''' + color_fragment_shader

vertex_color_vertex_shader = '''#version 460 core
#define use_vertex_colors''' + mesh_vertex_shader

vertex_color_fragment_shader = '''#version 460 core
#define use_vertex_colors''' + color_fragment_shader

reflection_to_diffuse_fragment_shader = '''#version 460 core
#define NUM_SAMPLES 512
in vec3 fragment_direction;
in vec2 fragment_uv;
out vec3 color;

uniform float brightness;
uniform float contrast;
uniform float color_scale;
uniform samplerCube reflect_sampler;
uniform vec3 sphere_samples[NUM_SAMPLES];
uniform int num_importance_samples;
uniform float importance_sample_gain;
uniform float random_sample_gain;

void main(){
    vec3 importance_color = vec3(0,0,0);
    vec3 random_color = vec3(0,0,0);
    vec3 fragment_direction_n = normalize(fragment_direction);
    for(int i = 0; i < NUM_SAMPLES; ++i){
        float d = dot(fragment_direction_n, sphere_samples[i]);
        vec3 flipped_sample = sphere_samples[i] * sign(d);
        vec3 sample_color = vec3(texture(reflect_sampler, flipped_sample));
        
        // brightness
        sample_color += brightness;
        // contrast
        sample_color = (sample_color + 0.5) * contrast - 0.5;
        
        if(i >= num_importance_samples){ // || d < 0.){
            random_color += sample_color * abs(d);
        }
        else{
            if(d > 0.){
                importance_color += sample_color * abs(d);
            }
        }
    }
    importance_color /= num_importance_samples;
    random_color /= (NUM_SAMPLES - num_importance_samples);
    
    color = (importance_sample_gain * importance_color +
            random_sample_gain * random_color) /
            (importance_sample_gain + random_sample_gain);
    color *= color_scale;
}
'''

mask_vertex_shader = '''#version 460 core

layout(location=0) in vec3 vertex_position;

out vec4 fragment_position;

uniform mat4 projection_matrix;
uniform mat4 model_pose;
uniform mat4 camera_pose;

void main(){
    
    mat4 pvm = projection_matrix * camera_pose * model_pose;
    
    gl_Position = pvm * vec4(vertex_position,1);
}
'''

mask_fragment_shader = '''#version 460 core

out vec3 color;

uniform vec3 mask_color;

void main(){
    color = mask_color;
}
'''

coord_vertex_shader = '''#version 460 core

layout(location=0) in vec3 vertex_position;

out vec3 coord;

uniform mat4 projection_matrix;
uniform mat4 model_pose;
uniform mat4 camera_pose;

uniform vec3 box_min;
uniform vec3 box_max;

void main(){
    mat4 pvm = projection_matrix * camera_pose * model_pose;
    
    gl_Position = pvm * vec4(vertex_position, 1);
    coord = (vertex_position - box_min) / (box_max - box_min);
}
'''

coord_fragment_shader = '''#version 460 core

in vec3 coord;

out vec3 color;

void main(){
    color = coord;
}
'''

textured_depthmap_vertex_shader = '''#version 460 core

layout(location=0) in float vertex_depth;

uniform mat4 projection_matrix;
uniform mat4 model_pose;
uniform mat4 camera_pose;

uniform vec2 focal_length;
uniform int width;
uniform int height;

out vec2 fragment_uv;

void main() {
    mat4 vm = camera_pose * model_pose;
    mat4 pvm = projection_matrix * vm;
    
    int u_pixel = gl_VertexID % width;
    int v_pixel = gl_VertexID / width;
    
    float u = float(u_pixel) / width;
    float v = float(v_pixel) / height;
    
    float height_ratio = float(height) / width;
    
    fragment_uv.x = u;
    fragment_uv.y = v * height_ratio;
    
    float x = (u * 2.0 - 1.0) * vertex_depth / focal_length.x;
    float y = ((1. - v) * 2.0 - 1.0) * height_ratio * vertex_depth / focal_length.y;
    
    gl_Position = pvm * vec4(x, y, -vertex_depth, 1);
    //gl_Position = pvm * vec4(vertex_depth, 1);
}
'''

textured_depthmap_fragment_shader = '''#version 460 core
layout(binding=0) uniform sampler2D texture_sampler;
in vec2 fragment_uv;
out vec3 color;

void main() {
    color = texture(texture_sampler, fragment_uv).rgb;
    //color = vec3(1,1,0);
}
'''
