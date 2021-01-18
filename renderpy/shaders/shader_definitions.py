from renderpy.shaders.utils import phong_fn, softish_step_fn, intensity_fn
from renderpy.shaders.skybox import skybox_fn, panorama_to_cube_fragment_shader
from renderpy.shaders.image_light import reflection_to_diffuse_fragment_shader
from renderpy.shaders.mesh import mesh_vertex_shader
#from renderpy.shaders.lighting_model import color_fragment_shader
from renderpy.shaders.color_render import (
        textured_vertex_shader, textured_fragment_shader,
        vertex_color_vertex_shader, vertex_color_fragment_shader,
        flat_color_vertex_shader, flat_color_fragment_shader)
from renderpy.shaders.mask_render import (
        mask_vertex_shader, mask_fragment_shader)
from renderpy.shaders.coord_render import (
        coord_vertex_shader, coord_fragment_shader)
from renderpy.shaders.depthmap import (
        textured_depthmap_vertex_shader, textured_depthmap_fragment_shader)
from renderpy.shaders.background import (
        background_vertex_shader,
        background_2D_fragment_shader,
        background_fragment_shader)

#textured_vertex_shader = '''#version 460 core
##define COMPILE_TEXTURE''' + mesh_vertex_shader
#
#textured_fragment_shader = '''#version 460 core
##define COMPILE_TEXTURE''' + color_fragment_shader
#
#vertex_color_vertex_shader = '''#version 460 core
##define COMPILE_VERTEX_COLORS''' + mesh_vertex_shader
#
#vertex_color_fragment_shader = '''#version 460 core
##define COMPILE_VERTEX_COLORS''' + color_fragment_shader

#reflection_to_diffuse_fragment_shader = '''#version 460 core
##define NUM_SAMPLES 512
#in vec3 fragment_direction;
#in vec2 fragment_uv;
#out vec3 color;
#
#uniform float brightness;
#uniform float contrast;
#uniform float color_scale;
#uniform samplerCube reflect_sampler;
#uniform vec3 sphere_samples[NUM_SAMPLES];
#uniform int num_importance_samples;
#uniform float importance_sample_gain;
#uniform float random_sample_gain;
#
#void main(){
#    vec3 importance_color = vec3(0,0,0);
#    vec3 random_color = vec3(0,0,0);
#    vec3 fragment_direction_n = normalize(fragment_direction);
#    for(int i = 0; i < NUM_SAMPLES; ++i){
#        float d = dot(fragment_direction_n, sphere_samples[i]);
#        vec3 flipped_sample = sphere_samples[i] * sign(d);
#        vec3 sample_color = vec3(texture(reflect_sampler, flipped_sample));
#        
#        // brightness
#        sample_color += brightness;
#        // contrast
#        sample_color = (sample_color + 0.5) * contrast - 0.5;
#        
#        if(i >= num_importance_samples){ // || d < 0.){
#            random_color += sample_color * abs(d);
#        }
#        else{
#            if(d > 0.){
#                importance_color += sample_color * abs(d);
#            }
#        }
#    }
#    importance_color /= num_importance_samples;
#    random_color /= (NUM_SAMPLES - num_importance_samples);
#    
#    color = (importance_sample_gain * importance_color +
#            random_sample_gain * random_color) /
#            (importance_sample_gain + random_sample_gain);
#    color *= color_scale;
#}
#'''
