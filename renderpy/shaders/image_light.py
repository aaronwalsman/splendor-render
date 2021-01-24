from renderpy.shaders.utils import softish_step_fn, intensity_fn
from renderpy.shaders.skybox import skybox_fn

image_light_diffuse_fn = f'''
{softish_step_fn}
{intensity_fn}
{skybox_fn}''' + '''
vec3 image_light_diffuse(
        float diffuse_contribution,
        vec3 fragment_normal,
        mat4 image_light_offset_matrix,
        float diffuse_min,
        float diffuse_max,
        float intensity_contrast,
        float intensity_target_lo,
        float intensity_target_hi,
        vec3 tint_lo,
        vec3 tint_hi){
    
    //==========================================================================
    // sample the raw diffuse light color from the diffuse sampler
    //==========================================================================
    vec4 offset_fragment_normal = image_light_offset_matrix * vec4(
            fragment_normal, 0.0);
    vec3 diffuse_color = vec3(skybox_texture(
            diffuse_sampler, offset_fragment_normal));
    float diffuse_intensity = intensity(diffuse_color);
    
    //==========================================================================
    // rescale the intensity
    //==========================================================================
    float diffuse_range = diffuse_max - diffuse_min;
    float diffuse_retarget_range = intensity_target_hi - intensity_target_lo;
    
    float soft_normalized_diffuse_intensity = 0.;
    if(diffuse_range > 0.){
        float normalized_diffuse_intensity =
                (diffuse_intensity - diffuse_min) / diffuse_range;
        soft_normalized_diffuse_intensity = softish_step(
            normalized_diffuse_intensity, intensity_contrast);
        float soft_diffuse_intensity = (
                soft_normalized_diffuse_intensity * diffuse_retarget_range) +
                intensity_target_lo;
        if(diffuse_intensity > 0.){
            diffuse_color *= soft_diffuse_intensity / diffuse_intensity;
        }
    }
    
    //==========================================================================
    // add the tint offset
    //==========================================================================
    vec3 diffuse_tint_color = mix(
            tint_lo, tint_hi, soft_normalized_diffuse_intensity);
    return diffuse_contribution * diffuse_color + diffuse_tint_color;
}
'''

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
