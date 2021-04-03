from renderpy.shaders.utils import softish_step_fn, intensity_fn
from renderpy.shaders.skybox import skybox_fn

image_light_diffuse_fn = f'''
// DEPRECATED
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

def reflect_to_diffuse_fragment_shader(num_samples=512):
    return f'''#version 460 core
#define NUM_SAMPLES {num_samples}''' + '''
in vec3 fragment_direction;
in vec2 fragment_uv;
out vec3 color;

uniform samplerCube reflect_sampler;
uniform vec4 sample_direction_importance_ratio[NUM_SAMPLES];

void main(){
    color = vec3(0,0,0);
    vec3 fragment_direction_n = normalize(fragment_direction);
    for(int i = 0; i < NUM_SAMPLES; ++i){
        vec3 sample_direction = vec3(sample_direction_importance_ratio[i]);
        float importance_ratio = sample_direction_importance_ratio[i].w;
        if(importance_ratio == 0.){
            continue;
        }
        float d = dot(fragment_direction_n, sample_direction);
        vec3 flipped_sample = sample_direction * sign(d);
        
        vec4 reflect_sample = texture(reflect_sampler, flipped_sample);
        vec3 sample_color = vec3(reflect_sample);
        float sample_intensity = reflect_sample.w;
        
        color += sample_color * abs(d) * sample_intensity * importance_ratio;
    }
    color /= NUM_SAMPLES;
}
'''
