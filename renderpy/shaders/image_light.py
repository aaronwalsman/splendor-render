from renderpy.shaders.utils import softish_step_fn, intensity_fn
from renderpy.shaders.skybox import skybox_fn

image_light_diffuse_fn = softish_step_fn + intensity_fn + skybox_fn + '''
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

