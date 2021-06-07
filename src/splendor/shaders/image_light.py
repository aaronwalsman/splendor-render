from splendor.shaders.utils import softish_step_fn, intensity_fn
from splendor.shaders.skybox import skybox_fn

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
    float num_samples = 0.;
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
        num_samples += 1.;
    }
    color /= num_samples;
}
'''
