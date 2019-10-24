phong_fn = '''
vec2 phong(
        vec3 contact_normal,
        vec3 light_direction,
        vec3 eye_direction,
        float shine){
    
    float diffuse = clamp(dot(-light_direction, contact_normal), 0, 1);
    
    vec3 reflected_direction = reflect(-light_direction, contact_normal);
    float specular = clamp(dot(-eye_direction, reflected_direction), 0, 1);
    if(specular > 0.0){
        specular = pow(specular, shine);
    }
    
    return vec2(diffuse, specular);
}
'''

softish_step_fn = '''
float softish_step(float t, float alpha){
    if(t <= 0){
        return 0;
    }
    else if(t >= 1){
        return 1;
    }
    
    if(t < 0.5){
        return 0.5 * pow((2*t),alpha);
    }
    else{
        return -0.5 * pow((2*(1-t)),alpha)+1;
    }
}
'''

intensity_fn = '''
float intensity(vec3 c){
    return c.x * 0.2990 + c.y * 0.5870 + c.z * 0.1140;
}
'''

image_light_diffuse_fn = softish_step_fn + intensity_fn + '''
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
    vec3 diffuse_color = vec3(texture(
            diffuse_sampler, vec3(offset_fragment_normal)));
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

textured_vertex_shader = '''#version 330 core

layout(location=0) in vec3 vertex_position;
layout(location=1) in vec3 vertex_normal;
layout(location=2) in vec2 vertex_uv;

out vec4 fragment_position;
out vec4 fragment_normal;
out vec2 fragment_uv;
//out vec4 fragment_shadow_position;
//out vec4 fragment_shadow_normal;
//out vec4 fragment_shadow_projection;

//uniform mat4 shadow_light_pose;
//uniform mat4 shadow_light_projection;

uniform mat4 projection_matrix;
uniform mat4 model_pose;
uniform mat4 camera_pose;

void main(){
    
    mat4 vm = camera_pose * model_pose;
    mat4 pvm = projection_matrix * vm;
    
    //mat4 lm = shadow_light_pose * model_pose;
    
    gl_Position = pvm * vec4(vertex_position,1);
    
    fragment_position = vm * vec4(vertex_position,1);
    fragment_normal = vm * vec4(vertex_normal,0);
    
    fragment_uv.x = vertex_uv.x;
    fragment_uv.y =-vertex_uv.y;
    
    //fragment_shadow_position = lm * vec4(vertex_position,1);
    //fragment_shadow_projection =
    //        shadow_light_projection * fragment_shadow_position;
    //fragment_shadow_normal = lm * vec4(vertex_normal,0);
}
'''

textured_fragment_shader = '''#version 330 core

const int MAX_NUM_LIGHTS = 8;

in vec4 fragment_position;
in vec4 fragment_normal;
in vec2 fragment_uv;
//in vec4 fragment_shadow_position;
//in vec4 fragment_shadow_projection;
//in vec4 fragment_shadow_normal;

out vec3 color;

uniform vec4 material_properties;

uniform vec3 ambient_color;
uniform int num_point_lights;
uniform int num_direction_lights;
//uniform int enable_shadow_light;
//uniform vec3 shadow_light_color;
//uniform mat4 shadow_light_pose;
//uniform mat4 shadow_light_projection;
uniform mat4 image_light_offset_matrix;
uniform vec2 image_light_diffuse_minmax;
uniform vec3 image_light_diffuse_rescale;
uniform vec3 image_light_diffuse_tint_lo;
uniform vec3 image_light_diffuse_tint_hi;
uniform vec3 image_light_reflect_tint;
uniform vec3 image_light_material_properties;
uniform vec3 point_light_data[2*MAX_NUM_LIGHTS];
uniform vec3 direction_light_data[2*MAX_NUM_LIGHTS];

uniform mat4 camera_pose;

uniform sampler2D texture_sampler;
//uniform sampler2D shadow_sampler;
uniform samplerCube diffuse_sampler;
uniform samplerCube reflect_sampler;
''' + phong_fn + image_light_diffuse_fn + '''

void main(){
    
    float ka = material_properties.x;
    float kd = material_properties.y;
    float ks = material_properties.z;
    float shine = material_properties.w;
    
    float k_image_light_diffuse = image_light_material_properties.x;
    float k_image_light_reflect = image_light_material_properties.y;
    float k_image_light_reflect_blur = image_light_material_properties.z;
    
    float k_image_light_contrast = image_light_diffuse_rescale.x;
    float k_image_light_target_lo = image_light_diffuse_rescale.y;
    float k_image_light_target_hi = image_light_diffuse_rescale.z;
    
    vec3 ambient_contribution = ambient_color;
    vec3 diffuse_contribution = vec3(0.0);
    vec3 specular_contribution = vec3(0.0);
    
    vec3 eye_direction = normalize(vec3(-fragment_position));
    
    vec3 fragment_normal_n = normalize(vec3(fragment_normal));
    
    
    //if(enable_shadow_light){
    //    vec2 shadow_phong = phong(
    //            ...);
    //    diffuse_contribution += shadow_light_color * shadow_phong.x;
    //    specular_contribution += shadow_light_color * shadow_phong.y;
    //}
    
    
    // image light
    vec3 camera_fragment_normal =
            vec3(inverse(camera_pose) * vec4(fragment_normal_n,0));
    vec3 image_light_diffuse_color = image_light_diffuse(
            k_image_light_diffuse,
            camera_fragment_normal,
            image_light_offset_matrix,
            image_light_diffuse_minmax.x,
            image_light_diffuse_minmax.y,
            k_image_light_contrast,
            k_image_light_target_lo,
            k_image_light_target_hi,
            image_light_diffuse_tint_lo,
            image_light_diffuse_tint_hi);
    
    vec3 reflected_direction = vec3(
            inverse(camera_pose) *
            vec4(reflect(-eye_direction, fragment_normal_n),0));
    vec4 offset_reflected_direction = image_light_offset_matrix *
            vec4(reflected_direction, 0.);
    vec3 reflected_color = vec3(texture(
            reflect_sampler,
            vec3(offset_reflected_direction),
            k_image_light_reflect_blur)) + image_light_reflect_tint;
    reflected_color = pow(reflected_color, vec3(shine, shine, shine));
    vec3 image_light_reflection = k_image_light_reflect * reflected_color;
    
    // point lights
    for(int i = 0; i < num_point_lights; ++i){
        
        vec3 light_color = vec3(point_light_data[2*i]);
        vec3 light_position = vec3(
                camera_pose * vec4(point_light_data[2*i+1],1));
        vec3 light_direction = vec3(fragment_position) - light_position;
        float light_distance = length(light_direction);
        light_direction = light_direction / light_distance;
        
        vec2 light_phong = phong(
                fragment_normal_n,
                light_direction,
                eye_direction,
                shine);
        
        diffuse_contribution += light_color * light_phong.x;
        specular_contribution += light_color * light_phong.y;
    }
    
    // direction lights
    for(int i = 0; i < num_direction_lights; ++i){
        
        vec3 light_color = vec3(direction_light_data[2*i]);
        vec3 light_direction = vec3(
                camera_pose * vec4(direction_light_data[2*i+1],0));
        
        vec2 light_phong = phong(
                fragment_normal_n,
                light_direction,
                eye_direction,
                shine);
        
        diffuse_contribution += light_color * light_phong.x;
        specular_contribution += light_color * light_phong.y;
    }
    
    vec3 texture_color = texture(texture_sampler, fragment_uv).rgb;
    
    //==========================================================================
    // THIS SECTION IS EXPERIMENTAL
    float diffuse_intensity = intensity(image_light_diffuse_color);
    
    // if diffuse contribution is greater than one
    // interpolate the texture color to white
    // this is designed to simulate blow-out or over-exposure
    if(diffuse_intensity > 1.0){
        texture_color = mix(texture_color, vec3(1.0,1.0,1.0),
                //image_light_diffuse_color,
                diffuse_intensity - 1.0);
    }
    // END EXPERIMENTAL SECTION
    //==========================================================================
    
    color = vec3(
            ambient_color * texture_color * ka +
            diffuse_contribution * texture_color * kd +
            specular_contribution * ks +
            image_light_diffuse_color * texture_color +
            image_light_reflection);
}
'''

vertex_color_vertex_shader = '''#version 330 core

layout(location=0) in vec3 vertex_position;
layout(location=1) in vec3 vertex_normal;
layout(location=2) in vec3 vertex_color;

out vec4 fragment_position;
out vec4 fragment_normal;
out vec3 fragment_color;

uniform mat4 projection_matrix;
uniform mat4 model_pose;
uniform mat4 camera_pose;

void main(){
    mat4 vm = camera_pose * model_pose;
    mat4 pvm = projection_matrix * vm;
    
    gl_Position = pvm * vec4(vertex_position,1);
    
    fragment_position = vm * vec4(vertex_position,1);
    fragment_normal = vm * vec4(vertex_normal,0);
    fragment_color = vertex_color;
}
'''

vertex_color_fragment_shader = '''#version 330 core

const int MAX_NUM_LIGHTS = 8;

in vec4 fragment_position;
in vec4 fragment_normal;
in vec3 fragment_color;

out vec3 color;

uniform vec4 material_properties;

uniform vec3 ambient_color;
uniform int num_point_lights;
uniform int num_direction_lights;
//uniform int enable_shadow_light;
//uniform vec3 shadow_light_color;
//uniform mat4 shadow_light_pose;
//uniform mat4 shadow_light_projection;
uniform mat4 image_light_offset_matrix;
uniform vec2 image_light_diffuse_minmax;
uniform vec3 image_light_diffuse_rescale;
uniform vec3 image_light_diffuse_tint_lo;
uniform vec3 image_light_diffuse_tint_hi;
uniform vec3 image_light_reflect_tint;
uniform vec3 image_light_material_properties;
uniform vec3 point_light_data[2*MAX_NUM_LIGHTS];
uniform vec3 direction_light_data[2*MAX_NUM_LIGHTS];

uniform mat4 camera_pose;

//uniform sampler2D shadow_sampler;
uniform samplerCube diffuse_sampler;
uniform samplerCube reflect_sampler;
''' + phong_fn + image_light_diffuse_fn + '''

void main(){
    
    float ka = material_properties.x;
    float kd = material_properties.y;
    float ks = material_properties.z;
    float shine = material_properties.w;
    
    float k_image_light_diffuse = image_light_material_properties.x;
    float k_image_light_reflect = image_light_material_properties.y;
    float k_image_light_reflect_blur = image_light_material_properties.z;
    
    float k_image_light_contrast = image_light_diffuse_rescale.x;
    float k_image_light_target_lo = image_light_diffuse_rescale.y;
    float k_image_light_target_hi = image_light_diffuse_rescale.z;
    
    vec3 ambient_contribution = ambient_color;
    vec3 diffuse_contribution = vec3(0.0);
    vec3 specular_contribution = vec3(0.0);
    
    vec3 eye_direction = normalize(vec3(-fragment_position));
    
    vec3 fragment_normal_n = normalize(vec3(fragment_normal));
    
    // image light
    vec3 camera_fragment_normal =
            vec3(inverse(camera_pose) * vec4(fragment_normal_n,0));
    vec3 image_light_diffuse_color = image_light_diffuse(
            k_image_light_diffuse,
            camera_fragment_normal,
            image_light_offset_matrix,
            image_light_diffuse_minmax.x,
            image_light_diffuse_minmax.y,
            k_image_light_contrast,
            k_image_light_target_lo,
            k_image_light_target_hi,
            image_light_diffuse_tint_lo,
            image_light_diffuse_tint_hi);
    
    vec3 reflected_direction = vec3(
            inverse(camera_pose) *
            vec4(reflect(-eye_direction, fragment_normal_n),0));
    vec4 offset_reflected_direction = image_light_offset_matrix *
            vec4(reflected_direction, 0.);
    vec3 reflected_color = vec3(texture(
            reflect_sampler,
            vec3(offset_reflected_direction),
            k_image_light_reflect_blur)) + image_light_reflect_tint;
    vec3 image_light_reflection = k_image_light_reflect * reflected_color;
    
    // point lights
    for(int i = 0; i < num_point_lights; ++i){
        
        vec3 light_color = vec3(point_light_data[2*i]);
        vec3 light_position = vec3(
                camera_pose * vec4(point_light_data[2*i+1],1));
        vec3 light_direction = vec3(fragment_position) - light_position;
        float light_distance = length(light_direction);
        light_direction = light_direction / light_distance;
        
        vec2 light_phong = phong(
                fragment_normal_n,
                light_direction,
                eye_direction,
                shine);
        
        diffuse_contribution += light_color * light_phong.x;
        specular_contribution += light_color * light_phong.y;
    }
    
    // direction lights
    for(int i = 0; i < num_direction_lights; ++i){
        
        vec3 light_color = vec3(direction_light_data[2*i]);
        vec3 light_direction = vec3(
                camera_pose * vec4(direction_light_data[2*i+1],0));
        
        vec2 light_phong = phong(
                fragment_normal_n,
                light_direction,
                eye_direction,
                shine);
        
        diffuse_contribution += light_color * light_phong.x;
        specular_contribution += light_color * light_phong.y;
    }
    
    vec3 texture_color = fragment_color;
    
    //==========================================================================
    // THIS SECTION IS EXPERIMENTAL
    float diffuse_intensity = intensity(image_light_diffuse_color);
    
    // if diffuse contribution is greater than one
    // interpolate the texture color to white
    // this is designed to simulate blow-out or over-exposure
    if(diffuse_intensity > 1.0){
        texture_color = mix(texture_color, vec3(1.0,1.0,1.0),
                //image_light_diffuse_color,
                diffuse_intensity - 1.0);
    }
    // END EXPERIMENTAL SECTION
    //==========================================================================
    
    color = vec3(
            ambient_color * texture_color * ka +
            diffuse_contribution * texture_color * kd +
            specular_contribution * ks +
            image_light_diffuse_color * texture_color +
            image_light_reflection);
}
'''

background_vertex_shader = '''#version 330 core

uniform mat4 projection_matrix;
uniform mat4 camera_pose;

out vec3 fragment_direction;
out vec2 fragment_uv;

#define FAR 1-(1e-4)

void main(){
    mat4 inv_vp = inverse(projection_matrix * camera_pose);
    vec4 p0 = inverse(camera_pose) * vec4(0,0,0,1);
    if(gl_VertexID == 0){
        gl_Position = vec4(-1,-1,FAR,1);
        fragment_uv = vec2(0,0);
    }
    else if(gl_VertexID == 1){
        gl_Position = vec4(-1, 1,FAR,1);
        fragment_uv = vec2(0,1);
    }
    else if(gl_VertexID == 2){
        gl_Position = vec4( 1, 1,FAR,1);
        fragment_uv = vec2(1,1);
    }
    else if(gl_VertexID == 3){
        gl_Position = vec4( 1,-1,FAR,1);
        fragment_uv = vec2(1,0);
    }
    
    vec4 p1 = inv_vp * gl_Position;
    p1 /= p1.w;
    fragment_direction = vec3(p1 - p0);
}
'''

background_2D_fragment_shader = '''#version 330 core
in vec2 fragment_uv;
out vec3 color;

uniform sampler2D texture_sampler;

void main(){
    color = texture(texture_sampler, fragment_uv).rgb;
}
'''

background_fragment_shader = '''#version 330 core
in vec3 fragment_direction;
out vec3 color;

uniform float blur;
uniform samplerCube cubemap_sampler;
uniform mat4 offset_matrix;

void main(){
    vec4 offset_direction = offset_matrix *
            vec4(fragment_direction, 0.);
    color = vec3(textureLod(cubemap_sampler, vec3(offset_direction), blur));
}
'''

panorama_to_cube_fragment_shader = '''#version 330 core
#define M_PI 3.1415926535897932384626433832795

in vec4 fragment_direction;
out vec3 color;

uniform sampler2D texture_sampler;

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

reflection_to_diffuse_fragment_shader = '''#version 330 core
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

mask_vertex_shader = '''#version 330 core

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

mask_fragment_shader = '''#version 330 core

out vec3 color;

uniform vec3 mask_color;

void main(){
    
    color = mask_color;
}
'''
