from renderpy.shaders.utils import phong_fn
from renderpy.shaders.image_light import image_light_diffuse_fn

lighting_model_fragment_shader = '''
const int MAX_NUM_LIGHTS = 8;

in vec4 fragment_position;
in vec4 fragment_normal;

#ifdef COMPILE_TEXTURE
in vec2 fragment_uv;
#endif

#ifdef COMPILE_VERTEX_COLORS
in vec3 fragment_color;
#endif

out vec3 color;

uniform vec4 material_properties;

#ifdef COMPILE_FLAT_COLOR
uniform vec3 flat_color;
#endif

uniform vec3 ambient_color;
uniform int num_point_lights;
uniform int num_direction_lights;
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

#ifdef COMPILE_TEXTURE
layout(binding=0) uniform sampler2D texture_sampler;
#endif

layout(binding=2) uniform samplerCube diffuse_sampler;
layout(binding=3) uniform samplerCube reflect_sampler;
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
    vec3 reflected_color = vec3(skybox_texture(
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
    
    #ifdef COMPILE_TEXTURE
    vec3 diffuse_color = texture(texture_sampler, fragment_uv).rgb;
    #endif
    
    #ifdef COMPILE_VERTEX_COLORS
    vec3 diffuse_color = fragment_color;
    #endif
    
    #ifdef COMPILE_FLAT_COLOR
    vec3 diffuse_color = flat_color;
    #endif
    
    float diffuse_intensity = intensity(image_light_diffuse_color);
    
    // if diffuse contribution is greater than one
    // interpolate the texture color to white
    // this is designed to simulate blow-out or over-exposure
    // and correct for some bad artifacts that happen with blown-out lights
    if(diffuse_intensity > 1.0){
        diffuse_color = mix(diffuse_color, vec3(1.0,1.0,1.0),
                diffuse_intensity - 1.0);
    }
    
    color = vec3(
            ambient_color * diffuse_color * ka +
            diffuse_contribution * diffuse_color * kd +
            specular_contribution * ks +
            image_light_diffuse_color * diffuse_color +
            image_light_reflection);
}
'''
