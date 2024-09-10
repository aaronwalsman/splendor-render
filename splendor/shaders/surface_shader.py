from splendor.shaders.pbr import pbr_fns
from splendor.shaders.skybox import skybox_fn

surface_vertex_shader = '''#version 460 core
layout(location=0) in vec3 vertex_position;
layout(location=1) in vec3 vertex_normal;
layout(location=2) in vec2 vertex_uv;

out vec4 fragment_position;
out vec4 fragment_normal;
out vec2 fragment_uv;
out vec3 fragment_local_position;

uniform mat4 projection_matrix;
uniform mat4 model_pose;
uniform mat4 view_matrix;

void main(){
    mat4 vm = view_matrix * model_pose;
    mat4 pvm = projection_matrix * vm;
    gl_Position = pvm * vec4(vertex_position,1);
    
    fragment_position = vm * vec4(vertex_position,1);
    fragment_normal = vm * vec4(vertex_normal,0);
    
    //fragment_uv.x = vertex_uv.x;
    //fragment_uv.y =-vertex_uv.y;
    fragment_uv = vertex_uv;
    
    fragment_local_position = vertex_position;
}
'''

surface_fragment_shader = '''#version 460 core
in vec4 fragment_position;
in vec4 fragment_normal;
in vec2 fragment_uv;
in vec3 fragment_location_position;

out vec4 color;

uniform int use_textured_material_properties;
uniform vec4 material_properties;
uniform vec4 image_light_properties;
uniform bool image_light_active;
uniform vec3 background_color;

uniform int is_textured;
uniform vec3 flat_color;

uniform vec3 ambient_color;
//uniform int num_point_lights;
//uniform int num_direction_lights;

uniform mat4 image_light_offset_matrix;
uniform bool lock_image_light_to_camera;

//uniform vec3 point_light_data[2*MAX_NUM_LIGHTS];
//uniform vec3 direction_light_data[2*MAX_NUM_LIGHTS];

uniform mat4 view_matrix;

//uniform mat4 shadow_view_matrix;
//uniform mat4 shadow_projection_matrix;

layout(binding=0) uniform sampler2D texture_sampler;
layout(binding=1) uniform sampler2D material_properties_sampler;

/*
layout(binding=2) uniform samplerCube diffuse_sampler;
layout(binding=3) uniform samplerCube reflect_sampler;
*/

//layout(binding=4) uniform sampler2D shadow_depth_sampler;

const float MAX_MIPMAP = 4.;

''' + f'''
{pbr_fns}
{skybox_fn}''' + '''
void main(){
    
    // material properties =====================================================
    float metal = 0;
    float rough = 1;
    float base_reflect = 2;
    float ambient = 3;
    /*
    if(bool(use_textured_material_properties)){
        vec4 textured_material_properties = texture(
            material_properties_sampler, fragment_uv);
        metal = textured_material_properties.x;
        rough = textured_material_properties.y;
        base_reflect = textured_material_properties.z;
        ambient = textured_material_properties.w;
    }
    else{
    */
        metal = material_properties.x;
        rough = material_properties.y;
        base_reflect = material_properties.z;
        ambient = material_properties.w;
    //}
    
    float diffuse_gamma = image_light_properties.x;
    float diffuse_bias = image_light_properties.y;
    float reflect_gamma = image_light_properties.z;
    float reflect_bias = image_light_properties.w;
    
    vec3 eye = normalize(vec3(-fragment_position));
    vec3 normal = normalize(vec3(fragment_normal));
    vec3 camera_normal = normal;
    if(!lock_image_light_to_camera){
        camera_normal = vec3(inverse(view_matrix) * vec4(camera_normal, 0.));
    }
    // albedo ==================================================================
    vec3 albedo = flat_color;
    if(bool(is_textured)){
        albedo = texture(texture_sampler, fragment_uv).rgb;
    }
    
    vec3 f0 = mix(vec3(base_reflect), albedo, metal);
    
    color = vec4(0., 0., 0., 1.);
    
    /*
    // point lights ============================================================
    for(int i = 0; i < num_point_lights; ++i){
        
        vec3 light_color = point_light_data[2*i];
        vec3 light_position = point_light_data[2*i+1];
        light_position = vec3(view_matrix * vec4(light_position, 1.));
        vec3 light_direction = light_position - vec3(fragment_position);
        light_direction = normalize(light_direction);
        
        vec3 half_direction = normalize(eye + light_direction);
        
        vec3 light_contribution = cook_torrance(
            rough,
            metal,
            f0,
            albedo,
            eye,
            normal,
            half_direction,
            light_direction,
            light_color);
        
        color += vec4(light_contribution, 0.);
    }
    
    // direction lights ========================================================
    for(int i = 0; i < num_direction_lights; ++i){
            
        vec3 light_color = vec3(direction_light_data[2*i]);
        vec3 light_direction = -normalize(direction_light_data[2*i+1]);
        light_direction = vec3(view_matrix * vec4(light_direction, 0.));
        vec3 half_direction = normalize(eye + light_direction);
        
        vec3 light_contribution = cook_torrance(
            rough,
            metal,
            f0,
            albedo,
            eye,
            normal,
            half_direction,
            light_direction,
            light_color);
        
        vec4 shadow_fragment_position = (
            shadow_projection_matrix * shadow_view_matrix * fragment_position);
        vec3 shadow_p = (
            shadow_fragment_position.xyz / shadow_fragment_position.w);
        shadow_p = shadow_p * 0.5 + 0.5;
        shadow_p.y = (shadow_p.y - 0.5) * 2 + 0.5;
        shadow_p.x = (shadow_p.x - 0.5) * 2 + 0.5;
        float shadow_depth = texture(shadow_depth_sampler, shadow_p.xy).x;
        float shadow_bias = max(
            0.05 * (1. - dot(normal, light_direction)), 0.005);
        float in_shadow = shadow_p.z-shadow_bias > shadow_depth ? 1.0 : 0.0;
        
        color += vec4(light_contribution, 0.) * (1. - in_shadow);
    }
    */
    
    // reflect =================================================================
    float cos_theta = dot(normal, eye);
    vec3 ks = fresnel_schlick_rough(cos_theta, f0, rough);
    vec3 kd = (1. - ks) * (1. - metal);
    
    /*
    // image light =============================================================
    if(image_light_active){
        
        vec3 offset_fragment_normal = vec3(
                image_light_offset_matrix * vec4(camera_normal, 1.));
        
        vec3 diffuse_color = vec3(skybox_texture(
                diffuse_sampler, offset_fragment_normal));
        diffuse_color =
                pow(diffuse_color, vec3(diffuse_gamma));
        
        // This correction is based on the very crude approximation that
        // the distribution of intensities in the reflection image is uniform
        // which means the area under the intensity curve (from 0 to 1) would
        // be 1/2.  If we apply a gamma exponent to this, the new area will be
        // 1/(gamma+1).  A multiplicative correction is then (gamma+1)/2.
        float diffuse_correction = (diffuse_gamma+1)/2;
        diffuse_color *= diffuse_correction;
        diffuse_color += vec3(diffuse_bias);
        
        color += vec4(kd * diffuse_color * albedo, 0.);
        
        vec4 reflected_direction =
                inverse(view_matrix) *
                vec4(reflect(-eye, normal), 0.);
        reflected_direction = image_light_offset_matrix * reflected_direction;
        vec3 reflect_color = vec3(skybox_texture(
                reflect_sampler, reflected_direction, rough*MAX_MIPMAP));
        reflect_color =
                pow(reflect_color, vec3(reflect_gamma));
        
        // See note above about diffuse correction
        float reflect_correction = (reflect_gamma+1)/2;
        reflect_color *= reflect_correction;
        reflect_color += vec3(reflect_bias);
        
        reflect_color = reflect_color * ks;
        color += vec4(reflect_color, 0.);
    }
    */
    
    // ambient and background ==================================================
    color += vec4(kd * ambient_color * ambient * albedo, 0.);
    color += vec4(ks * background_color, 0.);
}
'''
