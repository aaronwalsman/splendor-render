from renderpy.shaders.utils import phong_fn
from renderpy.shaders.image_light import image_light_diffuse_fn
from renderpy.shaders.pbr import pbr_fns
from renderpy.shaders.skybox import skybox_fn
from renderpy.shaders.utils import softish_step_fn

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

out vec4 color;

uniform vec4 material_properties;
uniform vec4 image_light_properties;

#ifdef COMPILE_FLAT_COLOR
uniform vec3 flat_color;
#endif

uniform vec3 ambient_color;
uniform int num_point_lights;
uniform int num_direction_lights;

uniform mat4 image_light_offset_matrix;

uniform vec3 point_light_data[2*MAX_NUM_LIGHTS];
uniform vec3 direction_light_data[2*MAX_NUM_LIGHTS];

uniform mat4 camera_matrix;

#ifdef COMPILE_TEXTURE
layout(binding=0) uniform sampler2D texture_sampler;
#endif

layout(binding=2) uniform samplerCube diffuse_sampler;
layout(binding=3) uniform samplerCube reflect_sampler;

const float MAX_MIPMAP = 4.;

''' + f'''
{pbr_fns}
{image_light_diffuse_fn}''' + '''
void main(){
    
    // material properties =====================================================
    float ambient = material_properties.x;
    float metal = material_properties.y;
    float rough = material_properties.z;
    float base_reflect = material_properties.w;
    
    float diffuse_gamma = image_light_properties.x;
    float diffuse_bias = image_light_properties.y;
    float reflect_gamma = image_light_properties.z;
    float reflect_bias = image_light_properties.w;
    
    vec3 eye = normalize(vec3(-fragment_position));
    vec3 normal = normalize(vec3(fragment_normal));
    vec3 camera_normal = vec3(inverse(camera_matrix) * vec4(normal, 0.));
    
    // albedo ==================================================================
    #ifdef COMPILE_TEXTURE
    vec3 albedo = texture(texture_sampler, fragment_uv).rgb;
    #endif
    
    #ifdef COMPILE_VERTEX_COLORS
    vec3 albedo = fragment_color;
    #endif
    
    #ifdef COMPILE_FLAT_COLOR
    vec3 albedo = flat_color;
    #endif
    
    vec3 f0 = mix(vec3(base_reflect), albedo, metal);
    
    color = vec4(0., 0., 0., 1.);
    
    // point lights ============================================================
    for(int i = 0; i < num_point_lights; ++i){
        
        vec3 light_color = vec3(point_light_data[2*i]);
        vec3 light_position = point_light_data[2*i+1];
        light_position = vec3(camera_matrix * vec4(light_position,1));
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
        vec3 light_direction = direction_light_data[2*i+1];
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
    }
    
    // image light =============================================================
    vec3 ks = fresnel_schlick_rough(normal, eye, f0, rough);
    vec3 kd = (1. - ks) * (1. - metal);
    
    vec3 offset_fragment_normal = vec3(
            image_light_offset_matrix * vec4(camera_normal, 1.));
    
    vec3 diffuse_color = vec3(skybox_texture(
            diffuse_sampler, offset_fragment_normal));
    diffuse_color =
            pow(diffuse_color, vec3(diffuse_gamma)) + vec3(diffuse_bias);
    
    color += vec4(kd * diffuse_color * albedo, 0.);
    
    vec4 reflected_direction =
            inverse(camera_matrix) *
            vec4(reflect(-eye, normal), 0.);
    reflected_direction = image_light_offset_matrix * reflected_direction;
    vec3 reflect_color = vec3(skybox_texture(
            reflect_sampler, reflected_direction, rough*MAX_MIPMAP));
    reflect_color =
            pow(reflect_color, vec3(reflect_gamma)) + vec3(reflect_bias);
    
    reflect_color = reflect_color * ks;
    color += vec4(reflect_color, 0.);
    
    // ambient light ===========================================================
    color += vec4(ambient_color * albedo, 0.);
    
    // HDR tonemapping
    //color = color / (color + vec4(1.0, 1.0, 1.0, 0.0));
    // gamma correct
    //float g = 1./2.2;
    //color = pow(color, vec4(g,g,g,1.));
}
'''
