from renderpy.shaders.utils import phong_fn
from renderpy.shaders.image_light import image_light_diffuse_fn
from renderpy.shaders.cook_torrance import cook_torrance_fn
from renderpy.shaders.skybox import skybox_fn

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
uniform vec3 reflect_color;

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

layout(binding=1) uniform sampler2D brdf_lut;
layout(binding=2) uniform samplerCube diffuse_sampler;
layout(binding=3) uniform samplerCube reflect_sampler;
''' + f'''
{cook_torrance_fn}
{image_light_diffuse_fn}''' + '''
void main(){
    
    // material properties =====================================================
    float ambient = material_properties.x;
    float metal = material_properties.y;
    float rough = material_properties.z;
    float reflect_gamma = material_properties.w;
    
    vec3 eye = normalize(vec3(-fragment_position));
    vec3 normal = normalize(vec3(fragment_normal));
    vec3 camera_normal = vec3(inverse(camera_matrix) * vec4(normal, 0.));
    
    // albedo ==================================================================
    #ifdef COMPILE_TEXTURE
    vec3 albedo = texture(texture_sampler, fragment_uv).rgb;
    //vec3 albedo = texture(brdf_lut, fragment_uv).rgb;
    #endif
    
    #ifdef COMPILE_VERTEX_COLORS
    vec3 albedo = fragment_color;
    #endif
    
    #ifdef COMPILE_FLAT_COLOR
    vec3 albedo = flat_color;
    #endif
    
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
            albedo,
            eye,
            normal,
            half_direction,
            light_direction,
            light_color,
            reflect_color,
            false);
        
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
            albedo,
            eye,
            normal,
            half_direction,
            light_direction,
            light_color,
            reflect_color,
            false);
    }
    
    // image light =============================================================
    vec3 f0 = mix(reflect_color, albedo, metal);
    vec3 ks = fresnel_schlick_rough(normal, eye, f0, rough);
    vec3 kd = (1. - ks) * (1. - metal);
    
    
    /*
    vec3 f0 = mix(vec3(1.,1.,1.), albedo, metal);
    float kks = fresnel_schlick_simple(normal, eye, reflect_color.x, rough);
    kks = kks + (1. - kks) * metal;
    vec3 ks = kks * f0;
    float kd = (1. - kks) * (1. - metal);
    */
    vec3 offset_fragment_normal = vec3(
            image_light_offset_matrix * vec4(camera_normal, 1.));
    
    vec3 diffuse_color = vec3(skybox_texture(
            diffuse_sampler, offset_fragment_normal));
    
    color += vec4(kd * diffuse_color * albedo, 0.);
    
    vec4 reflected_direction =
            inverse(camera_matrix) *
            vec4(reflect(-eye, normal), 0.);
    reflected_direction = image_light_offset_matrix * reflected_direction;
    vec3 specular_color = vec3(skybox_texture(
            reflect_sampler, reflected_direction, rough*4));
    //specular_color = specular_color * 0.00001 + vec3(0,0,1);
    specular_color = pow(specular_color, vec3(reflect_gamma));
    
    vec2 brdf = texture(
        brdf_lut, vec2(max(dot(normal, eye), 0.), rough), 0).rg;
    //brdf.x = brdf.x * 0.0000001 + 1.;
    //brdf.x = brdf.x + (1. - brdf.x); // MAYBE JUST ADJUST THIS FOR METAL?
    //brdf.y = brdf.y * 0.0000001 + 0.;
    specular_color = specular_color * (ks * brdf.x + brdf.y);
    color += vec4(specular_color, 0.);
    
    // ambient light ===========================================================
    color += vec4(ambient_color * albedo, 0.);
}
'''
