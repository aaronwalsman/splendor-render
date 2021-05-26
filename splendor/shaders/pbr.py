'''
Notes:
------
Much of this follows a series of articles that starts here
with my own tweaks and hacks sprinkled liberally throughout:
https://learnopengl.com/PBR/Theory
'''

pbr_fns = '''
const float PI = 3.14159265359;

float distribution_trowbridge_reitz(
    float rough,
    vec3 normal,
    vec3 half_direction){
    
    float rough2 = rough * rough;
    rough2 = rough2 * rough2; // Disney/Epic adjustment
    float normal_half = max(dot(normal, half_direction), 0.);
    float normal_half2 = normal_half * normal_half;
    float denominator = normal_half2 * (rough2 - 1.) + 1.;
    denominator = PI * denominator * denominator;
    
    return rough2 / max(denominator, 0.0000001);
}

float obstruction_schlick_beckmann(
    float rough,
    vec3 normal,
    vec3 direction,
    bool image_based_light){
    
    float k = 0.;
    if(image_based_light){
        k = rough * rough / 2.;
    }
    else{
        k = (rough+1)*(rough+1) / 8.;
    }
    
    float normal_eye = max(dot(normal, direction), 0.);
    float denominator = normal_eye * (1. - k) + k;
    
    return normal_eye / denominator;
}

float obstruction_smith(
    float rough,
    vec3 normal,
    vec3 light_direction,
    vec3 eye,
    bool image_based_light){
    
    float obstruction_eye = obstruction_schlick_beckmann(
        rough, normal, eye, image_based_light);
    float obstruction_light = obstruction_schlick_beckmann(
        rough, normal, light_direction, image_based_light);
    return obstruction_eye * obstruction_light;
}

vec3 fresnel_schlick(
    float cos_theta,
    vec3 f0){
    
    // the 0.2 clamp here is to prevent fresnel chattering around the edges
    cos_theta = clamp(cos_theta, 0.2, 1.);
    return f0 + (1. - f0) * pow(1. - cos_theta, 5.);
}

vec3 fresnel_schlick_rough(
    float cos_theta,
    vec3 f0,
    float rough){
    
    // the 0.2 clamp here is to prevent fresnel chattering around the edges
    cos_theta = clamp(cos_theta, 0.2, 1.);
    return f0 + (max(vec3(1. - rough), f0) - f0) * pow(1. - cos_theta, 5.);
}

vec3 cook_torrance(
    float rough,
    float metal,
    vec3 f0,
    vec3 albedo,
    vec3 eye,
    vec3 normal,
    vec3 half_direction,
    vec3 light_direction,
    vec3 light_color){
    
    float specular_distribution = distribution_trowbridge_reitz(
        rough, normal, half_direction);
    float specular_obstruction = obstruction_smith(
            rough, normal, light_direction, eye, false);
    float cos_theta = clamp(dot(half_direction, eye), 0., 1.);
    
    vec3 fresnel = fresnel_schlick(cos_theta, f0);
    
    float eye_incidence = max(dot(eye, normal), 0.);
    float light_incidence = max(dot(light_direction, normal), 0.);
    float denominator = 4 * eye_incidence * light_incidence;
    vec3 numerator = specular_distribution * fresnel * specular_obstruction;
    vec3 specular = numerator / max(denominator, 0.001);
    
    vec3 diffuse = (vec3(1.) - fresnel) * (1. - metal);
    
    float lambert = max(dot(normal, light_direction), 0.0);
    lambert = lambert;
    
    vec3 result = (diffuse * albedo / PI + specular) * light_color * lambert;
    return result;
}
'''
