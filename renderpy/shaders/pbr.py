'''
Notes:
------
Much of this follows a series of articles that starts here:
https://learnopengl.com/PBR/Theory
'''

cook_torrance_fn = '''
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
    vec3 half_direction,
    vec3 eye,
    vec3 f0,
    vec3 albedo,
    float metal){
    f0 = mix(f0, albedo, metal);
    
    float half_eye = clamp(dot(half_direction, eye), 0., 1.);
    return f0 + (1. - f0) * pow(1. - half_eye, 5.);
}

vec3 fresnel_schlick_rough(
    vec3 normal,
    vec3 eye,
    vec3 f0,
    float rough){
    
    float incidence = clamp(dot(normal, eye), 0., 1.);
    return f0 + (max(vec3(1. - rough), f0) - f0) * pow(1. - incidence, 5.);
    //return f0 + (1. - f0) * pow(1. - incidence, 5.);
}

float fresnel_schlick_simple(
    vec3 normal,
    vec3 eye,
    float base_reflect,
    float rough){
    float incidence = clamp(dot(normal, eye), 0., 1.);
    return base_reflect + (max((1. - rough), base_reflect) - base_reflect) *
            pow(1. - incidence, 5.);
}

/*
float fresnel_schlick_rough2(
    vec3 normal,
    vec3 eye,
    float rough){
    
    float incidence = clamp(dot(normal, eye), 0., 1.);
    return 1. + 
}
*/

vec3 cook_torrance(
    float rough,
    float metal,
    vec3 albedo,
    vec3 eye,
    vec3 normal,
    vec3 half_direction,
    vec3 light_direction,
    vec3 light_color,
    vec3 reflect_color,
    bool image_based_light){
    
    float specular_distribution = distribution_trowbridge_reitz(
        rough, normal, half_direction);
    float specular_obstruction = obstruction_smith(
            rough, normal, light_direction, eye, image_based_light);
    vec3 fresnel = fresnel_schlick(
        half_direction, eye, reflect_color, albedo, metal);
    
    float eye_incidence = max(dot(eye, normal), 0.);
    float light_incidence = max(dot(light_direction, normal), 0.);
    float denominator = 4 * eye_incidence * light_incidence;
    vec3 numerator = specular_distribution * fresnel * specular_obstruction;
    vec3 specular = numerator / max(denominator, 0.001);
    
    vec3 diffuse = (vec3(1.) - fresnel) * (1. - metal);
    float lambert = max(dot(normal, light_direction), 0.0);
    
    return (diffuse * albedo /* / PI*/ + specular) * light_color * lambert;
}
'''
