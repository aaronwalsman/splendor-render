from splendor.shaders.utils import phong_fn
from splendor.shaders.pbr import pbr_fns
from splendor.shaders.skybox import skybox_fn
from splendor.shaders.utils import softish_step_fn

MAX_SHADOW_CASTERS = 4


def _build_shadow_samplers():
    return '\n'.join(
        f'uniform sampler2D shadow_depth_sampler_{i};'
        for i in range(MAX_SHADOW_CASTERS)
    )


def _build_sample_shadow_slot():
    lines = ['float sample_shadow_slot(int slot, vec2 uv) {']
    for i in range(MAX_SHADOW_CASTERS):
        kw = 'if' if i == 0 else 'else if'
        lines.append(f'    {kw}(slot == {i})')
        lines.append(f'        return texture(shadow_depth_sampler_{i}, uv).r;')
    lines.append('    return 1.0;')
    lines.append('}')
    return '\n'.join(lines)


def _build_get_shadow_texel():
    lines = ['vec2 get_shadow_texel(int slot) {']
    for i in range(MAX_SHADOW_CASTERS):
        kw = 'if' if i == 0 else 'else if'
        lines.append(f'    {kw}(slot == {i})')
        lines.append(
            f'        return 1.0 / vec2(textureSize(shadow_depth_sampler_{i}, 0));')
    lines.append('    return vec2(1.0 / 1024.0);')
    lines.append('}')
    return '\n'.join(lines)


lighting_model_fragment_shader = f'''
const int MAX_NUM_LIGHTS = 8;
const int MAX_SHADOW_CASTERS = {MAX_SHADOW_CASTERS};

in vec4 fragment_position;
in vec4 fragment_normal;

#ifdef COMPILE_TEXTURE
in vec2 fragment_uv;
#endif

#ifdef COMPILE_VERTEX_COLORS
in vec3 fragment_color;
#endif

out vec4 color;

#ifndef COMPILE_TEXURED_MATERIAL_PROPERTIES
uniform vec4 material_properties;
#endif

uniform vec4 image_light_properties;
uniform bool image_light_active;
uniform vec3 background_color;

#ifdef COMPILE_FLAT_COLOR
uniform vec3 flat_color;
#endif

uniform vec3 ambient_color;
uniform int num_point_lights;
uniform int num_direction_lights;

uniform mat4 image_light_offset_matrix;
uniform bool lock_image_light_to_camera;

uniform vec3 point_light_data[2*MAX_NUM_LIGHTS];
uniform vec3 direction_light_data[2*MAX_NUM_LIGHTS];

uniform mat4 view_matrix;

// Shadow maps: shared pool of {MAX_SHADOW_CASTERS} slots used by any light type
uniform mat4 shadow_view_matrices[MAX_SHADOW_CASTERS];
uniform mat4 shadow_projection_matrices[MAX_SHADOW_CASTERS];
uniform int  shadow_pcf_radii[MAX_SHADOW_CASTERS];
uniform int  direction_light_shadow_slots[MAX_NUM_LIGHTS]; // -1 = no shadow
uniform int  image_light_shadow_slot;                      // -1 = no shadow
uniform vec3 image_light_shadow_direction;                 // world-space, surface→light
uniform vec3 image_light_shadow_color;                     // min irradiance, used as shadow floor

#ifdef COMPILE_TEXTURE
uniform sampler2D texture_sampler;
#endif

#ifdef COMPILE_TEXTURED_MATERIAL_PROPERTIES
uniform sampler2D material_properties_sampler;
#endif

uniform vec3 irradiance_sh[9];
uniform samplerCube reflect_sampler;

{_build_shadow_samplers()}

const float MAX_MIPMAP = 4.;

''' + f'''
{pbr_fns}
{skybox_fn}

{_build_sample_shadow_slot()}

{_build_get_shadow_texel()}
''' + '''

float shadow_lookup(int slot, vec4 world_pos, vec3 normal, vec3 light_dir) {
    vec4 shadow_clip =
        shadow_projection_matrices[slot] * shadow_view_matrices[slot] * world_pos;
    vec3 shadow_p = shadow_clip.xyz / shadow_clip.w * 0.5 + 0.5;
    if(shadow_p.x < 0.0 || shadow_p.x > 1.0 ||
       shadow_p.y < 0.0 || shadow_p.y > 1.0 ||
       shadow_p.z <= 0.0 || shadow_p.z >= 1.0)
        return 0.0;
    float bias = max(0.002 * (1.0 - dot(normal, light_dir)), 0.0002);
    int r = shadow_pcf_radii[slot];
    vec2 texel = get_shadow_texel(slot);
    float sum = 0.0, count = 0.0;
    for(int sx = -r; sx <= r; ++sx) {
        for(int sy = -r; sy <= r; ++sy) {
            float d = sample_shadow_slot(slot, shadow_p.xy + vec2(sx, sy) * texel);
            sum += (shadow_p.z - bias > d) ? 1.0 : 0.0;
            count += 1.0;
        }
    }
    return sum / count;
}

void main(){

    // material properties =====================================================
#ifdef COMPILE_TEXTURED_MATERIAL_PROPERTIES
    vec4 material_properties = texture(
        material_properties_sampler, fragment_uv);
#endif
    float metal = material_properties.x;
    float rough = material_properties.y;
    float base_reflect = material_properties.z;
    float ambient = material_properties.w;

    float diffuse_scale = image_light_properties.x;
    float diffuse_bias = image_light_properties.y;
    float reflect_gamma = image_light_properties.z;
    float reflect_bias = image_light_properties.w;

    mat4 inv_view_matrix = inverse(view_matrix);
    vec4 world_position = inv_view_matrix * fragment_position;

    vec3 eye = normalize(vec3(-fragment_position));
    vec3 normal = normalize(vec3(fragment_normal));
    vec3 camera_normal = normal;
    if(!lock_image_light_to_camera){
        camera_normal = vec3(inv_view_matrix * vec4(camera_normal, 0.));
    }

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

        vec3 light_color = point_light_data[2*i];
        vec3 light_position = point_light_data[2*i+1];
        light_position = vec3(view_matrix * vec4(light_position, 1.));
        vec3 light_direction = light_position - vec3(fragment_position);
        light_direction = normalize(light_direction);

        vec3 half_direction = normalize(eye + light_direction);

        vec3 light_contribution = cook_torrance(
            rough, metal, f0, albedo, eye, normal,
            half_direction, light_direction, light_color);

        color += vec4(light_contribution, 0.);
    }

    // direction lights ========================================================
    for(int i = 0; i < num_direction_lights; ++i){

        vec3 light_color = vec3(direction_light_data[2*i]);
        vec3 light_direction = -normalize(direction_light_data[2*i+1]);
        light_direction = vec3(view_matrix * vec4(light_direction, 0.));
        vec3 half_direction = normalize(eye + light_direction);

        vec3 light_contribution = cook_torrance(
            rough, metal, f0, albedo, eye, normal,
            half_direction, light_direction, light_color);

        float in_shadow = 0.0;
        int shadow_slot = direction_light_shadow_slots[i];
        if(shadow_slot >= 0) {
            in_shadow = shadow_lookup(shadow_slot, world_position, normal, light_direction);
        }

        color += vec4(light_contribution, 0.) * (1.0 - in_shadow);
    }

    // reflect =================================================================
    float cos_theta = dot(normal, eye);
    vec3 ks = fresnel_schlick_rough(cos_theta, f0, rough);
    vec3 kd = (1. - ks) * (1. - metal);

    // image light =============================================================
    if(image_light_active){

        vec3 offset_fragment_normal = vec3(
                image_light_offset_matrix * vec4(camera_normal, 1.));
        vec3 offset_n = normalize(offset_fragment_normal);

        vec3 diffuse_color = max(
            irradiance_sh[0]
            + irradiance_sh[1] * offset_n.y
            + irradiance_sh[2] * offset_n.z
            + irradiance_sh[3] * offset_n.x
            + irradiance_sh[4] * offset_n.x * offset_n.y
            + irradiance_sh[5] * offset_n.y * offset_n.z
            + irradiance_sh[6] * (3.0*offset_n.z*offset_n.z - 1.0)
            + irradiance_sh[7] * offset_n.x * offset_n.z
            + irradiance_sh[8] * (offset_n.x*offset_n.x - offset_n.y*offset_n.y),
            vec3(0.0));
        diffuse_color = diffuse_color * diffuse_scale + vec3(diffuse_bias);

        vec4 reflected_direction =
                inv_view_matrix * vec4(reflect(-eye, normal), 0.);
        reflected_direction = image_light_offset_matrix * reflected_direction;
        vec3 reflect_color = vec3(skybox_texture(
                reflect_sampler, reflected_direction, rough*MAX_MIPMAP));
        reflect_color = pow(reflect_color, vec3(reflect_gamma));

        float reflect_correction = (reflect_gamma+1)/2;
        reflect_color *= reflect_correction;
        reflect_color += vec3(reflect_bias);
        reflect_color = reflect_color * ks;

        vec3 shadow_floor = image_light_shadow_color * diffuse_scale
            + vec3(diffuse_bias);

        float ibl_shadow = 0.0;
        if(image_light_shadow_slot >= 0) {
            vec3 ibl_light_dir = normalize(
                vec3(view_matrix * vec4(image_light_shadow_direction, 0.0)));
            float n_dot_l = dot(normal, ibl_light_dir);
            if(n_dot_l > 0.0) {
                ibl_shadow = shadow_lookup(
                    image_light_shadow_slot, world_position,
                    normal, ibl_light_dir);
            } else {
                // Back face: treat as fully in shadow
                ibl_shadow = 1.0;
            }
        }

        vec3 shadowed_diffuse = mix(diffuse_color, shadow_floor, ibl_shadow);

        color += vec4(kd * shadowed_diffuse * albedo, 0.);
        color += vec4(reflect_color, 0.) * (1.0 - ibl_shadow);
    }

    // ambient and background ==================================================
    color += vec4(kd * ambient_color * ambient * albedo, 0.);
    color += vec4(ks * background_color, 0.);
}
'''
