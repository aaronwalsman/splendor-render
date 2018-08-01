color_vertex_shader = '''#version 330 core

layout(location=0) in vec3 vertex_position;
layout(location=1) in vec3 vertex_normal;
layout(location=2) in vec2 vertex_uv;

out vec4 fragment_position;
out vec4 fragment_normal;
out vec2 fragment_uv;
out vec4 fragment_shadow_position;
out vec4 fragment_shadow_normal;
out vec4 fragment_shadow_projection;

uniform mat4 shadow_light_pose;
uniform mat4 shadow_light_projection;

uniform mat4 projection_matrix;
uniform mat4 model_pose;
uniform mat4 camera_pose;

void main(){
    
    mat4 vm = camera_pose * model_pose;
    mat4 pvm = projection_matrix * vm;
    
    mat4 lm = shadow_light_pose * model_pose;
    
    gl_Position = pvm * vec4(vertex_position,1);
    
    fragment_position = vm * vec4(vertex_position,1);
    fragment_normal = vm * vec4(vertex_normal,0);
    
    fragment_uv.x = vertex_uv.x;
    fragment_uv.y =-vertex_uv.y;
    
    fragment_shadow_position = lm * vec4(vertex_position,1);
    fragment_shadow_projection =
            shadow_light_projection * fragment_shadow_position;
    fragment_shadow_normal = lm * vec4(vertex_normal,0);
}
'''

color_fragment_shader = '''#version 330 core

const int MAX_NUM_LIGHTS = 8;

in vec4 fragment_position;
in vec4 fragment_normal;
in vec2 fragment_uv;
in vec4 fragment_shadow_position;
in vec4 fragment_shadow_projection;
in vec4 fragment_shadow_normal;

out vec3 color;

uniform vec4 material_properties;

uniform vec3 ambient_color;
uniform int num_point_lights;
uniform int num_direction_lights;
uniform int enable_shadow_light;
uniform vec3 shadow_light_color;
//uniform mat4 shadow_light_pose;
//uniform mat4 shadow_light_projection;
uniform vec3 point_light_data[2*MAX_NUM_LIGHTS];
uniform vec3 direction_light_data[2*MAX_NUM_LIGHTS];

uniform mat4 camera_pose;

uniform sampler2D texture_sampler;
uniform sampler2D shadow_sampler;

vec2 phong(
        vec3 contact_normal,
        vec3 light_direction,
        vec3 eye_direction,
        float shine){
    
    float diffuse = clamp(
            dot(-light_direction, contact_normal), 0, 1);
    
    vec3 reflected_direction = reflect(
            -light_direction, contact_normal);
    float specular = clamp(
            dot(-eye_direction, reflected_direction), 0, 1);
    if(specular > 0.0){
        specular = pow(specular, shine);
    }
    
    return vec2(diffuse, specular);
}

void main(){
    
    float ka = material_properties.x;
    float kd = material_properties.y;
    float ks = material_properties.z;
    float shine = material_properties.w;
    
    vec3 ambient_contribution = ambient_color;
    vec3 diffuse_contribution = vec3(0.0);
    vec3 specular_contribution = vec3(0.0);
    
    vec3 eye_direction = normalize(vec3(-fragment_position));
    
    vec3 fragment_normal_n = normalize(vec3(fragment_normal));
    
    /*
    if(enable_shadow_light){
        vec2 shadow_phong = phong(
                ...);
        diffuse_contribution += shadow_light_color * shadow_phong.x;
        specular_contribution += shadow_light_color * shadow_phong.y;
    }
    */
    
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
    
    color = vec3(
            ambient_color * texture_color * ka +
            diffuse_contribution * texture_color * kd +
            specular_contribution * ks);
}
'''

background_vertex_shader = '''#version 330 core

uniform mat4 projection_matrix;
uniform mat4 camera_pose;

out vec4 fragment_direction;

void main(){
    mat4 inv_vp = inverse(projection_matrix * camera_pose);
    float SMALL = 1-1e-5;
    if(gl_VertexID == 0){
        gl_Position = vec4(-1,-1,SMALL,1);
        //fragment_direction = vec4(-1,-1,-1,0);
    }
    else if(gl_VertexID == 1){
        gl_Position = vec4(-1, 1,SMALL,1);
        //fragment_direction = vec4(-1, 1,-1,0);
    }
    else if(gl_VertexID == 2){
        gl_Position = vec4( 1, 1,SMALL,1);
        //fragment_direction = vec4( 1, 1,-1,0);
    }
    else if(gl_VertexID == 3){
        gl_Position = vec4( 1,-1,SMALL,1);
        //fragment_direction = vec4( 1,-1,-1,0);
    }
    
    vec4 direction = gl_Position;
    //direction.z = 0.;
    direction.w = 0.;
    fragment_direction = inv_vp * direction;
}
'''

background_fragment_shader = '''#version 330 core
in vec4 fragment_direction;

out vec3 color;

uniform sampler2D texture_sampler;

#define M_PI 3.1415926535897932384626433832795

void main(){
    
    vec3 target_direction = vec3(0,0,1);
    
    
    vec2 uv;
    vec3 direction_n = normalize(vec3(fragment_direction));
    uv.y = -asin(fragment_direction.y)/(M_PI * 0.5);
    uv.y = (uv.y + 1) * 0.5;
    
    vec3 planar_n = direction_n;
    planar_n.y = 0.;
    planar_n = normalize(planar_n);
    //uv.x = atan(planar_n.x, planar_n.z);
    //uv.x = (uv.x / (2 * M_PI)) + 0.5;
    //uv.x = asin(planar_n.x)/(M_PI * 0.5);
    //uv.x = (uv.x + 1) * 0.5;
    
    //color = texture(texture_sampler, uv).rgb;
    //color = vec3(direction_n.x*0.5+0.5, direction_n.y*0.5+0.5, direction_n.z*0.5+0.5);
    
    if(dot(target_direction, direction_n) < 0.05){
        color = vec3(1,0,0);
    }
    else{
        color = vec3(0,0,1);
    }
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
