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

