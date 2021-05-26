import OpenGL.GL as gl
import OpenGL.GL.shaders as shaders

from splendor.shaders.color_render import (
        textured_material_properties_vertex_shader,
        textured_material_properties_fragment_shader,
        textured_vertex_shader, textured_fragment_shader,
        vertex_color_vertex_shader, vertex_color_fragment_shader,
        flat_color_vertex_shader, flat_color_fragment_shader)
from splendor.shaders.mask_render import (
        mask_vertex_shader, mask_fragment_shader)
from splendor.shaders.coord_render import (
        coord_vertex_shader, coord_fragment_shader)
from splendor.shaders.background import (
        background_vertex_shader, background_fragment_shader)
from splendor.shaders.depthmap import (
        textured_depthmap_vertex_shader, textured_depthmap_fragment_shader)

default_shader_code = {
    'textured_material_properties_shader':(
        textured_material_properties_vertex_shader,
        textured_material_properties_fragment_shader,
    ),
    'textured_shader' :
        (textured_vertex_shader, textured_fragment_shader),
    'vertex_color_shader' :
        (vertex_color_vertex_shader, vertex_color_fragment_shader),
    'flat_color_shader' :
        (flat_color_vertex_shader, flat_color_fragment_shader),
    'mask_shader' :
        (mask_vertex_shader, mask_fragment_shader),
    'coord_shader' :
        (coord_vertex_shader, coord_fragment_shader),
    'background_shader' :
        (background_vertex_shader, background_fragment_shader),
    'textured_depthmap_shader' :
        (textured_depthmap_vertex_shader, textured_depthmap_fragment_shader)
}

class ShaderLibrary:
    def __init__(self, shader_code=None):
        if shader_code is None:
            shader_code = default_shader_code
        
        self.gl_data = {}
        for shader_name, (vertex_code, fragment_code) in shader_code.items():
            self.gl_data[shader_name] = {}
            
            # compile shaders
            vertex_shader = shaders.compileShader(
                    vertex_code, gl.GL_VERTEX_SHADER)
            fragment_shader = shaders.compileShader(
                    fragment_code, gl.GL_FRAGMENT_SHADER)
            self.gl_data[shader_name]['vertex_shader'] = vertex_shader
            self.gl_data[shader_name]['fragment_shader'] = fragment_shader
            
            # compile programs
            program = shaders.compileProgram(
                    self.gl_data[shader_name]['vertex_shader'],
                    self.gl_data[shader_name]['fragment_shader'])
            self.gl_data[shader_name]['program'] = program
            
            # get locations
            locations = {}
            num_attributes = gl.glGetProgramiv(program, gl.GL_ACTIVE_ATTRIBUTES)
            for i in range(num_attributes):
                attribute_name, _, _ = gl.glGetActiveAttrib(program, i)
                attribute_name = attribute_name.decode('utf-8')
                attribute_name = attribute_name.split('[')[0]
                location = gl.glGetAttribLocation(program, attribute_name)
                locations[attribute_name] = location
            num_uniforms = gl.glGetProgramiv(program, gl.GL_ACTIVE_UNIFORMS)
            for i in range(num_uniforms):
                uniform_name, _, _ = gl.glGetActiveUniform(program, i)
                uniform_name = uniform_name.decode('utf-8')
                uniform_name = uniform_name.split('[')[0]
                location = gl.glGetUniformLocation(program, uniform_name)
                locations[uniform_name] = location
            self.gl_data[shader_name]['locations'] = locations
            
            gl.glUseProgram(program)
            if 'texture_sampler' in locations:
                gl.glUniform1i(locations['texture_sampler'], 0)
            if 'diffuse_sampler' in locations:
                gl.glUniform1i(locations['diffuse_sampler'], 2)
            if 'reflect_sampler' in locations:
                gl.glUniform1i(locations['reflect_sampler'], 3)
            if 'cubemap_sampler' in locations:
                gl.glUniform1i(locations['cubemap_sampler'], 0)
    
    def get_shader_locations(self, shader):
        return self.gl_data[shader]['locations']
    
    def get_location(self, shader, location_name):
        return self.gl_data[shader]['locations'][location_name]
    
    def use_program(self, shader_name):
        gl.glUseProgram(self.gl_data[shader_name]['program'])
