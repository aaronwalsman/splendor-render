import numpy as np

import OpenGL.GL as gl
import OpenGL.GL.shaders as shaders

from splendor.shaders.color_render import (
    textured_material_properties_vertex_shader,
    textured_material_properties_fragment_shader,
    textured_vertex_shader, textured_fragment_shader,
    vertex_color_vertex_shader, vertex_color_fragment_shader,
    flat_color_vertex_shader, flat_color_fragment_shader,
)
from splendor.shaders.mask_render import (
    mask_vertex_shader, mask_fragment_shader,
)
from splendor.shaders.coord_render import (
    coord_vertex_shader, coord_fragment_shader,
)
from splendor.shaders.background import (
    background_vertex_shader, background_fragment_shader,
)
from splendor.shaders.depthmap import (
    textured_depthmap_vertex_shader, textured_depthmap_fragment_shader,
)
from splendor.shaders.shadows import (
    depthmap_shadow_vertex_shader, depthmap_shadow_fragment_shader,
)
from splendor.shaders.radial_warp import (
    radial_warp_vertex_shader, radial_warp_fragment_shader,
)
from splendor.shaders.lines import (
    lines_vertex_shader, lines_fragment_shader,
)

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
        (textured_depthmap_vertex_shader, textured_depthmap_fragment_shader),
    'depthmap_shadow_shader' :
        (depthmap_shadow_vertex_shader, depthmap_shadow_fragment_shader),
    'radial_warp_shader' :
        (radial_warp_vertex_shader, radial_warp_fragment_shader),
    'lines_shader' :
        (lines_vertex_shader, lines_fragment_shader),
}

def gl_name_to_str(gl_name):
    if isinstance(gl_name, np.ndarray):
        gl_name = gl_name.tobytes()
    str_name = gl_name.decode('utf-8')
    str_name = str_name.split('[')[0]
    str_name = str_name.rstrip('\x00')
    return str_name

class ShaderLibrary:
    def __init__(self, shader_code=None):
        if shader_code is None:
            shader_code = default_shader_code
        
        tmp_vao = gl.glGenVertexArrays(1)
        tmp_image = np.zeros((16,16,3), dtype=np.uint8)
        tex0 = gl.glGenTextures(1)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, tex0)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGB,
            tmp_image.shape[1],
            tmp_image.shape[0],
            0,
            gl.GL_RGB,
            gl.GL_UNSIGNED_BYTE,
            tmp_image,
        )
        
        tex1 = gl.glGenTextures(1)
        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, tex1)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGB,
            tmp_image.shape[1],
            tmp_image.shape[0],
            0,
            gl.GL_RGB,
            gl.GL_UNSIGNED_BYTE,
            tmp_image,
        )
        
        tex2 = gl.glGenTextures(1)
        gl.glActiveTexture(gl.GL_TEXTURE2)
        gl.glBindTexture(gl.GL_TEXTURE_CUBE_MAP, tex2)
        for i in range(6):
            gl.glTexImage2D(
                gl.GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                0,
                gl.GL_RGB,
                16,
                16,
                0,
                gl.GL_RGB,
                gl.GL_UNSIGNED_BYTE,
                tmp_image,
            )
        
        tex3 = gl.glGenTextures(1)
        gl.glActiveTexture(gl.GL_TEXTURE3)
        gl.glBindTexture(gl.GL_TEXTURE_CUBE_MAP, tex3)
        for i in range(6):
            gl.glTexImage2D(
                gl.GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                0,
                gl.GL_RGB,
                16,
                16,
                0,
                gl.GL_RGB,
                gl.GL_UNSIGNED_BYTE,
                tmp_image,
            )
        
        
        self.gl_data = {}
        for shader_name, (vertex_code, fragment_code) in shader_code.items():
            print('compiling', shader_name)
            self.gl_data[shader_name] = {}
            
            # compile shaders
            vertex_shader = shaders.compileShader(
                    vertex_code, gl.GL_VERTEX_SHADER)
            fragment_shader = shaders.compileShader(
                    fragment_code, gl.GL_FRAGMENT_SHADER)
            self.gl_data[shader_name]['vertex_shader'] = vertex_shader
            self.gl_data[shader_name]['fragment_shader'] = fragment_shader
            
            gl.glBindVertexArray(tmp_vao)
            
            # compile programs
            try:
                program = shaders.compileProgram(
                    self.gl_data[shader_name]['vertex_shader'],
                    self.gl_data[shader_name]['fragment_shader'],
                    validate=False,
                )
            except:
                print(f'Compiling {shader_name} failed.')
                #breakpoint()
                raise
            else:
                print(f'Compiled {shader_name} successfully.')
                #breakpoint()
            
            gl.glBindVertexArray(0)
            
            self.gl_data[shader_name]['program'] = program
            
            # get locations
            locations = {}
            num_attributes = gl.glGetProgramiv(program, gl.GL_ACTIVE_ATTRIBUTES)
            for i in range(num_attributes):
                attribute_name, _, _ = gl.glGetActiveAttrib(program, i)
                #attribute_name = attribute_name.decode('utf-8')
                #attribute_name = attribute_name.split('[')[0]
                attribute_name = gl_name_to_str(attribute_name)
                location = gl.glGetAttribLocation(program, attribute_name)
                locations[attribute_name] = location
            num_uniforms = gl.glGetProgramiv(program, gl.GL_ACTIVE_UNIFORMS)
            for i in range(num_uniforms):
                uniform_name, _, _ = gl.glGetActiveUniform(program, i)
                #uniform_name = uniform_name.decode('utf-8')
                #uniform_name = uniform_name.split('[')[0]
                uniform_name = gl_name_to_str(uniform_name)
                location = gl.glGetUniformLocation(program, uniform_name)
                locations[uniform_name] = location
            self.gl_data[shader_name]['locations'] = locations
            
            gl.glUseProgram(program)
            if 'texture_sampler' in locations:
                gl.glUniform1i(locations['texture_sampler'], 0)
            if 'reflect_sampler' in locations:
                gl.glUniform1i(locations['reflect_sampler'], 3)
            if 'cubemap_sampler' in locations:
                gl.glUniform1i(locations['cubemap_sampler'], 0)
            from splendor.shaders.lighting_model import MAX_SHADOW_CASTERS
            for _i in range(MAX_SHADOW_CASTERS):
                _name = f'shadow_depth_sampler_{_i}'
                if _name in locations:
                    gl.glUniform1i(locations[_name], 4 + _i)
            if 'intermediate_sampler' in locations:
                gl.glUniform1i(locations['intermediate_sampler'], 5)
            if 'intermediate_depth_sampler' in locations:
                gl.glUniform1i(locations['intermediate_depth_sampler'], 6)
        
        gl.glDeleteVertexArrays(1, [tmp_vao])
    
    def get_shader_locations(self, shader):
        return self.gl_data[shader]['locations']
    
    def get_location(self, shader, location_name):
        return self.gl_data[shader]['locations'][location_name]
    
    def use_program(self, shader_name):
        gl.glUseProgram(self.gl_data[shader_name]['program'])

    def get_program_id(self, shader_name):
        return self.gl_data[shader_name]['program']
