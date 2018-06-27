
# Sources

# https://github.com/BerkeleyAutomation/meshrender
# (I didn't use this because it has too many dependencies
# and no textured meshes)

# https://github.com/rbmj/SimpleRender
# (I didn't use this because it is too simple for my needs)

# system
from ctypes import sizeof, c_float, c_uint
import json

# opengl
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.arrays import vbo
from OpenGL.GL import shaders
from OpenGL.arrays import vbo

# scipy/numpy
import numpy
import scipy.misc

# local
import renderpy_shaders
import obj_mesh

max_num_lights = 8

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self.obj)

class Renderpy:
    
    def __init__(self,
            width = 256,
            height = 256):
        
        # scene data
        self.scene_description = {
                'meshes':{},
                'materials':{},
                'instances':{},
                'ambient_color':numpy.array([0,0,0]),
                'point_lights':{},
                'direction_lights':{},
                'camera':{
                    'pose':numpy.array([
                        [1,0,0,0],
                        [0,1,0,0],
                        [0,0,1,0],
                        [0,0,0,1]]),
                    'projection':numpy.array([
                        [1,0,0,0],
                        [0,1,0,0],
                        [0,0,1,0],
                        [0,0,0,1]])}
        }
        
        self.loaded_data = {
                'meshes':{},
                'textures':{},
        }
        
        self.gl_data = {
                'mesh_buffers':{},
                'material_buffers':{},
                'color_shader':{},
                'mask_shader':{}
        }
        
        self.width = width
        self.height = height
        
        self.opengl_init()
        self.compile_shaders()
    
    
    def get_json_description(self, **kwargs):
        return json.dumps(self.scene_description, cls=NumpyEncoder, **kwargs)
    
    
    def opengl_init(self):
        
        renderer = glGetString(GL_RENDERER).decode('utf-8')
        version = glGetString(GL_VERSION).decode('utf-8')
        print('Renderer: %s'%renderer)
        print('OpenGL Version: %s'%version)
        
        glEnable(GL_DEPTH_TEST)
        glDepthMask(GL_TRUE)
        glDepthFunc(GL_LESS)
        glDepthRange(0.0, 1.0)
        glEnable(GL_NORMALIZE)

        glClearColor(0.,0.,0.,0.)
    
    
    def compile_shaders(self):
        
        # book keeping
        color_shader_data = {}
        mask_shader_data = {}
        
        # compile the shaders
        color_shader_data['vertex_shader'] = shaders.compileShader(
                renderpy_shaders.color_vertex_shader, GL_VERTEX_SHADER)
        color_shader_data['fragment_shader'] = shaders.compileShader(
                renderpy_shaders.color_fragment_shader, GL_FRAGMENT_SHADER)
        mask_shader_data['vertex_shader'] = shaders.compileShader(
                renderpy_shaders.mask_vertex_shader, GL_VERTEX_SHADER)
        mask_shader_data['fragment_shader'] = shaders.compileShader(
                renderpy_shaders.mask_fragment_shader, GL_FRAGMENT_SHADER)
        
        # compile the programs
        color_program = shaders.compileProgram(
                color_shader_data['vertex_shader'],
                color_shader_data['fragment_shader'])
        color_shader_data['program'] = color_program
        mask_program = shaders.compileProgram(
                mask_shader_data['vertex_shader'],
                mask_shader_data['fragment_shader'])
        mask_shader_data['program'] = mask_program
        
        # get attribute locations
        color_shader_data['locations'] = {}
        mask_shader_data['locations'] = {}
        
        # (position/normal/uv)
        color_shader_data['locations']['vertex_position'] = (
                glGetAttribLocation(color_program, 'vertex_position'))
        color_shader_data['locations']['vertex_normal'] = (
                glGetAttribLocation(color_program, 'vertex_normal'))
        color_shader_data['locations']['vertex_uv'] = (
                glGetAttribLocation(color_program, 'vertex_uv'))
        
        mask_shader_data['locations']['vertex_position'] = (
                glGetAttribLocation(mask_program, 'vertex_position'))
        mask_shader_data['locations']['vertex_normal'] = (
                glGetAttribLocation(mask_program, 'vertex_normal'))
        mask_shader_data['locations']['vertex_uv'] = (
                glGetAttribLocation(mask_program, 'vertex_uv'))
        
        # (pose and projection matrices)
        for variable in 'camera_pose', 'projection_matrix', 'model_pose':
            color_shader_data['locations'][variable] = (
                    glGetUniformLocation(color_program, variable))
            mask_shader_data['locations'][variable] = (
                    glGetUniformLocation(mask_program, variable))
        
        # (material data)
        color_shader_data['locations']['material_properties'] = (
                glGetUniformLocation(color_program, 'material_properties'))
        
        # (light data)
        for variable in (
                'ambient_color',
                'num_point_lights',
                'num_direction_lights',
                'point_light_data',
                'direction_light_data'):
            color_shader_data['locations'][variable] = (
                    glGetUniformLocation(color_program, variable))
        
        # (mask data)
        mask_shader_data['locations']['mask_color'] = (
                glGetUniformLocation(mask_program, 'mask_color'))
        
        self.gl_data['color_shader'] = color_shader_data
        self.gl_data['mask_shader'] = mask_shader_data
        
    
    def load_scene(self, scene, clear_existing=False):
    
        if clear_existing:
            self.clear_scene()
        
        for mesh in scene['meshes']:
            self.load_mesh(mesh, **scene['meshes'][mesh])
        
        for material in scene['materials']:
            self.load_material(material, **scene['materials'][material])
        
        for instance in scene['instances']:
            self.add_instance(instance, **scene['instances'][instance])
        
        if 'ambient_color' in scene:
            self.set_ambient_color(scene['ambient_color'])
        
        for point_light in scene['point_lights']:
            self.add_point_light(
                    point_light,
                    **scene['point_lights'][point_light])
        
        for direction_light in scene['direction_lights']:
            self.add_direction_light(
                    direction_light,
                    **scene['direction_lights'][direction_light])
        
        if 'camera' in scene:
            if 'pose' in scene['camera']:
                self.move_camera(scene['camera']['pose'])
            if 'projection' in scene['camera']:
                self.set_projection(scene['camera']['projection'])
    
    
    def clear_scene(self):
        self.clear_meshes()
        self.clear_materials()
        self.clear_instances()
        self.set_ambient_color([0,0,0])
        self.clear_point_lights()
        self.clear_direction_lights()
        self.reset_camera()
    
    
    def reset_camera(self):
        self.scene_description['camera'] = {
                'pose':numpy.array([
                    [1,0,0,0],
                    [0,1,0,0],
                    [0,0,1,0],
                    [0,0,0,1]]),
                'projection':numpy.array([
                    [1,0,0,0],
                    [0,1,0,0],
                    [0,0,1,0],
                    [0,0,0,1]])}
    
    
    def load_mesh(self,
            name,
            mesh_path):#,
            #texture=None,
            #ka=1.0,
            #kd=1.0,
            #ks=0.5,
            #shine=4.0):
        
        mesh = obj_mesh.load_mesh(mesh_path)
        self.scene_description['meshes'][name] = {'mesh_path':mesh_path}
        
        #self.scene_description['materials'][name] = numpy.array([
        #        ka, kd, ks, shine])
        mesh_buffers = {}
        
        vertex_floats = numpy.array(mesh['vertices'], dtype=numpy.float32)
        normal_floats = numpy.array(mesh['normals'], dtype=numpy.float32)
        uv_floats = numpy.array(mesh['uvs'], dtype=numpy.float32)
        combined_floats = numpy.concatenate(
                (vertex_floats, normal_floats, uv_floats), axis = 1)
        mesh_buffers['vertex_buffer'] = vbo.VBO(combined_floats)
        
        face_ints = numpy.array(mesh['faces'], dtype=numpy.int32)
        mesh_buffers['face_buffer'] = vbo.VBO(
                face_ints,
                target = GL_ELEMENT_ARRAY_BUFFER)
        
        '''
        if texture is not None:
            image = scipy.misc.imread(texture)[:,:,:3]
            
            mesh_buffers['texture'] = {}
            mesh_buffers['texture']['id'] = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, mesh_buffers['texture']['id'])
            glTexImage2D(
                    GL_TEXTURE_2D, 0, GL_RGB,
                    image.shape[0], image.shape[1], 0,
                    GL_RGB, GL_UNSIGNED_BYTE, image)
            # GL_NEAREST?
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                    GL_LINEAR_MIPMAP_LINEAR)
            glGenerateMipmap(GL_TEXTURE_2D)
            
            self.loaded_data['textures'][name] = image
        '''
        self.loaded_data['meshes'][name] = mesh
        self.gl_data['mesh_buffers'][name] = mesh_buffers
    
    
    def remove_mesh(self, name):
        del(self.scene_description['mesh_paths'][name])
        self.gl_data['mesh_buffers'][name]['vertex_buffer'].delete()
        self.gl_data['mesh_buffers'][name]['face_buffer'].delete()
        del(self.gl_data['mesh_buffers'][name])
        del(self.loaded_data['meshes'][name])
    
    
    def clear_meshes(self):
        for name in self.scene_description['meshes']:
            self.remove_mesh(name)
    
    
    def load_material(self,
            name,
            texture,
            ka = 1.0,
            kd = 1.0,
            ks = 0.5,
            shine = 4.0):
        
        self.scene_description['materials'][name] = {
                'texture' : texture,
                'ka' : ka,
                'kd' : kd,
                'ks' : ks,
                'shine' : shine}
        
        image = scipy.misc.imread(texture)[:,:,:3]
        self.loaded_data['textures'][name] = image
        
        material_buffers = {}
        material_buffers['texture'] = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, material_buffers['texture'])
        glTexImage2D(
                GL_TEXTURE_2D, 0, GL_RGB,
                image.shape[0], image.shape[1], 0,
                GL_RGB, GL_UNSIGNED_BYTE, image)
        
        # GL_NEAREST?
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                GL_LINEAR_MIPMAP_LINEAR)
        glGenerateMipmap(GL_TEXTURE_2D)
        
        self.gl_data['material_buffers'][name] = material_buffers
    
    
    def remove_material(self, name):
        glDeleteTextures(1, self.gl_data['material_buffers'][name]['texture'])
        del(self.loaded_data['textures'][name])
        del(self.gl_data['material_buffers'][name])
        del(self.scene_description['materials'][name])
    
    
    def clear_materials(self):
        for name in self.scene_description['materials']:
            self.remove_material(name)
    
    def add_instance(self,
            instance_name,
            mesh_name,
            material_name,
            transform,
            mask_color = numpy.array([0,0,0])):
        
        instance_data = {}
        instance_data['mesh_name'] = mesh_name
        instance_data['material_name'] = material_name
        instance_data['transform'] = numpy.array(transform)
        instance_data['mask_color'] = numpy.array(mask_color)
        self.scene_description['instances'][instance_name] = instance_data
    
    def remove_instance(self, instance_name):
        del(self.scene_description['instances'][instance_name])
    
    def clear_instances(self):
        self.scene_description['instances'] = {}
    
    def add_point_light(self, name, position, color):
        self.scene_description['point_lights'][name] = {
                'position' : numpy.array(position),
                'color' : numpy.array(color)}
    
    def remove_point_light(self, name):
        del(self.scene_description['point_lights'][name])
    
    def clear_point_lights(self):
        self.scene_description['point_lights'] = {}
    
    def add_direction_light(self, name, direction, color):
        self.scene_description['direction_lights'][name] = {
                'direction' : numpy.array(direction),
                'color' : numpy.array(color)}
    
    def remove_direction_light(self, name):
        del(self.scene_description['direction_lights'][name])
    
    def clear_direction_lights(self):
        self.scene_description['direction_lights'] = {}
    
    def set_ambient_color(self, color):
        self.scene_description['ambient_color'] = numpy.array(color)
    
    def set_projection(self, projection_matrix):
        self.scene_description['camera']['projection'] = numpy.array(
                projection_matrix)
    
    def move_camera(self, camera_pose):
        self.scene_description['camera']['pose'] = numpy.array(
                camera_pose)
    
    def color_render(self):
        
        # clear
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # turn on the color shader
        glUseProgram(self.gl_data['color_shader']['program'])
        
        try:
            
            location_data = self.gl_data['color_shader']['locations']
            
            # set the camera's pose
            camera_pose = self.scene_description['camera']['pose']
            glUniformMatrix4fv(
                    location_data['camera_pose'],
                    1, GL_TRUE,
                    camera_pose.astype(numpy.float32))
            
            # set the camera's projection matrix
            projection_matrix = self.scene_description['camera']['projection']
            glUniformMatrix4fv(
                    location_data['projection_matrix'],
                    1, GL_TRUE,
                    projection_matrix.astype(numpy.float32))
            
            # set the ambient light's color
            ambient_color = self.scene_description['ambient_color']
            glUniform3fv(
                    location_data['ambient_color'], 1,
                    ambient_color.astype(numpy.float32))
            
            # set the point light data
            glUniform1i(
                    location_data['num_point_lights'],
                    len(self.scene_description['point_lights']))
            point_light_data = numpy.zeros((max_num_lights*2,3))
            for i, light_name in enumerate(
                    self.scene_description['point_lights']):
                light_data = self.scene_description[
                        'point_lights'][light_name]
                point_light_data[i*2] = light_data['color']
                point_light_data[i*2+1] = light_data['position']
            glUniform3fv(
                    location_data['point_light_data'], max_num_lights*2,
                    point_light_data.astype(numpy.float32))
            
            # set the direction light data
            glUniform1i(
                    location_data['num_direction_lights'],
                    len(self.scene_description['direction_lights']))
            direction_light_data = numpy.zeros((max_num_lights*2,3))
            for i, light_name in enumerate(
                    self.scene_description['direction_lights']):
                light_data = self.scene_description[
                        'direction_lights'][light_name]
                direction_light_data[i*2] = light_data['color']
                direction_light_data[i*2+1] = light_data['direction']
            glUniform3fv(
                    location_data['direction_light_data'], max_num_lights*2,
                    direction_light_data.astype(numpy.float32))
                
            
            # render all instances
            for instance_name in self.scene_description['instances']:
                self.color_render_instance(instance_name)
            
        finally:
            glUseProgram(0)
        
        glFinish()
    
    def color_render_instance(self, instance_name):
        instance_data = self.scene_description['instances'][instance_name]
        instance_mesh = instance_data['mesh_name']
        instance_material = instance_data['material_name']
        material_data = (
                self.scene_description['materials'][instance_material])
        material_properties = numpy.array([
                material_data['ka'],
                material_data['kd'],
                material_data['ks'],
                material_data['shine']])
        mesh_buffers = self.gl_data['mesh_buffers'][instance_mesh]
        material_buffers = self.gl_data['material_buffers'][instance_material]
        num_triangles = len(self.loaded_data['meshes'][instance_mesh]['faces'])
        
        location_data = self.gl_data['color_shader']['locations']
        
        glUniformMatrix4fv(
                location_data['model_pose'],
                1, GL_TRUE,
                instance_data['transform'].astype(numpy.float32))
        
        glUniform4fv(
                location_data['material_properties'],
                1, material_properties.astype(numpy.float32))
        
        mesh_buffers['face_buffer'].bind()
        mesh_buffers['vertex_buffer'].bind()
        
        glBindTexture(GL_TEXTURE_2D, material_buffers['texture'])
        
        try:
            
            glEnableVertexAttribArray(location_data['vertex_position'])
            glEnableVertexAttribArray(location_data['vertex_normal'])
            glEnableVertexAttribArray(location_data['vertex_uv'])
            
            stride = (3+3+2) * 4
            glVertexAttribPointer(
                    location_data['vertex_position'],
                    3, GL_FLOAT, False, stride,
                    mesh_buffers['vertex_buffer'])
            glVertexAttribPointer(
                    location_data['vertex_normal'],
                    3, GL_FLOAT, False, stride,
                    mesh_buffers['vertex_buffer']+((3)*4))
            glVertexAttribPointer(
                    location_data['vertex_uv'],
                    2, GL_FLOAT, False, stride,
                    mesh_buffers['vertex_buffer']+((3+3)*4))
            
            glDrawElements(
                    GL_TRIANGLES,
                    num_triangles*3,
                    GL_UNSIGNED_INT,
                    None)
        
        finally:
            mesh_buffers['face_buffer'].unbind()
            mesh_buffers['vertex_buffer'].unbind()
    
    
    def mask_render(self):
        
        # clear
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # tun on the shader
        glUseProgram(self.gl_data['mask_shader']['program'])
        
        try:
            
            location_data = self.gl_data['mask_shader']['locations']
            
            # set the camera's pose
            camera_pose = self.scene_description['camera']['pose']
            glUniformMatrix4fv(
                    location_data['camera_pose'],
                    1, GL_TRUE,
                    camera_pose.astype(numpy.float32))
            
            # set the camera's projection matrix
            projection_matrix = self.scene_description['camera']['projection']
            glUniformMatrix4fv(
                    location_data['projection_matrix'],
                    1, GL_TRUE,
                    projection_matrix.astype(numpy.float32))
            
            # render all instances
            for instance_name in self.scene_description['instances']:
                self.mask_render_instance(instance_name)
            
        finally:
            glUseProgram(0)
        
        glFinish()
    
    
    def mask_render_instance(self, instance_name):
        instance_data = self.scene_description['instances'][instance_name]
        instance_mesh = instance_data['mesh_name']
        mask_color = instance_data['mask_color']
        mesh_buffers = self.gl_data['mesh_buffers'][instance_mesh]
        mesh = self.loaded_data['meshes'][instance_mesh]
        
        location_data = self.gl_data['mask_shader']['locations']
        
        glUniformMatrix4fv(
                location_data['model_pose'],
                1, GL_TRUE,
                instance_data['transform'].astype(numpy.float32))
        
        glUniform3fv(
                location_data['mask_color'],
                1, mask_color.astype(numpy.float32))
        
        mesh_buffers['face_buffer'].bind()
        mesh_buffers['vertex_buffer'].bind()
        
        try:
            
            glEnableVertexAttribArray(location_data['vertex_position'])
            
            stride = (3+3+2) * 4
            glVertexAttribPointer(
                    location_data['vertex_position'],
                    3, GL_FLOAT, False, stride,
                    mesh_buffers['vertex_buffer'])
            
            glDrawElements(
                    GL_TRIANGLES,
                    len(mesh['faces'])*3,
                    GL_UNSIGNED_INT,
                    None)
        
        finally:
            mesh_buffers['face_buffer'].unbind()
            mesh_buffers['vertex_buffer'].unbind()
