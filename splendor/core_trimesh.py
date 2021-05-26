import math
import json
import os

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GL import shaders
from OpenGL.arrays import vbo

import numpy

import trimesh

from renderpy.assets import AssetLibrary
import renderpy.camera as camera
import renderpy.masks as masks
import renderpy.shaders.shader_definitions as shader_definitions
import renderpy.obj_mesh as obj_mesh
from renderpy.image import load_image, load_depth
import renderpy.json_numpy as json_numpy
from renderpy.exceptions import RenderpyException

max_num_lights = 8
default_default_camera_pose = numpy.eye(4)
default_default_camera_projection = camera.projection_matrix(
        math.radians(90.), 1.0)

class Renderpy:
    
    global_parameters = (
            'ambient_color', 'background_color', 'active_image_light')
    asset_types = (
            ('mesh', 'meshes'),
            ('material', 'materials'),
            ('image_light', 'image_lights'),
            ('depthmap', 'depthmaps'))
    instance_types = (
            ('instance', 'instances'),
            ('depthmap_instance', 'depthmap_instances'),
            ('point_light', 'point_lights'))
    
    def __init__(self,
            assets=None,
            default_camera_pose=None,
            default_camera_projection=None):
        
        # asset library
        if isinstance(assets, AssetLibrary):
            self.asset_library = assets
        else:
            self.asset_library = AssetLibrary(assets)
        
        # default camera settings
        if default_camera_pose is None:
            default_camera_pose = default_default_camera_pose
        self.default_camera_pose = default_camera_pose
        if default_camera_projection is None:
            default_camera_projection = default_default_camera_projection
        self.default_camera_projection = default_camera_projection
        
        # scene data
        self.scene_description = {
                'meshes':{},
                'depthmaps':{},
                'materials':{},
                'instances':{},
                'depthmap_instances':{},
                'background_color':numpy.array([0,0,0,0]),
                'ambient_color':numpy.array([0,0,0]),
                'point_lights':{},
                'direction_lights':{},
                'camera':{
                    'pose':default_camera_pose,
                    'projection':default_camera_projection,
                },
                'image_lights':{},
                'active_image_light':None
        }
        
        self.loaded_data = {
                'meshes':{},
                'depthmaps':{},
                'textures':{},
        }
        
        self.gl_data = {
                'mesh_buffers':{},
                'depthmap_buffers':{},
                'material_buffers':{},
                'light_buffers':{},
                'textured_shader':{},
                'vertex_color_shader':{},
                'mask_shader':{},
                'coord_shader':{},
                'background_shader':{}
        }
        
        self.opengl_init()
        self.compile_shaders()
    
    def get_json_description(self, **kwargs):
        return json.dumps(
                self.scene_description, cls=json_numpy.NumpyEncoder, **kwargs)
    
    def opengl_init(self):
        renderer = glGetString(GL_RENDERER).decode('utf-8')
        version = glGetString(GL_VERSION).decode('utf-8')
        
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS)
        glDepthMask(GL_TRUE)
        glDepthFunc(GL_LESS)
        glDepthRange(0.0, 1.0)
        glEnable(GL_NORMALIZE)

        glClearColor(0.,0.,0.,0.)
    
    def compile_shaders(self):
        
        # book keeping
        textured_shader_data = {}
        vertex_color_shader_data = {}
        mask_shader_data = {}
        coord_shader_data = {}
        background_shader_data = {}
        textured_depthmap_shader_data = {}
        
        # compile the shaders
        textured_shader_data['vertex_shader'] = shaders.compileShader(
                shader_definitions.textured_vertex_shader, GL_VERTEX_SHADER)
        textured_shader_data['fragment_shader'] = shaders.compileShader(
                shader_definitions.textured_fragment_shader, GL_FRAGMENT_SHADER)
        vertex_color_shader_data['vertex_shader'] = shaders.compileShader(
                shader_definitions.vertex_color_vertex_shader, GL_VERTEX_SHADER)
        vertex_color_shader_data['fragment_shader'] = shaders.compileShader(
                shader_definitions.vertex_color_fragment_shader,
                GL_FRAGMENT_SHADER)
        mask_shader_data['vertex_shader'] = shaders.compileShader(
                shader_definitions.mask_vertex_shader, GL_VERTEX_SHADER)
        mask_shader_data['fragment_shader'] = shaders.compileShader(
                shader_definitions.mask_fragment_shader, GL_FRAGMENT_SHADER)
        coord_shader_data['vertex_shader'] = shaders.compileShader(
                shader_definitions.coord_vertex_shader, GL_VERTEX_SHADER)
        coord_shader_data['fragment_shader'] = shaders.compileShader(
                shader_definitions.coord_fragment_shader, GL_FRAGMENT_SHADER)
        background_shader_data['vertex_shader'] = shaders.compileShader(
                shader_definitions.background_vertex_shader, GL_VERTEX_SHADER)
        background_shader_data['fragment_shader'] = shaders.compileShader(
                shader_definitions.background_fragment_shader,
                GL_FRAGMENT_SHADER)
        textured_depthmap_shader_data['vertex_shader'] = shaders.compileShader(
                shader_definitions.textured_depthmap_vertex_shader,
                GL_VERTEX_SHADER)
        textured_depthmap_shader_data['fragment_shader'] = (
                shaders.compileShader(
                    shader_definitions.textured_depthmap_fragment_shader,
                    GL_FRAGMENT_SHADER))
        
        # compile the programs
        textured_program = shaders.compileProgram(
                textured_shader_data['vertex_shader'],
                textured_shader_data['fragment_shader'])
        textured_shader_data['program'] = textured_program
        vertex_color_program = shaders.compileProgram(
                vertex_color_shader_data['vertex_shader'],
                vertex_color_shader_data['fragment_shader'])
        vertex_color_shader_data['program'] = vertex_color_program
        mask_program = shaders.compileProgram(
                mask_shader_data['vertex_shader'],
                mask_shader_data['fragment_shader'])
        mask_shader_data['program'] = mask_program
        coord_program = shaders.compileProgram(
                coord_shader_data['vertex_shader'],
                coord_shader_data['fragment_shader'])
        coord_shader_data['program'] = coord_program
        background_program = shaders.compileProgram(
                background_shader_data['vertex_shader'],
                background_shader_data['fragment_shader'])
        background_shader_data['program'] = background_program
        textured_depthmap_program = shaders.compileProgram(
                textured_depthmap_shader_data['vertex_shader'],
                textured_depthmap_shader_data['fragment_shader'])
        textured_depthmap_shader_data['program'] = textured_depthmap_program
        
        # get attribute locations
        textured_shader_data['locations'] = {}
        vertex_color_shader_data['locations'] = {}
        mask_shader_data['locations'] = {}
        coord_shader_data['locations'] = {}
        background_shader_data['locations'] = {}
        textured_depthmap_shader_data['locations'] = {}
        
        # (position/normal/uv)
        textured_shader_data['locations']['vertex_position'] = (
                glGetAttribLocation(textured_program, 'vertex_position'))
        textured_shader_data['locations']['vertex_normal'] = (
                glGetAttribLocation(textured_program, 'vertex_normal'))
        textured_shader_data['locations']['vertex_uv'] = (
                glGetAttribLocation(textured_program, 'vertex_uv'))
        
        vertex_color_shader_data['locations']['vertex_position'] = (
                glGetAttribLocation(vertex_color_program, 'vertex_position'))
        vertex_color_shader_data['locations']['vertex_normal'] = (
                glGetAttribLocation(vertex_color_program, 'vertex_normal'))
        vertex_color_shader_data['locations']['vertex_color'] = (
                glGetAttribLocation(vertex_color_program, 'vertex_color'))
        
        mask_shader_data['locations']['vertex_position'] = (
                glGetAttribLocation(mask_program, 'vertex_position'))
        
        coord_shader_data['locations']['vertex_position'] = (
                glGetAttribLocation(coord_program, 'vertex_position'))
        
        textured_depthmap_shader_data['locations']['vertex_depth'] = (
                glGetAttribLocation(textured_depthmap_program, 'vertex_depth'))
        
        # (pose and projection matrices)
        for variable in 'camera_pose', 'projection_matrix', 'model_pose':
            textured_shader_data['locations'][variable] = (
                    glGetUniformLocation(textured_program, variable))
            vertex_color_shader_data['locations'][variable] = (
                    glGetUniformLocation(vertex_color_program, variable))
            mask_shader_data['locations'][variable] = (
                    glGetUniformLocation(mask_program, variable))
            coord_shader_data['locations'][variable] = (
                    glGetUniformLocation(coord_program, variable))
            textured_depthmap_shader_data['locations'][variable] = (
                    glGetUniformLocation(textured_depthmap_program, variable))
        
        for variable in (
                'camera_pose', 'projection_matrix', 'offset_matrix', 'blur'):
            background_shader_data['locations'][variable] = (
                    glGetUniformLocation(background_program, variable))
        
        # (material data)
        for variable in (
                'material_properties',
                'image_light_offset_matrix',
                'image_light_diffuse_minmax',
                'image_light_diffuse_rescale',
                'image_light_diffuse_tint_lo',
                'image_light_diffuse_tint_hi',
                'image_light_reflect_tint',
                'image_light_material_properties',
                'diffuse_sampler',
                'reflect_sampler'):
            textured_shader_data['locations'][variable] = (
                    glGetUniformLocation(textured_program, variable))
            vertex_color_shader_data['locations'][variable] = (
                    glGetUniformLocation(vertex_color_program, variable))
        
        # (sampler data)
        textured_shader_data['locations']['texture_sampler'] = (
                glGetUniformLocation(textured_program, 'texture_sampler'))
        background_shader_data['locations']['cubemap_sampler'] = (
                glGetUniformLocation(background_program, 'cubemap_sampler'))
        textured_depthmap_shader_data['locations']['texture_sampler'] = (
                glGetUniformLocation(
                    textured_depthmap_program, 'texture_sampler'))
        
        glUseProgram(textured_program)
        glUniform1i(textured_shader_data['locations']['texture_sampler'], 0)
        glUniform1i(textured_shader_data['locations']['diffuse_sampler'], 2)
        glUniform1i(textured_shader_data['locations']['reflect_sampler'], 3)
        
        glUseProgram(vertex_color_program)
        glUniform1i(vertex_color_shader_data['locations']['diffuse_sampler'], 2)
        glUniform1i(vertex_color_shader_data['locations']['reflect_sampler'], 3)
        
        glUseProgram(background_program)
        glUniform1i(background_shader_data['locations']['cubemap_sampler'], 0)
        
        glUseProgram(textured_depthmap_program)
        glUniform1i(
                textured_depthmap_shader_data['locations']['texture_sampler'],
                0)
        
        # (light data)
        for variable in (
                'ambient_color',
                'num_point_lights',
                'num_direction_lights',
                'point_light_data',
                'direction_light_data'):
            textured_shader_data['locations'][variable] = (
                    glGetUniformLocation(textured_program, variable))
            vertex_color_shader_data['locations'][variable] = (
                    glGetUniformLocation(vertex_color_program, variable))
        
        # (mask data)
        mask_shader_data['locations']['mask_color'] = (
                glGetUniformLocation(mask_program, 'mask_color'))
        
        # (coord_data)
        coord_shader_data['locations']['box_min'] = (
                glGetUniformLocation(coord_program, 'box_min'))
        coord_shader_data['locations']['box_max'] = (
                glGetUniformLocation(coord_program, 'box_max'))
        
        # (textured depthmap data)
        textured_depthmap_shader_data['locations']['focal_length'] = (
                glGetUniformLocation(textured_depthmap_program, 'focal_length'))
        textured_depthmap_shader_data['locations']['width'] = (
                glGetUniformLocation(textured_depthmap_program, 'width'))
        textured_depthmap_shader_data['locations']['height'] = (
                glGetUniformLocation(textured_depthmap_program, 'height'))
        
        self.gl_data['textured_shader'] = textured_shader_data
        self.gl_data['vertex_color_shader'] = vertex_color_shader_data
        self.gl_data['mask_shader'] = mask_shader_data
        self.gl_data['coord_shader'] = coord_shader_data
        self.gl_data['background_shader'] = background_shader_data
        self.gl_data['textured_depthmap_shader'] = textured_depthmap_shader_data
    
    #===========================================================================
    # scene methods
    #===========================================================================
    def load_scene(self, scene, clear_scene=False, reload_assets=False):
        if clear_scene:
            self.clear_scene()
        
        if isinstance(scene, str):
            scene = self.asset_library['scenes'][scene]
            scene = json.load(open(scene))
        
        # meshes, depthmaps, materials, image_lights
        for singular, plural in self.asset_types:
            if plural in scene:
                for asset_name, asset_args in scene[plural].items():
                    exists_fn = getattr(self, singular + '_exists')
                    if reload_assets or not exists_fn(asset_name):
                        load_fn = getattr(self, 'load_' + singular)
                        load_fn(asset_name, **asset_args)
        
        '''
        if 'meshes' in scene:
            for mesh_name, mesh_args in scene['meshes'].items():
                if reload_assets or not self.mesh_exists(mesh_name):
                    self.load_mesh(mesh_name, **mesh_args)
        
        if 'depthmaps' in scene:
            for depthmap_name, depthmap_args in scene['depthmaps'].items():
                if reload_assets or not self.depthmap_exists(depthmap_name):
                    self.load_depthmap(depthmap_name, **depthmap_args)
        
        if 'materials' in scene:
            for material in scene['materials']:
                if reload_assets or not self.material_exists(material):
                    self.load_material(material, **scene['materials'][material])
        
        if 'image_lights' in scene:
            for image_light in scene['image_lights']:
                if reload_assets or not self.image_light_exists(image_light):
                    image_light_arguments = scene['image_lights'][image_light]
                    image_light_arguments.setdefault('set_active', False)
                    self.load_image_light(image_light, **image_light_arguments)
        '''
        
        # instances, depthmap_instances, point_lights, direction_lights
        for singular, plural in self.instance_types:
            if plural in scene:
                for instance_name, instance_args in scene[plural].items():
                    add_fn = getattr(self, 'add_' + singular)
                    add_fn(instance_name, **instance_args)
        
        '''
        if 'instances' in scene:
            for instance in scene['instances']:
                self.add_instance(instance, **scene['instances'][instance])
        
        if 'depthmap_instances' in scene:
            for depthmap_instance in scene['depthmap_instances']:
                self.add_depthmap_instance(
                        depthmap_instance,
                        **scene['depthmap_instances'][depthmap_instance])
        
        if 'point_lights' in scene:
            for point_light in scene['point_lights']:
                self.add_point_light(
                        point_light,
                        **scene['point_lights'][point_light])
        
        if 'direction_lights' in scene:
            for direction_light in scene['direction_lights']:
                self.add_direction_light(
                        direction_light,
                        **scene['direction_lights'][direction_light])
        '''
        
        for global_parameter in self.global_parameters:
            if global_parameter in scene:
                set_fn = getattr(self, 'set_' + global_parameter)
                set_fn(scene[global_parameter])
        
        '''
        if 'active_image_light' in scene:
            self.set_active_image_light(scene['active_image_light'])
        
        if 'ambient_color' in scene:
            self.set_ambient_color(scene['ambient_color'])
        
        if 'background_color' in scene:
            self.set_background_color(scene['background_color'])
        '''
        
        if 'camera' in scene:
            if 'pose' in scene['camera']:
                self.set_camera_pose(scene['camera']['pose'])
            if 'projection' in scene['camera']:
                self.set_projection(scene['camera']['projection'])
    
    def clear_scene(self):
        for singular, plural in self.asset_types:
            getattr(self, 'clear_' + plural)()
        #self.clear_meshes()
        #self.clear_depthmaps()
        #self.clear_materials()
        #self.clear_image_lights()
        for singular, plural in self.instance_types:
            getattr(self, 'clear_' + plural)()
        #self.clear_instances()
        #self.clear_depthmap_instances()
        #self.clear_point_lights()
        #self.clear_direction_lights()
        self.set_ambient_color([0,0,0])
        self.set_background_color([0,0,0,0])
        self.scene_description['active_image_light'] = None
        self.reset_camera()
    
    #===========================================================================
    # global settings
    #===========================================================================
    def set_ambient_color(self, color):
        self.scene_description['ambient_color'] = numpy.array(color)
    
    def set_background_color(self, background_color):
        if len(background_color) == 3:
            background_color = tuple(background_color) + (1,)
        self.scene_description['background_color'] = background_color
    
    def set_active_image_light(self, image_light):
        self.scene_description['active_image_light'] = image_light
    
    #===========================================================================
    # camera methods
    #===========================================================================
    def reset_camera(self):
        self.set_camera_pose(self.default_camera_pose)
        self.set_projection(self.default_camera_projection)
    
    def set_projection(self, projection_matrix):
        self.scene_description['camera']['projection'] = numpy.array(
                projection_matrix)
    
    def get_projection(self):
        return self.scene_description['camera']['projection']
    
    def set_camera_pose(self, camera_pose):
        camera_pose = camera.camera_pose_to_matrix(camera_pose)
        self.scene_description['camera']['pose'] = numpy.array(
                camera_pose)
    
    def get_camera_pose(self):
        return self.scene_description['camera']['pose']
    
    def camera_frame_scene(self, *args, **kwargs):
        bbox = self.get_instance_center_bbox()
        camera_matrix = camera.frame_bbox(
                bbox, self.get_projection(), 3.0,
                *args, **kwargs)
        self.set_camera_pose(camera_matrix)
    
    #===========================================================================
    # mesh methods
    #===========================================================================
    def load_mesh(self,
            name,
            mesh_asset = None,
            mesh_path = None,
            mesh_data = None,
            scale = 1.0,
            create_uvs = False):
        
        print('importing mesh: %s (%s)'%(name, mesh_asset))
        
        # if a mesh asset name was provided, load that
        if mesh_asset is not None:
            mesh_path = self.asset_library['meshes'][mesh_asset]
            #mesh = obj_mesh.load_mesh(mesh_path, scale=scale)
            mesh = trimesh.load(
                    mesh_path, process=False, skip_materials=True, force='mesh')
            self.scene_description['meshes'][name] = {'mesh_asset':mesh_asset}
        
        # otherwise, load name as an asset path
        elif mesh_path is not None:
            #mesh = obj_mesh.load_mesh(mesh_path, scale=scale)
            mesh = trimesh.load(
                    mesh_path, process=False, skip_materials=True, force='mesh')
            self.scene_description['meshes'][name] = {'mesh_path':mesh_path}
        
        # otherwise if mesh data was provided, load that
        elif mesh_data is not None:
            # need to figure out what this looks like with trimesh
            raise NotImplementedError
            mesh = mesh_data
            self.scene_description['meshes'][name] = {'mesh_data':mesh_data}
        
        else:
            raise RenderpyException(
                    'Must supply a "mesh_asset", "mesh_path" or "mesh_data" '
                    'argument when loading a mesh')
        
        # create mesh buffers and load the mesh data
        mesh_buffers = {}
        
        vertex_floats = numpy.array(mesh.vertices, dtype=numpy.float32)
        normal_floats = numpy.array(mesh.vertex_normals, dtype=numpy.float32)
        if mesh.visual.uv is None and create_uvs:
            visual = trimesh.visual.texture.TextureVisuals(
                    uv=numpy.array([[0,0] for _ in mesh.vertices]))
            mesh.visual = visual
            #mesh['uvs'] = [[0,0] for _ in mesh['vertices']]
        
        #if len(mesh['uvs']):
        if mesh.visual.uv is not None:
            uv_floats = numpy.array(mesh.visual.uv, dtype=numpy.float32)
            combined_floats = numpy.concatenate(
                    (vertex_floats, normal_floats, uv_floats), axis=1)
            mesh_buffers['vertex_buffer'] = vbo.VBO(combined_floats)
        
        else:
            # update this to the age of trimesh
            raise NotImplementedError
            vertex_color_floats = numpy.array(
                    mesh['vertex_colors'], dtype=numpy.float32)
            combined_floats = numpy.concatenate(
                    (vertex_floats, normal_floats, vertex_color_floats), axis=1)
            mesh_buffers['vertex_buffer'] = vbo.VBO(combined_floats)
            
        face_ints = numpy.array(mesh.faces, dtype=numpy.int32)
        mesh_buffers['face_buffer'] = vbo.VBO(
                face_ints,
                target = GL_ELEMENT_ARRAY_BUFFER)
        
        # store the loaded and gl data
        self.loaded_data['meshes'][name] = mesh
        self.gl_data['mesh_buffers'][name] = mesh_buffers
    
    def load_background_mesh(self):
        if 'BACKGROUND' not in self.gl_data['mesh_buffers']:
            mesh_buffers = {}
            vertex_floats = numpy.array([
                    [-1,-1,0],
                    [-1, 1,0],
                    [ 1, 1,0],
                    [ 1,-1,0]])
            mesh_buffers['vertex_buffer'] = vbo.VBO(vertex_floats)
            
            face_ints = numpy.array([
                    [0,1,2],
                    [2,3,0]], dtype=numpy.int32)
            mesh_buffers['face_buffer'] = vbo.VBO(
                    face_ints,
                    target = GL_ELEMENT_ARRAY_BUFFER)
            self.gl_data['mesh_buffers']['BACKGROUND'] = mesh_buffers
    
    def remove_mesh(self, name):
        del(self.scene_description['meshes'][name])
        self.gl_data['mesh_buffers'][name]['vertex_buffer'].delete()
        self.gl_data['mesh_buffers'][name]['face_buffer'].delete()
        del(self.gl_data['mesh_buffers'][name])
        del(self.loaded_data['meshes'][name])
    
    def clear_meshes(self):
        for name in list(self.scene_description['meshes'].keys()):
            self.remove_mesh(name)
    
    def list_meshes(self):
        return list(self.scene_description['meshes'].keys())
    
    def mesh_exists(self, mesh):
        return mesh in self.scene_description['meshes']
    
    def get_mesh(self, mesh):
        return self.scene_description['meshes'][mesh]
    
    #===========================================================================
    # depthmap methods
    #===========================================================================
    def load_depthmap(self,
            name,
            depthmap_asset = None,
            depthmap_path = None,
            depthmap_data = None,
            indices = None,
            focal_length = (1,1)):
        
        if name in self.scene_description['depthmaps']:
            self.remove_depthmap(name)
        
        # if an asset was provided, load that
        if depthmap_asset is not None:
            depthmap_path = self.asset_library['depthmaps'][depthmap_path]
            depthmap = load_depth(depthmap_path)
            self.scene_description['depthmaps'][name] = {
                    'depthmap_asset':depthmap_asset}
        
        # if a path was provided, load that
        elif depthmap_path is not None:
            depthmap = load_depth(depthmap_path)
            self.scene_description['depthmaps'][name] = {
                    'depthmap_path':depthmap_path}
        
        # if depthmap data was provided, load that
        elif depthmap_data is not None:
            self.scene_description['depthmaps'][name] = {
                    'depthmap_data':depthmap_data}
            depthmap = depthmap_data
        
        else:
            raise RenderpyException(
                    'Must supply a "depthmap_asset", "depthmap_path" or a '
                    '"depthmap_data" argument when loading a depthmap')
        
        depthmap = numpy.array(depthmap, dtype=numpy.float32)
        
        self.scene_description['depthmaps'][name]['height'] = depthmap.shape[0]
        self.scene_description['depthmaps'][name]['width'] = depthmap.shape[1]
        self.scene_description['depthmaps'][name]['focal_length'] = focal_length
        
        # create depth VBO
        depthmap_buffers = {}
        depthmap_buffers['depth_buffer'] = vbo.VBO(depthmap)
        
        # create index VBO
        if indices is None:
            indices = numpy.arange(
                    depthmap.shape[0] * depthmap.shape[1],
                    dtype = numpy.int32)
        depthmap_buffers['index_buffer'] = vbo.VBO(
                indices,
                target = GL_ELEMENT_ARRAY_BUFFER)
        
        # store the loaded and gl data
        self.loaded_data['depthmaps'][name] = depthmap
        self.gl_data['depthmap_buffers'][name] = depthmap_buffers
    
    def remove_depthmap(self, name):
        del(self.scene_description['depthmaps'][name])
        self.gl_data['depthmap_buffers'][name]['depth_buffer'].delete()
        del(self.gl_data['depthmap_buffers'][name])
        del(self.loaded_data['depthmaps'][name])
    
    def clear_depthmaps(self):
        for name in list(self.scene_description['depthmaps'].keys()):
            self.remove_depthmap(name)
    
    def list_depthmaps(self):
        return list(self.scene_description['depthmaps'].keys())
    
    def depthmap_exists(self, depthmap):
        return depthmap in self.scene_description['depthmaps']
    
    def get_depthmap(self, depthmap):
        return self.scene_description['depthmaps'][depthmap]
    
    #===========================================================================
    # image_light methods
    #===========================================================================
    def load_image_light(self,
            name,
            diffuse_textures = None,
            example_diffuse_textures = None,
            reflection_textures = None,
            texture_directory = None,
            reflection_mipmaps = None,
            offset_matrix = numpy.eye(4),
            blur = 0.0,
            diffuse_contrast = 1.,
            rescale_diffuse_intensity = False,
            diffuse_intensity_target_lo = 0.,
            diffuse_intensity_target_hi = 1.,
            diffuse_tint_lo = (0,0,0),
            diffuse_tint_hi = (0,0,0),
            reflect_tint = (0,0,0),
            render_background = True,
            crop = None,
            set_active = False):
        
        if texture_directory is None:
            if diffuse_textures is not None:
                pass
            
            else:
                raise RenderpyException('Must specify a '
                        'diffuse_texture or a texture_directory'
                        'when loading an image_light')
            
            if reflection_textures is not None:
                pass
            
            else:
                raise RenderpyException('Must specify a '
                        'reflection_texture or a texture directory'
                        'when loading an image_light')
        
        if name in self.gl_data['light_buffers']:
            glDeleteTextures([
                    self.gl_data['light_buffers'][name]['diffuse_texture'],
                    self.gl_data['light_buffers'][name]['reflection_texture']])
        
        light_buffers = {}
        light_buffers['diffuse_texture'] = glGenTextures(1)
        light_buffers['reflection_texture'] = glGenTextures(1)
        self.gl_data['light_buffers'][name] = light_buffers
        
        image_light_data = {}
        image_light_data['offset_matrix'] = numpy.array(offset_matrix)
        image_light_data['blur'] = blur
        image_light_data['render_background'] = render_background
        image_light_data['diffuse_contrast'] = diffuse_contrast
        image_light_data['rescale_diffuse_intensity'] = (
                rescale_diffuse_intensity)
        image_light_data['diffuse_intensity_target_lo'] = (
                diffuse_intensity_target_lo)
        image_light_data['diffuse_intensity_target_hi'] = (
                diffuse_intensity_target_hi)
        image_light_data['diffuse_tint_lo'] = diffuse_tint_lo
        image_light_data['diffuse_tint_hi'] = diffuse_tint_hi
        image_light_data['reflect_tint'] = reflect_tint
        self.scene_description['image_lights'][name] = image_light_data
        self.replace_image_light_textures(
                name,
                diffuse_textures,
                reflection_textures,
                texture_directory,
                reflection_mipmaps)
        
        self.load_background_mesh()
        
        if set_active:
            self.set_active_image_light(name)
    
    def remove_image_light(self, name):
        glDeleteTextures(
                self.gl_data['light_buffers'][name]['diffuse_texture'])
        glDeleteTextures(
                self.gl_data['light_buffers'][name]['reflection_texture'])
        del(self.gl_data['light_buffers'][name])
        del(self.loaded_data['textures'][name + '_diffuse'])
        del(self.loaded_data['textures'][name + '_reflect'])
        del(self.scene_description['image_lights'][name])
        
        # delete the background mesh if there are no image lights left
        if len(self.scene_description['image_lights']) == 0:
            self.gl_data['mesh_buffers']['BACKGROUND']['vertex_buffer'].delete()
            self.gl_data['mesh_buffers']['BACKGROUND']['face_buffer'].delete()
            del(self.gl_data['mesh_buffers']['BACKGROUND'])
    
    def clear_image_lights(self):
        for image_light in list(self.scene_description['image_lights'].keys()):
            self.remove_image_light(image_light)
        self.set_active_image_light(None)
    
    def list_image_lights(self):
        return list(self.scene_description['image_lights'].keys())
    
    def image_light_exists(self, image_light):
        return image_light in self.scene_description['image_lights']
    
    def get_image_light(self, image_light):
        return self.scene_description['image_lights'][image_light]
    
    #===========================================================================
    # texture methods
    #===========================================================================
    @staticmethod
    def validate_texture(image):
        if image.shape[0] not in [1,2,4,8,16,32,64,128,256,512,1024,2048,4096]:
            raise Exception('Image height must be a power of 2 '
                    'less than or equal to 4096 (Got %i)'%(image.shape[0]))
        if image.shape[1] not in [1,2,4,8,16,32,64,128,256,512,1024,2048,4096]:
            raise Exception('Image width must be a power of 2 '
                    'less than or equal to 4096 (Got %i)'%(image.shape[1]))
    
    def replace_texture(self,
            name,
            texture,
            crop = None):
        
        if isinstance(texture, str):
            self.scene_description['materials'][name]['texture'] = texture
            texture = self.asset_library['textures'][texture]
            image = load_image(texture)
        else:
            self.scene_description['materials'][name]['texture'] = -1
            image = numpy.array(texture)
        
        if crop is not None:
            image = image[crop[0]:crop[2], crop[1]:crop[3]]
        
        image = numpy.array(image)
        self.validate_texture(image)
        self.loaded_data['textures'][name] = image
        
        material_buffers = self.gl_data['material_buffers'][name]
        glBindTexture(GL_TEXTURE_2D, material_buffers['texture'])
        try:
            glTexImage2D(
                    GL_TEXTURE_2D, 0, GL_RGB,
                    image.shape[1], image.shape[0], 0,
                    GL_RGB, GL_UNSIGNED_BYTE, image)
            
            # GL_NEAREST?
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                    GL_LINEAR_MIPMAP_LINEAR)
            glGenerateMipmap(GL_TEXTURE_2D)
            
        finally:
            glBindTexture(GL_TEXTURE_2D, 0)
    
    def replace_image_light_textures(self,
            name,
            diffuse_textures = None,
            reflection_textures = None,
            texture_directory = None,
            reflection_mipmaps = None):
        
        # make sure that either diffuse and reflection textures were provided
        # or an image directory was provided
        if texture_directory is not None:
            texture_directory = (
                    self.asset_library['image_lights'][texture_directory])
            cube_order = {'px':0, 'nx':1, 'py':2, 'ny':3, 'pz':4, 'nz':5}
            texture_directory = os.path.expanduser(texture_directory)
            all_images = os.listdir(texture_directory)
            diffuse_files = sorted(
                    [image for image in all_images if '_dif.' in image],
                    key = lambda x : cube_order[x[:2]])
            diffuse_textures = [
                    os.path.join(texture_directory, diffuse_file)
                    for diffuse_file in diffuse_files]
            reflection_files = sorted(
                    [image for image in all_images if '_ref.' in image],
                    key = lambda x : cube_order[x[:2]])
            reflection_textures = [
                    os.path.join(texture_directory, reflection_file)
                    for reflection_file in reflection_files]
        
        elif diffuse_textures is None or reflection_textures is None:
            raise Exception('Must provide either diffuse and reflection'
                    'textures, or an image directory')
        
        light_description = self.scene_description['image_lights'][name]
        
        try:
            diffuse_textures = [
                    diffuse_textures['px'],
                    diffuse_textures['nx'],
                    diffuse_textures['py'],
                    diffuse_textures['ny'],
                    diffuse_textures['pz'],
                    diffuse_textures['nz']]
        except TypeError:
            pass
        
        try:
            reflection_textures = [
                    reflection_textures['px'],
                    reflection_textures['nx'],
                    reflection_textures['py'],
                    reflection_textures['ny'],
                    reflection_textures['pz'],
                    reflection_textures['nz']]
        except TypeError:
            pass
        
        if isinstance(diffuse_textures[0], str):
            light_description['diffuse_textures'] = diffuse_textures
            diffuse_images = [load_image(diffuse_texture)
                    for diffuse_texture in diffuse_textures]
        else:
            light_description['diffuse_textures'] = -1
            diffuse_images = diffuse_textures
        
        if isinstance(reflection_textures[0], str):
            light_description['reflection_textures'] = reflection_textures
            reflection_images = [load_image(reflection_texture)
                    for reflection_texture in reflection_textures]
        else:
            light_description['reflection_textures'] = -1
            reflection_images = reflection_textures
            
        if reflection_mipmaps:
            if isinstance(reflection_mipmaps[0][0], str):
                light_description['reflection_mipmaps'] = reflection_mipmaps
                reflection_mipmaps = [
                        [load_image(mipmap) for mipmap in mipmaps]
                        for mipmaps in reflection_mipmaps]
            else:
                light_description['reflection_mipmaps'] = -1
        else:
            light_description['reflection_mipmaps'] = -1
        
        light_buffers = self.gl_data['light_buffers'][name]
        glBindTexture(GL_TEXTURE_CUBE_MAP, light_buffers['diffuse_texture'])
        try:
            diffuse_min = float('inf')
            diffuse_max = -float('inf')
            for i, diffuse_image in enumerate(diffuse_images):
                diffuse_image = numpy.array(diffuse_image)
                self.validate_texture(diffuse_image)
                glTexImage2D(
                        GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                        0, GL_RGB,
                        diffuse_image.shape[1], diffuse_image.shape[0],
                        0, GL_RGB, GL_UNSIGNED_BYTE, diffuse_image)
                
                diffuse_intensity = (
                        diffuse_image[:,:,0] * 0.2989 +
                        diffuse_image[:,:,1] * 0.5870 +
                        diffuse_image[:,:,2] * 0.1140)
                
                diffuse_min = min(diffuse_min, numpy.min(diffuse_intensity))
                diffuse_max = max(diffuse_max, numpy.max(diffuse_intensity))
            
            light_description['diffuse_min'] = diffuse_min / 255.
            light_description['diffuse_max'] = diffuse_max / 255.
            
            glTexParameteri(
                    GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(
                    GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER,
                    GL_LINEAR_MIPMAP_LINEAR)
            glGenerateMipmap(GL_TEXTURE_CUBE_MAP)
            glTexParameteri(
                    GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(
                    GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexParameteri(
                    GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
        finally:
            glBindTexture(GL_TEXTURE_CUBE_MAP, 0)
        
        glBindTexture(
                GL_TEXTURE_CUBE_MAP, light_buffers['reflection_texture'])
        try:
            for i, reflection_image in enumerate(reflection_images):
                reflection_image = numpy.array(reflection_image)
                self.validate_texture(reflection_image)
                glTexImage2D(
                        GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                        0, GL_RGB,
                        reflection_image.shape[1], reflection_image.shape[0],
                        0, GL_RGB, GL_UNSIGNED_BYTE, reflection_image)
                if reflection_mipmaps is not None:
                    for j, mipmap in enumerate(reflection_mipmaps[i]):
                        mipmap = numpy.array(mipmap)
                        self.validate_texture(mipmap)
                        glTexImage2D(
                                GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                                j+1, GL_RGB,
                                mipmap.shape[1], mipmap.shape[0],
                                0, GL_RGB, GL_UNSIGNED_BYTE, mipmap)
            
            glTexParameteri(
                    GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(
                    GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER,
                    GL_LINEAR_MIPMAP_LINEAR)
            if reflection_mipmaps is None:
                glGenerateMipmap(GL_TEXTURE_CUBE_MAP)
            else:
                glTexParameteri(
                        GL_TEXTURE_CUBE_MAP,
                        GL_TEXTURE_MAX_LEVEL, 
                        len(reflection_mipmaps[0]))
            glTexParameteri(
                    GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(
                    GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexParameteri(
                    GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
        finally:
            glBindTexture(GL_TEXTURE_CUBE_MAP, 0)
        
        self.loaded_data['textures'][name + '_diffuse'] = diffuse_images
        self.loaded_data['textures'][name + '_reflect'] = reflection_images
    
    def get_texture(self, texture_name):
        return self.loaded_data['textures'][texture_name]
    
    #===========================================================================
    # material methods
    #===========================================================================
    def load_material(self,
            name,
            texture = None,
            color = None,
            ka = 1.0,
            kd = 1.0,
            ks = 0.5,
            shine = 1.0,
            image_light_kd = 0.85,
            image_light_ks = 0.15,
            image_light_blur_reflection = 2.0,
            crop = None):
        
        self.scene_description['materials'][name] = {
                'ka' : ka,
                'kd' : kd,
                'ks' : ks,
                'shine' : shine,
                'image_light_kd' : image_light_kd,
                'image_light_ks' : image_light_ks,
                'image_light_blur_reflection' : image_light_blur_reflection}
        
        if name in self.gl_data['material_buffers']:
            glBindTexture(GL_TEXTURE_2D,0)
            glDeleteTextures(
                    [self.gl_data['material_buffers'][name]['texture']])
        
        material_buffers = {}
        material_buffers['texture'] = glGenTextures(1)
        self.gl_data['material_buffers'][name] = material_buffers
        
        if color is not None:
            texture = numpy.zeros((16,16,3), dtype=numpy.uint8)
            texture[:] = color
        
        if texture is not None:
            self.replace_texture(name, texture, crop)
    
    def remove_material(self, name):
        glDeleteTextures(self.gl_data['material_buffers'][name]['texture'])
        if name in self.loaded_data['textures']:
            del(self.loaded_data['textures'][name])
        del(self.gl_data['material_buffers'][name])
        del(self.scene_description['materials'][name])
    
    def clear_materials(self):
        for name in list(self.scene_description['materials'].keys()):
            self.remove_material(name)
    
    def list_materials(self):
        return list(self.scene_description['materials'].keys())
    
    def material_exists(self, material):
        return material in self.scene_description['materials']
    
    def get_material(self, material_name):
        return self.scene_description['materials'][material_name]
    
    #===========================================================================
    # instance methods
    #===========================================================================
    def add_instance(self,
            instance_name,
            mesh_name,
            material_name,
            transform = numpy.eye(4),
            mask_color = numpy.array([0,0,0]),
            coord_box = ((0,0,0),(0,0,0)),
            hidden = False):
        
        instance_data = {}
        instance_data['mesh_name'] = mesh_name
        instance_data['material_name'] = material_name
        instance_data['transform'] = numpy.array(transform)
        instance_data['mask_color'] = numpy.array(mask_color)
        instance_data['coord_box'] = numpy.array(coord_box)
        instance_data['hidden'] = hidden
        self.scene_description['instances'][instance_name] = instance_data
    
    def remove_instance(self, instance_name):
        del(self.scene_description['instances'][instance_name])
    
    def clear_instances(self):
        self.scene_description['instances'] = {}
    
    def get_instance_transform(self, instance_name):
        return self.scene_description['instances'][instance_name]['transform']
    
    def set_instance_transform(self, instance_name, transform):
        self.scene_description['instances'][instance_name]['transform'] = (
                numpy.array(transform))
    
    def set_instance_material(self, instance_name, material_name):
        self.scene_description['instances'][instance_name]['material_name'] = (
                material_name)
    
    def hide_instance(self, instance_name):
        self.scene_description['instances'][instance_name]['hidden'] = True
    
    def show_instance(self, instance_name):
        self.scene_description['instances'][instance_name]['hidden'] = False
    
    def get_instance_mesh_name(self, instance_name):
        return self.scene_description['instances'][instance_name]['mesh_name']
    
    def get_instance_center_bbox(self, instances=None):
        if instances is None:
            instances = self.scene_description['instances'].keys()
        centers = numpy.stack([
                self.scene_description['instances'][instance]['transform'][:3,3]
                for instance in instances])
        bbox_min = numpy.min(centers, axis=0)
        bbox_max = numpy.max(centers, axis=0)
        return bbox_min, bbox_max
    
    def set_instance_masks_to_instance_indices(self, instance_indices):
        for instance_name, index in instance_indices.items():
            instance_data = self.scene_description['instances'][instance_name]
            instance_data['mask_color'] = masks.color_index_to_float(index)
        
    def set_instance_masks_to_mesh_indices(self, mesh_indices, instances=None):
        if instances is None:
            instances = self.scene_description['instances'].keys()
        for instance in instances:
            instance_data = self.scene_description['instances'][instance]
            mesh_name = instance_data['mesh_name']
            try:
                mesh_index = mesh_indices[mesh_name]
            except KeyError:
                continue
            instance_data['mask_color'] = (
                    masks.color_index_to_float(mesh_index))
    
    def list_instances(self):
        return list(self.scene_description['instances'].keys())
    
    def instance_exists(self, instance):
        return instance in self.scene_description['instances']
    
    def instance_hidden(self, instance):
        return self.scene_description['instances'][instance]['hidden']
    
    #===========================================================================
    # depthmap_instance methods
    #===========================================================================
    def add_depthmap_instance(self,
            depthmap_instance_name,
            depthmap_name,
            material_name,
            transform = numpy.eye(4),
            point_size = 1):
        
        depthmap_instance_data = {}
        depthmap_instance_data['depthmap_name'] = depthmap_name
        depthmap_instance_data['material_name'] = material_name
        depthmap_instance_data['transform'] = numpy.array(transform)
        depthmap_instance_data['point_size'] = point_size
        self.scene_description['depthmap_instances'][depthmap_instance_name] = (
                depthmap_instance_data)
    
    def remove_depthmap_instance(self, depthmap_instance_name):
        del(self.scene_description['depthmap_instances'][
                depthmap_instance_name])
    
    def clear_depthmap_instances(self):
        self.scene_description['depthmap_instances'] = {}
    
    def set_depthmap_instance_material(self,
            depthmap_instance_name, material_name):
        self.scene_description['depthmap_instances'][depthmap_instance_name][
                'material_name'] = material_name
    
    def set_depthmap_instance_transform(self,
            depthmap_instance_name, transform):
        self.scene_description['depthmap_instances'][depthmap_instance_name][
                'transform'] = numpy.array(transform)
    
    def get_depthmap_instance_transform(self, depthmap_instance_name):
        return self.scene_description['depthmap_instances'][
                depthmap_instance_name]['transform']
    
    def depthmap_instance_exists(self, depthmap_instance):
        return depthmap_instance in self.scene_description['depthmap_instances']
    
    #===========================================================================
    # point_light methods
    #===========================================================================
    def add_point_light(self, name, position, color):
        self.scene_description['point_lights'][name] = {
                'position' : numpy.array(position),
                'color' : numpy.array(color)}
    
    def remove_point_light(self, name):
        del(self.scene_description['point_lights'][name])
    
    def clear_point_lights(self):
        self.scene_description['point_lights'] = {}
    
    #===========================================================================
    # direction_light methods
    #===========================================================================
    def add_direction_light(self,
            name,
            direction,
            color):
            #use_shadows = False,
            #shadow_matrix = None,
            #shadow_resolution = None):
        
        self.scene_description['direction_lights'][name] = {
                'direction' : numpy.array(direction),
                'color' : numpy.array(color)}
    
    def remove_direction_light(self, name):
        del(self.scene_description['direction_lights'][name])
    
    def clear_direction_lights(self):
        self.scene_description['direction_lights'] = {}
    
    #===========================================================================
    # render methods
    #===========================================================================
    def clear_frame(self):
        glClearColor(*self.scene_description['background_color'])
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    def finish_frame(self):
        glFinish()
    
    #---------------------------------------------------------------------------
    # color_render methods
    #---------------------------------------------------------------------------
    def color_render(self,
            instances = None,
            depthmap_instances = None,
            flip_y = True,
            clear = True,
            finish = True):
        
        # clear
        if clear:
            self.clear_frame()
        
        # render the background
        image_light_name = self.scene_description['active_image_light']
        if image_light_name is not None:
            image_light_description = (
                    self.scene_description['image_lights'][image_light_name])
            if image_light_description['render_background']:
                self.render_background(image_light_name, flip_y = flip_y)
        
        # set image light maps
        if image_light_name is not None:
            glActiveTexture(GL_TEXTURE2)
            glBindTexture(GL_TEXTURE_CUBE_MAP, self.gl_data[
                    'light_buffers'][image_light_name]['diffuse_texture'])
            glActiveTexture(GL_TEXTURE3)
            glBindTexture(GL_TEXTURE_CUBE_MAP, self.gl_data[
                    'light_buffers'][image_light_name]['reflection_texture'])
        
        # depthmap_instances
        if depthmap_instances is None:
            depthmap_instances = self.scene_description['depthmap_instances']
        
        glUseProgram(self.gl_data['textured_depthmap_shader']['program'])
        try:
            location_data = (
                    self.gl_data['textured_depthmap_shader']['locations'])
            
            # set the camera's pose
            camera_pose = self.scene_description['camera']['pose']
            glUniformMatrix4fv(
                    location_data['camera_pose'],
                    1, GL_TRUE,
                    camera_pose.astype(numpy.float32))
            
            # set the camera's projection matrix
            projection_matrix = (
                    self.scene_description['camera']['projection'])
            if flip_y:
                projection_matrix = numpy.dot(
                        projection_matrix,
                        numpy.array([
                            [ 1, 0, 0, 0],
                            [ 0,-1, 0, 0],
                            [ 0, 0, 1, 0],
                            [ 0, 0, 0, 1]]))
            glUniformMatrix4fv(
                    location_data['projection_matrix'],
                    1, GL_TRUE,
                    projection_matrix.astype(numpy.float32))
            
            # render the depthmap instances
            for depthmap_instance_name in depthmap_instances:
                self.color_render_depthmap_instance(depthmap_instance_name)
        
        finally:
            glUseProgram(0)
        
        # figure out which programs we need (color/vertex_color)
        if instances is None:
            instances = self.scene_description['instances']
        
        textured_shader_instances = []
        vertex_color_shader_instances = []
        for instance in instances:
            instance_mesh = (
                    self.scene_description['instances'][instance]['mesh_name'])
            mesh_data = self.loaded_data['meshes'][instance_mesh]
            if len(mesh_data['uvs']):
                textured_shader_instances.append(instance)
            else:
                vertex_color_shader_instances.append(instance)
        
        for shader_name, shader_instances in (
                ('textured_shader', textured_shader_instances),
                ('vertex_color_shader', vertex_color_shader_instances)):
            
            if len(shader_instances) == 0:
                continue
            
            # turn on the shader
            glUseProgram(self.gl_data[shader_name]['program'])
            
            try:
                
                location_data = self.gl_data[shader_name]['locations']
                
                # set the camera's pose
                camera_pose = self.scene_description['camera']['pose']
                glUniformMatrix4fv(
                        location_data['camera_pose'],
                        1, GL_TRUE,
                        camera_pose.astype(numpy.float32))
                
                # set the camera's projection matrix
                projection_matrix = (
                        self.scene_description['camera']['projection'])
                if flip_y:
                    projection_matrix = numpy.dot(
                            projection_matrix,
                            numpy.array([
                                [ 1, 0, 0, 0],
                                [ 0,-1, 0, 0],
                                [ 0, 0, 1, 0],
                                [ 0, 0, 0, 1]]))
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
                
                # set the image light parameters
                if image_light_name is not None:
                    image_light_data = (
                            self.scene_description['image_lights'][
                                image_light_name])
                    offset_matrix = image_light_data['offset_matrix']
                    glUniformMatrix4fv(
                            location_data['image_light_offset_matrix'],
                            1, GL_TRUE,
                            offset_matrix.astype(numpy.float32))
                    diffuse_minmax = numpy.array(
                            [image_light_data['diffuse_min'],
                             image_light_data['diffuse_max']],
                            dtype=numpy.float32)
                    if image_light_data['rescale_diffuse_intensity']:
                        diffuse_intensity_target_lo = (
                                image_light_data['diffuse_intensity_target_lo'])
                        diffuse_intensity_target_hi = (
                                image_light_data['diffuse_intensity_target_hi'])
                    else:
                        diffuse_intensity_target_lo = (
                                image_light_data['diffuse_min'])
                        diffuse_intensity_target_hi = (
                                image_light_data['diffuse_max'])
                    diffuse_rescale = numpy.array(
                            [image_light_data['diffuse_contrast'],
                             diffuse_intensity_target_lo,
                             diffuse_intensity_target_hi],
                             #image_light_data['diffuse_lo_rescale'],
                             #image_light_data['diffuse_hi_rescale']],
                            dtype=numpy.float32)
                    diffuse_tint_lo = numpy.array(
                            image_light_data['diffuse_tint_lo'],
                            dtype=numpy.float32)
                    diffuse_tint_hi = numpy.array(
                            image_light_data['diffuse_tint_hi'],
                            dtype=numpy.float32)
                    reflect_tint = numpy.array(
                            image_light_data['reflect_tint'],
                            dtype=numpy.float32)
                    glUniform2fv(
                            location_data['image_light_diffuse_minmax'],
                            1, diffuse_minmax)
                    glUniform3fv(
                            location_data['image_light_diffuse_rescale'],
                            1, diffuse_rescale)
                    glUniform3fv(
                            location_data['image_light_diffuse_tint_lo'],
                            1, diffuse_tint_lo)
                    glUniform3fv(
                            location_data['image_light_diffuse_tint_hi'],
                            1, diffuse_tint_hi)
                    glUniform3fv(
                            location_data['image_light_reflect_tint'],
                            1, reflect_tint)
                
                mesh_instances = {}
                for instance_name in shader_instances:
                    instance_data = (
                            self.scene_description['instances'][instance_name])
                    if instance_data['hidden']:
                        continue
                    mesh_name = instance_data['mesh_name']
                    try:
                        mesh_instances[mesh_name].append(instance_name)
                    except KeyError:
                        mesh_instances[mesh_name] = [instance_name]
                
                for mesh_name, instance_names in mesh_instances.items():
                    self.color_render_instance(instance_names[0])
                    for instance_name in instance_names[1:]:
                        self.color_render_instance(
                                instance_name, set_mesh_attrib_pointers=False)
                
            finally:
                glUseProgram(0)
        
        if finish:
            self.finish_frame()
    
    def color_render_instance(self,
            instance_name,
            set_mesh_attrib_pointers=True):
        instance_data = self.scene_description['instances'][instance_name]
        if instance_data['hidden']:
            return
        
        instance_mesh = instance_data['mesh_name']
        instance_material = instance_data['material_name']
        material_data = (
                self.scene_description['materials'][instance_material])
        material_properties = numpy.array([
                material_data['ka'],
                material_data['kd'],
                material_data['ks'],
                material_data['shine']])
        image_light_material_properties = numpy.array([
                material_data['image_light_kd'],
                material_data['image_light_ks'],
                material_data['image_light_blur_reflection']])
        #glTexParameterf(
        #        GL_TEXTURE_CUBE_MAP,
        #        GL_TEXTURE_MIN_LOD,
        #        material_data['image_light_blur_reflection'])
        mesh_buffers = self.gl_data['mesh_buffers'][instance_mesh]
        material_buffers = self.gl_data['material_buffers'][instance_material]
        mesh_data = self.loaded_data['meshes'][instance_mesh]
        num_triangles = len(mesh_data['faces'])
        
        do_textured_mesh = len(mesh_data['uvs']) > 0
        
        if do_textured_mesh:
            location_data = self.gl_data['textured_shader']['locations']
        else:
            location_data = self.gl_data['vertex_color_shader']['locations']
        
        glUniformMatrix4fv(
                location_data['model_pose'],
                1, GL_TRUE,
                instance_data['transform'].astype(numpy.float32))
        
        glUniform4fv(
                location_data['material_properties'],
                1, material_properties.astype(numpy.float32))
        
        glUniform3fv(
                location_data['image_light_material_properties'], 1,
                image_light_material_properties)
        
        mesh_buffers['face_buffer'].bind()
        mesh_buffers['vertex_buffer'].bind()
        
        if do_textured_mesh:
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, material_buffers['texture'])
        
        try:
            # SOMETHING BETWEEN HERE...
            if set_mesh_attrib_pointers:
                glEnableVertexAttribArray(location_data['vertex_position'])
                glEnableVertexAttribArray(location_data['vertex_normal'])
                if do_textured_mesh:
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
                else:
                    glEnableVertexAttribArray(location_data['vertex_color'])
                    stride = (3+3+3) * 4
                    glVertexAttribPointer(
                            location_data['vertex_position'],
                            3, GL_FLOAT, False, stride,
                            mesh_buffers['vertex_buffer'])
                    glVertexAttribPointer(
                            location_data['vertex_normal'],
                            3, GL_FLOAT, False, stride,
                            mesh_buffers['vertex_buffer']+((3)*4))
                    glVertexAttribPointer(
                            location_data['vertex_color'],
                            3, GL_FLOAT, False, stride,
                            mesh_buffers['vertex_buffer']+((3+3)*4))
            # AND HERE TAKES ~40% of the rendering time
            
            glDrawElements(
                    GL_TRIANGLES,
                    num_triangles*3,
                    GL_UNSIGNED_INT,
                    None)
        
        finally:
            mesh_buffers['face_buffer'].unbind()
            mesh_buffers['vertex_buffer'].unbind()
            glBindTexture(GL_TEXTURE_2D, 0)
    
    def color_render_depthmap_instance(self, depthmap_instance_name):
        depthmap_instance_data = self.scene_description['depthmap_instances'][
                depthmap_instance_name]
        instance_depthmap = depthmap_instance_data['depthmap_name']
        instance_material = depthmap_instance_data['material_name']
        material_data = (
                self.scene_description['materials'][instance_material])
        depthmap_buffers = self.gl_data['depthmap_buffers'][instance_depthmap]
        material_buffers = self.gl_data['material_buffers'][instance_material]
        depth_data = self.loaded_data['depthmaps'][instance_depthmap]
        location_data = self.gl_data['textured_depthmap_shader']['locations']
        
        glUniformMatrix4fv(
                location_data['model_pose'],
                1, GL_TRUE,
                depthmap_instance_data['transform'].astype(numpy.float32))
        
        depthmap_data = self.scene_description['depthmaps'][instance_depthmap]
        focal_length = numpy.array(
                depthmap_data['focal_length'],
                dtype = numpy.float32)
        width = depthmap_data['width']
        height = depthmap_data['height']
        glUniform2fv(
                location_data['focal_length'],
                1, focal_length)
        glUniform1i(location_data['width'], width)
        glUniform1i(location_data['height'], height)
        
        glPointSize(depthmap_instance_data['point_size'])
        
        depthmap_buffers['depth_buffer'].bind()
        depthmap_buffers['index_buffer'].bind()
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, material_buffers['texture'])
        try:
            glEnableVertexAttribArray(location_data['vertex_depth'])
            
            stride = 4
            glVertexAttribPointer(
                    location_data['vertex_depth'],
                    1, GL_FLOAT, False, stride,
                    depthmap_buffers['depth_buffer'])
            depth_data = self.loaded_data['depthmaps'][depthmap_instance_name]
            num_points = depth_data.shape[0] * depth_data.shape[1]
            glDrawElements(
                    GL_POINTS,
                    num_points,
                    GL_UNSIGNED_INT,
                    None)
        
        finally:
            depthmap_buffers['depth_buffer'].unbind()
            depthmap_buffers['index_buffer'].unbind()
            glBindTexture(GL_TEXTURE_2D, 0)
    
    def render_background(self, image_light_name, flip_y=True):
        glUseProgram(self.gl_data['background_shader']['program'])
        
        mesh_buffers = self.gl_data['mesh_buffers']['BACKGROUND']
        light_buffers = self.gl_data['light_buffers'][image_light_name]
        num_triangles = 2
        
        location_data = self.gl_data['background_shader']['locations']
        
        # set the camera's pose
        camera_pose = self.scene_description['camera']['pose']
        glUniformMatrix4fv(
                location_data['camera_pose'],
                1, GL_TRUE,
                camera_pose.astype(numpy.float32))
        
        # set the camera's projection matrix
        projection_matrix = self.scene_description['camera']['projection']
        if flip_y:
            projection_matrix = numpy.dot(
                    projection_matrix,
                    numpy.array([
                        [ 1, 0, 0, 0],
                        [ 0,-1, 0, 0],
                        [ 0, 0, 1, 0],
                        [ 0, 0, 0, 1]]))
        glUniformMatrix4fv(
                location_data['projection_matrix'],
                1, GL_TRUE,
                projection_matrix.astype(numpy.float32))
        
        light_data = self.scene_description['image_lights'][image_light_name]
        
        # set the offset matrix
        offset_matrix = light_data['offset_matrix']
        glUniformMatrix4fv(
                location_data['offset_matrix'],
                1, GL_TRUE,
                offset_matrix.astype(numpy.float32))
        
        # set the blur
        blur = light_data['blur']
        glUniform1f(location_data['blur'], blur)
        #glUniform1f(location_data['blur'], 0)
        
        mesh_buffers['face_buffer'].bind()
        mesh_buffers['vertex_buffer'].bind()
        
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(
                GL_TEXTURE_CUBE_MAP,
                light_buffers['reflection_texture'])
        #glTexParameterf(
        #        GL_TEXTURE_CUBE_MAP,
        #        GL_TEXTURE_MIN_LOD,
        #        0)
        # THIS IS WEIRD... THE MIN_LOD LOOKS BETTER FOR THE REFLECTIONS,
        # BUT WORSE HERE.  MAYBE THE RIGHT THING IS TO GENERATE BLURRED
        # MIPMAPS AND DO EXPLICIT LOD LOOKUPS INSTEAD OF BIAS???
        
        try:
            glDrawElements(
                    GL_TRIANGLES,
                    2*3,
                    GL_UNSIGNED_INT,
                    None)
        
        finally:
            mesh_buffers['face_buffer'].unbind()
            mesh_buffers['vertex_buffer'].unbind()
            glBindTexture(GL_TEXTURE_CUBE_MAP, 0)
    
    #---------------------------------------------------------------------------
    # mask_render methods
    #---------------------------------------------------------------------------
    def mask_render(self, instances=None, flip_y=True):
        
        # clear
        self.clear_frame()
        
        # turn on the shader
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
            if flip_y:
                projection_matrix = numpy.dot(
                        projection_matrix,
                        numpy.array([
                            [1,0,0,0],
                            [0,-1,0,0],
                            [0,0,1,0],
                            [0,0,0,1]]))
            glUniformMatrix4fv(
                    location_data['projection_matrix'],
                    1, GL_TRUE,
                    projection_matrix.astype(numpy.float32))
            
            # render all instances
            if instances is None:
                instances = self.scene_description['instances']
            for instance_name in instances:
                self.mask_render_instance(instance_name)
            
        finally:
            glUseProgram(0)
        
        glFinish()
    
    def mask_render_instance(self, instance_name):
        instance_data = self.scene_description['instances'][instance_name]
        if instance_data['hidden']:
            return
        
        instance_mesh = instance_data['mesh_name']
        mask_color = instance_data['mask_color']
        mesh_buffers = self.gl_data['mesh_buffers'][instance_mesh]
        mesh = self.loaded_data['meshes'][instance_mesh]
        
        location_data = self.gl_data['mask_shader']['locations']
        
        glUniformMatrix4fv(
                location_data['model_pose'],
                1, GL_TRUE,
                numpy.array(instance_data['transform'], dtype=numpy.float32))
        
        glUniform3fv(
                location_data['mask_color'],
                1, numpy.array(mask_color, dtype=numpy.float32))
        
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
                    len(mesh.faces)*3,
                    GL_UNSIGNED_INT,
                    None)
        
        finally:
            mesh_buffers['face_buffer'].unbind()
            mesh_buffers['vertex_buffer'].unbind()
    
    #---------------------------------------------------------------------------
    # coord_render methods
    #---------------------------------------------------------------------------
    def coord_render(self, instances=None, flip_y=True):
        
        #clear
        self.clear_frame()
        
        # turn on the shader
        glUseProgram(self.gl_data['coord_shader']['program'])
        
        try:
            location_data = self.gl_data['coord_shader']['locations']
            camera_pose = self.scene_description['camera']['pose']
            glUniformMatrix4fv(
                    location_data['camera_pose'],
                    1, GL_TRUE,
                    camera_pose.astype(numpy.float32))
            
            projection_matrix = self.scene_description['camera']['projection']
            if flip_y:
                projection_matrix = numpy.dot(
                        projection_matrix,
                        numpy.array([
                            [1,0,0,0],
                            [0,-1,0,0],
                            [0,0,1,0],
                            [0,0,0,1]]))
            glUniformMatrix4fv(
                    location_data['projection_matrix'],
                    1, GL_TRUE,
                    projection_matrix.astype(numpy.float32))
            
            # render all instances
            if instances is None:
                instances = self.scene_description['instances']
            for instance_name in instances:
                self.coord_render_instance(instance_name)
        
        finally:
            glUseProgram(0)
        
        glFinish()
    
    def coord_render_instance(self, instance_name):
        instance_data = self.scene_description['instances'][instance_name]
        if instance_data['hidden']:
            return
        
        instance_mesh = instance_data['mesh_name']
        coord_box = instance_data['coord_box']
        mesh_buffers = self.gl_data['mesh_buffers'][instance_mesh]
        mesh = self.loaded_data['meshes'][instance_mesh]
        
        location_data = self.gl_data['coord_shader']['locations']
        
        glUniformMatrix4fv(
                location_data['model_pose'],
                1, GL_TRUE,
                numpy.array(instance_data['transform'], dtype=numpy.float32))
        
        glUniform3fv(
                location_data['box_min'],
                1, numpy.array(coord_box[0], dtype=numpy.float32))
    
        glUniform3fv(
                location_data['box_max'],
                1, numpy.array(coord_box[1], dtype=numpy.float32))
        
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
                    len(mesh.faces)*3,
                    GL_UNSIGNED_INT,
                    None)
        
        finally:
            mesh_buffers['face_buffer'].unbind()
            mesh_buffers['vertex_buffer'].unbind()
    
    #---------------------------------------------------------------------------
    # misc render methods
    #---------------------------------------------------------------------------
    def render_points(self, points, color, point_size = 1, flip_y = True):
        glPushMatrix()
        try:
            projection_matrix = self.scene_description['camera']['projection']
            if flip_y:
                projection_matrix = numpy.dot(projection_matrix, numpy.array([
                        [1, 0, 0, 0],
                        [0,-1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]]))
            glMultMatrixf(numpy.transpose(numpy.dot(
                    projection_matrix,
                    self.scene_description['camera']['pose'])))
            
            glColor3f(*color)
            glPointSize(point_size)
            glBegin(GL_POINTS)
            for point in points:
                glVertex3f(*point)
            glEnd()
        finally:
            glPopMatrix()
        glFinish()
    
    def render_line(self, start, end, color, flip_y = True, finish = True):
        glPushMatrix()
        try:
            projection_matrix = self.scene_description['camera']['projection']
            if flip_y:
                projection_matrix = numpy.dot(projection_matrix, numpy.array([
                        [1, 0, 0, 0],
                        [0,-1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]]))
            glMultMatrixf(numpy.transpose(numpy.dot(
                    projection_matrix,
                    self.scene_description['camera']['pose'])))
            
            glColor3f(*color)
            glBegin(GL_LINES)
            glVertex3f(*start)
            glVertex3f(*end)
            glEnd()
        finally:
            glPopMatrix()
        if finish:
            self.finish_frame()
    
    def render_transform(self, transform, axis_length = 0.1, flip_y = True):
        glPushMatrix()
        try:
            projection_matrix = self.scene_description['camera']['projection']
            if flip_y:
                projection_matrix = numpy.dot(projection_matrix, numpy.array([
                        [1, 0, 0, 0],
                        [0,-1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]]))
            glMultMatrixf(numpy.transpose(numpy.dot(numpy.dot(
                    projection_matrix,
                    self.scene_description['camera']['pose']),
                    transform)))
        
            glColor3f(1., 0., 0.)
            glBegin(GL_LINES)
            glVertex3f(0., 0., 0.)
            glVertex3f(axis_length, 0., 0.)
            glEnd()
            
            glColor3f(0., 1., 0.)
            glBegin(GL_LINES)
            glVertex3f(0., 0., 0.)
            glVertex3f(0., axis_length, 0.)
            glEnd()
            
            glColor3f(0., 0., 1.)
            glBegin(GL_LINES)
            glVertex3f(0., 0., 0.)
            glVertex3f(0., 0., axis_length)
            glEnd()
            
            glColor3f(1., 0., 1.)
            glBegin(GL_LINES)
            glVertex3f(0., 0., 0.)
            glVertex3f(-axis_length, 0., 0.)
            glEnd()
            
            glColor3f(1., 1., 0.)
            glBegin(GL_LINES)
            glVertex3f(0., 0., 0.)
            glVertex3f(0., -axis_length, 0.)
            glEnd()
            
            glColor3f(0., 1., 1.)
            glBegin(GL_LINES)
            glVertex3f(0., 0., 0.)
            glVertex3f(0., 0., -axis_length)
            glEnd()
        
        finally:
            glPopMatrix()
        
    def render_vertices(self, instance_name, flip_y = True):
        #TODO: add this for debugging purposes
        raise NotImplementedError
