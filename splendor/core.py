# system
import math
import json
import os
import ctypes

# opengl
from OpenGL import GL

# numpy
import numpy

# splendor
from splendor.assets import AssetLibrary
import splendor.camera as camera
import splendor.masks as masks
from splendor.shader_library import ShaderLibrary
import splendor.obj_mesh as obj_mesh
from splendor.image import load_image, load_depth, validate_texture
import splendor.json_numpy as json_numpy
from splendor.exceptions import SplendorException, SplendorEmptyMeshException
from splendor.primitives import make_primitive
from splendor.shaders.lighting_model import MAX_SHADOW_CASTERS

max_num_lights = 8

class SplendorRender:
    """
    Core rendering functionality.
    
    Contains scene data, methods for manipulating it and for performing
    different rendering operations.
    """
    _global_parameters = (
            'ambient_color', 'background_color', 'active_image_light')
    _asset_types = (
            ('mesh', 'meshes'),
            ('texture', 'textures'),
            ('cubemap', 'cubemaps'),
            ('material', 'materials'),
            ('image_light', 'image_lights'),
            ('depthmap', 'depthmaps'))
    _instance_types = (
            ('instance', 'instances'),
            ('depthmap_instance', 'depthmap_instances'),
            ('point_light', 'point_lights'),
            ('direction_light', 'direction_lights'),
            ('coord_frame', 'coord_frames'),
            ('frustum', 'frustums'),
            ('line_set', 'line_sets'),
            ('point_cloud', 'point_clouds'),
            ('box', 'boxes'),
            ('arrow', 'arrows'))

    def __init__(self,
        assets=None,
    ):
        """
        SplendorRender initialization

        Parameters
        ----------
        assets : str or AssetLibrary, optional
            Either a path pointing to an asset library cfg file or an
            AssetLibrary object.  This is used to load assets such as meshes
            and textures by name rather than their full path.  If not provided,
            this will load the splendor-render default asset library.
        """
        # asset library
        if isinstance(assets, AssetLibrary):
            self.asset_library = assets
        else:
            self.asset_library = AssetLibrary(assets)

        # scene data
        self.scene_description = {
            'meshes':{},
            'depthmaps':{},
            'materials':{},
            'textures':{},
            'cubemaps':{},
            'instances':{},
            'depthmap_instances':{},
            'background_color':numpy.array([0,0,0,0]),
            'ambient_color':numpy.array([0,0,0]),
            'point_lights':{},
            'direction_lights':{},
            'coord_frames':{},
            'frustums':{},
            'line_sets':{},
            'point_clouds':{},
            'boxes':{},
            'arrows':{},
            'cameras':{},
            'sensors':{},
            'image_lights':{},
            'active_image_light':None,
        }

        self.loaded_data = {
            'meshes':{},
            'depthmaps':{},
            'textures':{},
            'cubemaps':{},
            'irradiance_sh':{},
        }

        self.gl_data = {
            'mesh_buffers':{},
            'depthmap_buffers':{},
            'texture_buffers':{},
            'cubemap_buffers':{},
            'sensor_buffers':{},
            'coord_frame_buffers': None,
        }
        
        self.opengl_init()
        self.shader_library = ShaderLibrary()
        self._init_coord_frame_buffers()
    
    def get_json_description(self, **kwargs):
        """
        Produce a json description of the scene for serialization.
        
        Parameters
        ----------
        **kwargs :
            All named arguments are passed through to json.dumps in order to
            provide formatting options such as indentation.
        """
        return json.dumps(
                self.scene_description, cls=json_numpy.NumpyEncoder, **kwargs)
    
    def opengl_init(self):
        """
        Initialize OpenGL.
        """
        renderer = GL.glGetString(GL.GL_RENDERER).decode('utf-8')
        version = GL.glGetString(GL.GL_VERSION).decode('utf-8')

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_SCISSOR_TEST)
        GL.glEnable(GL.GL_TEXTURE_CUBE_MAP_SEAMLESS)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glDepthFunc(GL.GL_LESS)
        GL.glDepthRange(0.0, 1.0)

        GL.glClearColor(0.,0.,0.,0.)

        # Empty VAO for attribute-less draws (e.g. fullscreen warp pass)
        self.empty_vao = GL.glGenVertexArrays(1)
    
    def viewport_scissor(self, x, y, width, height):
        GL.glViewport(x, y, width, height)
        GL.glScissor(x, y, width, height)
    
    # scene methods ============================================================
    
    def load_scene(self, scene, clear_scene=False, reload_assets=False):
        """
        Load a scene from JSON data.
        
        Parameters
        ----------
        scene : dict
            JSON data representing the scene to load
        clear_scene : bool, default=False
            Clear all data in the scene before loading the new scene.
        reload_assets : bool, default=False
            Reload assets that exist both in the new scene data and the
            already loaded scene data (irrelevant if clear_scene=True)
        """
        
        if clear_scene:
            self.clear_scene()

        if isinstance(scene, str):
            scene = self.asset_library['scenes'][scene]
            scene = json.load(open(scene))

        # meshes, depthmaps, textures, cubemaps, materials, image_lights
        for singular, plural in self._asset_types:
            if plural in scene:
                for asset_name, asset_args in scene[plural].items():
                    exists_fn = getattr(self, singular + '_exists')
                    if reload_assets or not exists_fn(asset_name):
                        load_fn = getattr(self, 'load_' + singular)
                        load_fn(asset_name, **asset_args)

        # instances, depthmap_instances, point_lights, direction_lights
        for singular, plural in self._instance_types:
            if plural in scene:
                for instance_name, instance_args in scene[plural].items():
                    add_fn = getattr(self, 'add_' + singular)
                    add_fn(instance_name, **instance_args)

        for global_parameter in self._global_parameters:
            if global_parameter in scene:
                set_fn = getattr(self, 'set_' + global_parameter)
                set_fn(scene[global_parameter])

        if 'cameras' in scene:
            for camera_name, camera_args in scene['cameras'].items():
                self.load_camera(camera_name, **camera_args)

        if 'sensors' in scene:
            for sensor_name, sensor_args in scene['sensors'].items():
                self.load_sensor(sensor_name, **sensor_args)

    def clear_scene(self):
        """
        Clears all assets and instances from the scene.
        """
        for singular, plural in self._asset_types:
            getattr(self, 'clear_' + plural)()
        for singular, plural in self._instance_types:
            getattr(self, 'clear_' + plural)()
        self.set_ambient_color([0,0,0])
        self.set_background_color([0,0,0,0])
        self.scene_description['active_image_light'] = None
        self.clear_cameras()
        self.clear_sensors()

    # global settings ==========================================================
    
    def set_ambient_color(self, color):
        """
        Sets the ambient light color for the scene.
        
        Parameters
        ----------
        color : array-like in [0-1]
            An ambient color which will be added to the lighting contribution
            of all lit objects.
        """
        self.scene_description['ambient_color'] = numpy.array(color)
    
    def get_ambient_color(self):
        """
        Gets the ambient light color for the scene.
        
        Returns
        -------
        3-channel array in [0-1]
        """
        return self.scene_description['ambient_color']

    def set_background_color(self, background_color):
        """
        Sets the background color for the scene.
        
        Parameters
        ----------
        background_color : 3 or 4 channel array-like in [0-1]
            If three channels are provided, an alpha channel of 1. is assumed.
        """
        if len(background_color) == 3:
            background_color = tuple(background_color) + (1,)
        self.scene_description['background_color'] = numpy.array(
                background_color)
    
    def get_background_color(self):
        """
        Gets the background color for the scene.
        
        Returns
        -------
        4-channel array in [0-1]
        """
        return self.scene_description['background_color']
    
    def set_active_image_light(self, image_light):
        """
        Sets the active image light.
        
        Parameters
        ----------
        image_light : str
            The name of the image light to make active.
        """
        self.scene_description['active_image_light'] = image_light
    
    def get_active_image_light(self):
        """
        Gets the active image light.
        
        Returns
        -------
        str : the name of the image light
        """
        return self.scene_description['active_image_light']

    # camera methods ===========================================================

    def load_camera(self,
        name,
        view_matrix=None,
        projection=None,
        radial_k1=0.,
        radial_k2=0.,
    ):
        """
        Load a named camera into the scene.

        Parameters:
        -----------
        name : str
            Unique name for this camera.
        view_matrix : 4x4 array-like or azimuthal parameters, optional
            The view matrix (inverse of the camera's world pose).
            Accepts the same formats as camera.view_matrix().
            Defaults to the identity (camera at origin looking down -Z).
        projection : 4x4 array-like, optional
            The projection matrix.  Defaults to a 90-degree FOV square
            projection.
        radial_k1 : float, default=0.
            First radial distortion coefficient.
        radial_k2 : float, default=0.
            Second radial distortion coefficient.
        """
        if view_matrix is None:
            view_matrix = numpy.eye(4)
        if projection is None:
            projection = camera.projection_matrix(math.radians(90.), 1.0)
        view_matrix = camera.view_matrix(view_matrix)
        self.scene_description['cameras'][name] = {
            'view_matrix': numpy.array(view_matrix),
            'projection': numpy.array(projection),
            'radial_k1': float(radial_k1),
            'radial_k2': float(radial_k2),
        }

    def remove_camera(self, name):
        del self.scene_description['cameras'][name]

    def camera_exists(self, name):
        return name in self.scene_description['cameras']

    def clear_cameras(self):
        self.scene_description['cameras'].clear()

    def set_camera_view_matrix(self, name, view_matrix):
        """
        Set the view matrix for a named camera.

        Parameters:
        -----------
        name : str
        view_matrix : 4x4 matrix or azimuthal parameters
        """
        view_matrix = camera.view_matrix(view_matrix)
        self.scene_description['cameras'][name]['view_matrix'] = view_matrix

    def get_camera_view_matrix(self, name):
        return self.scene_description['cameras'][name]['view_matrix']

    def set_camera_projection(self, name, projection):
        self.scene_description['cameras'][name]['projection'] = numpy.array(
            projection)

    def get_camera_projection(self, name):
        return self.scene_description['cameras'][name]['projection']

    def set_camera_radial_k1(self, name, radial_k1):
        self.scene_description['cameras'][name]['radial_k1'] = float(radial_k1)

    def get_camera_radial_k1(self, name):
        return self.scene_description['cameras'][name]['radial_k1']

    def set_camera_radial_k2(self, name, radial_k2):
        self.scene_description['cameras'][name]['radial_k2'] = float(radial_k2)

    def get_camera_radial_k2(self, name):
        return self.scene_description['cameras'][name]['radial_k2']

    def camera_frame_scene(self, camera_name, multiplier=3.0, *args, **kwargs):
        bbox = self.get_instance_center_bbox()
        view_matrix = camera.frame_bbox(
            bbox, self.get_camera_projection(camera_name), multiplier,
            *args, **kwargs)
        self.set_camera_view_matrix(camera_name, view_matrix)

    # sensor methods ===========================================================

    def load_sensor(self,
        name,
        width,
        height,
        enable_radial_distortion=False,
        anti_alias=True,
        anti_alias_samples=8,
        color_format=GL.GL_RGBA8,
        depth_only=False,
    ):
        """
        Load a named sensor (offscreen render target).

        A sensor owns one or more framebuffers and can be passed to render
        functions to direct output offscreen.  Use read_sensor() to retrieve
        pixel data after rendering.

        Parameters:
        -----------
        name : str
            Unique name for this sensor.
        width : int
        height : int
        enable_radial_distortion : bool, default=False
            If True, allocates an intermediate framebuffer for the two-stage
            radial distortion pass.
        anti_alias : bool, default=True
        anti_alias_samples : int, default=8
        color_format : GL enum, default=GL.GL_RGBA8
            GL.GL_RGBA8 or GL.GL_RGBA32F
        depth_only : bool, default=False
            If True, no color attachment is created.  The depth buffer is a
            sampleable texture (depth_texture).  Useful for shadow maps.
            Incompatible with enable_radial_distortion.
        """
        assert not (depth_only and enable_radial_distortion), (
            'depth_only sensors do not support radial distortion')
        from splendor.frame_buffer import FrameBufferWrapper
        main_fbo = FrameBufferWrapper(
            width, height,
            anti_alias=anti_alias,
            anti_alias_samples=anti_alias_samples,
            color_format=color_format,
            depth_only=depth_only,
        )
        sensor_buffers = {'main_fbo': main_fbo}
        if enable_radial_distortion:
            intermediate_fbo = FrameBufferWrapper(
                width, height,
                anti_alias=anti_alias,
                anti_alias_samples=anti_alias_samples,
                texture_output=True,
                color_format=color_format,
            )
            sensor_buffers['intermediate_fbo'] = intermediate_fbo
        self.scene_description['sensors'][name] = {
            'width': width,
            'height': height,
            'enable_radial_distortion': enable_radial_distortion,
            'depth_only': depth_only,
        }
        self.gl_data['sensor_buffers'][name] = sensor_buffers

    def remove_sensor(self, name):
        del self.scene_description['sensors'][name]
        del self.gl_data['sensor_buffers'][name]

    def clear_sensors(self):
        for name in list(self.scene_description['sensors'].keys()):
            self.remove_sensor(name)

    def sensor_exists(self, name):
        return name in self.scene_description['sensors']

    def _bind_sensor(self, name):
        """Bind the sensor's main framebuffer and set the viewport."""
        self.gl_data['sensor_buffers'][name]['main_fbo'].enable()

    def read_sensor(self, name, **kwargs):
        """
        Read pixel data from a named sensor.

        Parameters:
        -----------
        name : str
        **kwargs :
            Passed through to FrameBufferWrapper.read_pixels().
            Useful args: read_alpha, read_depth, projection.

        Returns:
        --------
        numpy array

        """
        return self.gl_data['sensor_buffers'][name]['main_fbo'].read_pixels(
            **kwargs)

    def display_sensor(self, name):
        """
        Blit a sensor's output to the screen (FBO 0).

        Call this after rendering into a sensor to show the result in an
        interactive window.  The caller is responsible for having the correct
        windowing context active.

        Parameters:
        -----------
        name : str
        """
        from OpenGL import GL
        sensor_data = self.scene_description['sensors'][name]
        fbo_wrapper = self.gl_data['sensor_buffers'][name]['main_fbo']
        w, h = sensor_data['width'], sensor_data['height']

        blit_mask = GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT

        # resolve multisampling if needed
        if fbo_wrapper.anti_alias:
            GL.glBindFramebuffer(
                GL.GL_READ_FRAMEBUFFER, fbo_wrapper.frame_buffer_multi)
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, fbo_wrapper.frame_buffer)
            GL.glBlitFramebuffer(
                0, 0, w, h, 0, 0, w, h,
                blit_mask, GL.GL_NEAREST)

        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, fbo_wrapper.frame_buffer)
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, 0)
        GL.glBlitFramebuffer(
            0, 0, w, h, 0, 0, w, h,
            blit_mask, GL.GL_NEAREST)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    # mesh methods =============================================================
    
    def load_mesh(self,
            name,
            mesh_asset = None,
            mesh_path = None,
            mesh_data = None,
            mesh_primitive = None,
            scale = 1.0,
            color_mode = 'textured'):
        """
        Load a mesh.
        
        Loads a mesh into memory but does not place it in the scene.  In order
        to be rendered, an instance must be created that uses this mesh.
        
        Parameters:
        -----------
        name : str
            Name of the mesh, must be unique to this scene among other meshes
        mesh_asset : str, optional
            Local file in an asset directory to load
        mesh_path : str, optional
            Full path to a mesh file
        mesh_data : dict, optional
            Dictionary containing the vertices, normals and faces of this mesh
        mesh_primitive : dict, optional
            Dictionary containing args to the primitives.make_primitive function
        scale : float, default=1.0
            Global scale for the mesh
        color_mode : {"textured", "vertex_color", "flat_color"}
            Describes the color mode of the surface.  Can be one of:
            "textured" : requires uvs
            "vertex_color" : requires specified vertex colors
            "flat" : the entire surface will be a single flat color
        """
        
        assert color_mode in ('textured', 'vertex_color', 'flat_color')
        
        # if a mesh asset name was provided, load that
        if mesh_asset is not None:
            asset_path = self.asset_library['meshes'][mesh_asset]
            mesh = obj_mesh.load_mesh(asset_path, scale=scale)
            self.scene_description['meshes'][name] = {
                'mesh_asset':mesh_asset
            }

        # otherwise if a mesh path was provided, load that
        elif mesh_path is not None:
            mesh = obj_mesh.load_mesh(mesh_path, scale=scale)
            self.scene_description['meshes'][name] = {
                'mesh_path':mesh_path
            }

        # otherwise if mesh data was provided, load that
        elif mesh_data is not None:
            mesh = mesh_data
            self.scene_description['meshes'][name] = {
                'mesh_data':mesh_data
            }
        
        # otherwise if a primitive is provided, load that
        elif mesh_primitive is not None:
            mesh = make_primitive(**mesh_primitive)
            self.scene_description['meshes'][name] = {
                'mesh_primitive':mesh_primitive
            }
        
        else:
            raise SplendorException(
                    'Must supply a "mesh_asset", "mesh_path", "mesh_data" '
                    ' or "mesh_primitive" argument when loading a mesh')

        self.scene_description['meshes'][name]['color_mode'] = color_mode

        # create mesh buffers and load the mesh data
        mesh_buffers = {}

        vertex_floats = numpy.array(mesh['vertices'], dtype=numpy.float32)
        if len(vertex_floats) == 0:
            raise SplendorEmptyMeshException
        if vertex_floats.shape[1] > 3:
            vertex_floats = vertex_floats[:,:3]
        normal_floats = numpy.array(mesh['normals'], dtype=numpy.float32)
        if normal_floats.shape[1] > 3:
            normal_floats = normal_floats[:,:3]
        
        if color_mode == 'textured':
            #if not len(mesh['uvs']) and create_uvs:
            #    mesh['uvs'] = [[0,0] for _ in mesh['vertices']]
            assert 'uvs' in mesh
            uv_floats = numpy.array(mesh['uvs'], dtype=numpy.float32)
            if uv_floats.shape[1]:
                uv_floats = uv_floats[:,:2]
            combined_floats = numpy.concatenate(
                    (vertex_floats, normal_floats, uv_floats), axis=1)
            shader_name = 'textured_shader'
            stride = (3+3+2) * 4

        elif color_mode == 'vertex_color':
            assert 'vertex_colors' in mesh
            vertex_color_floats = numpy.array(
                    mesh['vertex_colors'], dtype=numpy.float32)
            combined_floats = numpy.concatenate(
                    (vertex_floats, normal_floats, vertex_color_floats), axis=1)
            shader_name = 'vertex_color_shader'
            stride = (3+3+3) * 4

        elif color_mode == 'flat_color':
            combined_floats = numpy.concatenate(
                    (vertex_floats, normal_floats), axis=1)
            shader_name = 'flat_color_shader'
            stride = (3+3) * 4
        
        # make the vao
        mesh_buffers['vao'] = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(mesh_buffers['vao'])
        
        # make the vbo
        mesh_buffers['vbo'] = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, mesh_buffers['vbo'])
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER,
            combined_floats.nbytes,
            combined_floats,
            GL.GL_STATIC_DRAW,
        )
        
        # make the ebo
        face_ints = numpy.array(mesh['faces'], dtype=numpy.int32)
        mesh_buffers['ebo'] = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, mesh_buffers['ebo'])
        GL.glBufferData(
            GL.GL_ELEMENT_ARRAY_BUFFER,
            face_ints.nbytes,
            face_ints,
            GL.GL_STATIC_DRAW,
        )
        
        # vertex attribute setup
        shader_locations = self.shader_library.get_shader_locations(shader_name)
        
        GL.glVertexAttribPointer(
            shader_locations['vertex_position'],
            3, GL.GL_FLOAT, False, stride,
            ctypes.c_void_p(0),
        )
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(
            shader_locations['vertex_normal'],
            3, GL.GL_FLOAT, False, stride,
            ctypes.c_void_p((3)*4),
        )
        GL.glEnableVertexAttribArray(1)
        
        if color_mode == 'textured':
            GL.glVertexAttribPointer(
                shader_locations['vertex_uv'],
                2, GL.GL_FLOAT, False, stride,
                ctypes.c_void_p((3+3)*4),
            )
            GL.glEnableVertexAttribArray(2)
        elif color_mode == 'vertex_color':
            GL.glVertexAttribPointer(
                shader_locations['vertex_color'],
                3, GL.GL_FLOAT, False, stride,
                ctypes.c_void_p((3+3)*4),
            )
            GL.glEnableVertexAttribArray(2)

        # store the loaded and gl data
        self.loaded_data['meshes'][name] = mesh
        self.gl_data['mesh_buffers'][name] = mesh_buffers

    def load_background_mesh(self):
        """
        Load a square mesh placed almost at the far clipping plane to render
        the background onto.
        """
        
        if 'BACKGROUND' not in self.gl_data['mesh_buffers']:
            mesh_buffers = {}
            mesh_buffers['vao'] = GL.glGenVertexArrays(1)
            GL.glBindVertexArray(mesh_buffers['vao'])
            
            vertex_floats = numpy.array([
                    [-1,-1,0],
                    [-1, 1,0],
                    [ 1, 1,0],
                    [ 1,-1,0]])
            mesh_buffers['vbo'] = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, mesh_buffers['vbo'])
            GL.glBufferData(
                GL.GL_ARRAY_BUFFER,
                vertex_floats.nbytes,
                vertex_floats,
                GL.GL_STATIC_DRAW,
            )

            face_ints = numpy.array([
                    [0,1,2],
                    [2,3,0]], dtype=numpy.int32)
            mesh_buffers['ebo'] = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, mesh_buffers['ebo'])
            GL.glBufferData(
                GL.GL_ELEMENT_ARRAY_BUFFER,
                face_ints.nbytes,
                face_ints,
                GL.GL_STATIC_DRAW,
            )
            self.gl_data['mesh_buffers']['BACKGROUND'] = mesh_buffers

    def remove_mesh(self, name):
        """
        Deletes a mesh from the scene.
        
        Parameters:
        -----------
        name : str
        """
        
        del(self.scene_description['meshes'][name])
        GL.glDeleteVertexArrays(1, [self.gl_data['mesh_buffers'][name]['vao']])
        GL.glDeleteBuffers(2, [
            self.gl_data['mesh_buffers'][name]['vbo'],
            self.gl_data['mesh_buffers'][name]['ebo'],
        ])
        del(self.gl_data['mesh_buffers'][name])
        del(self.loaded_data['meshes'][name])

    def clear_meshes(self):
        """
        Deletes all meshes.
        """
        for name in list(self.scene_description['meshes'].keys()):
            self.remove_mesh(name)

    def list_meshes(self):
        """
        Returns:
        --------
        list :
            All mesh names in the scene.
        """
        return list(self.scene_description['meshes'].keys())

    def mesh_exists(self, name):
        """
        Parameters:
        -----------
        name : str
        
        Returns:
        --------
        bool
        """
        return name in self.scene_description['meshes']

    def get_mesh(self, name):
        """
        Parameters:
        -----------
        name : str
        
        Returns:
        --------
        str :
            The serialized description of the mesh.
        """
        return self.scene_description['meshes'][name]

    def get_mesh_color_mode(self, name):
        """
        Parameters:
        -----------
        name : str
        
        Returns:
        --------
        str : {"textured", "vertex_colors", "flat"}
        """
        return self.scene_description['meshes'][name]['color_mode']

    def get_mesh_stride(self, name):
        """
        Determines how many bytes correspond to each vertex.
        
        The different color modes require different per-vertex information
        which determines how the vertex data for the mesh can be packed.
        
        Parameters:
        -----------
        name : str
        
        Returns:
        --------
        int
        """
        color_mode = self.get_mesh_color_mode(name)
        if color_mode == 'textured':
            return (3+3+2) * 4
        elif color_mode == 'vertex_color':
            return (3+3+3) * 4
        elif color_mode == 'flat_color':
            return (3+3) * 4

    # depthmap methods =========================================================
    
    def load_depthmap(self,
            name,
            depthmap_asset = None,
            depthmap_path = None,
            depthmap_data = None,
            indices = None,
            focal_length = (1,1)):
        """
        Loads a depth map.
        
        Parameters:
        -----------
        name : str
            Name for the new depthmap.  Must be unique among other depthmap
            names in this scene.
        depthmap_asset : str, optional
            Local file name for the depthmap relative to the asset directories
        depthmap_path : str, optional
            A full path to a depthmap file
        depthmap_data : str, optional
            A numpy array with depthmap data
        indices : array, optional
            The indices for the VBO
        focal_length : tuple, default=(1,1)
            The x,y focal length of the camera that captured the depthmap for
            reprojection into 3D
        """
        
        if name in self.scene_description['depthmaps']:
            self.remove_depthmap(name)

        # if an asset was provided, load that
        if depthmap_asset is not None:
            depthmap_path = self.asset_library['depthmaps'][depthmap_asset]
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
            raise SplendorException(
                    'Must supply a "depthmap_asset", "depthmap_path" or a '
                    '"depthmap_data" argument when loading a depthmap')

        depthmap = numpy.array(depthmap, dtype=numpy.float32)

        self.scene_description['depthmaps'][name]['height'] = depthmap.shape[0]
        self.scene_description['depthmaps'][name]['width'] = depthmap.shape[1]
        self.scene_description['depthmaps'][name]['focal_length'] = focal_length

        depthmap_buffers = {}

        depthmap_buffers['vao'] = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(depthmap_buffers['vao'])

        # create depth vbo
        depth_floats = depthmap.flatten()
        depthmap_buffers['vbo'] = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, depthmap_buffers['vbo'])
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER,
            depth_floats.nbytes,
            depth_floats,
            GL.GL_STATIC_DRAW,
        )

        # create index ebo
        if indices is None:
            indices = numpy.arange(
                    depthmap.shape[0] * depthmap.shape[1],
                    dtype=numpy.int32)
        depthmap_buffers['ebo'] = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, depthmap_buffers['ebo'])
        GL.glBufferData(
            GL.GL_ELEMENT_ARRAY_BUFFER,
            indices.nbytes,
            indices,
            GL.GL_STATIC_DRAW,
        )

        # vertex attribute setup
        shader_locations = self.shader_library.get_shader_locations(
            'textured_depthmap_shader')
        GL.glVertexAttribPointer(
            shader_locations['vertex_depth'],
            1, GL.GL_FLOAT, False, 4,
            ctypes.c_void_p(0),
        )
        GL.glEnableVertexAttribArray(shader_locations['vertex_depth'])

        GL.glBindVertexArray(0)

        # store the loaded and gl data
        self.loaded_data['depthmaps'][name] = depthmap
        self.gl_data['depthmap_buffers'][name] = depthmap_buffers

    def remove_depthmap(self, name):
        """
        Deletes a depthmap from the scene.
        
        Parameters:
        -----------
        name : str
        """
        
        del(self.scene_description['depthmaps'][name])
        GL.glDeleteVertexArrays(
            1, [self.gl_data['depthmap_buffers'][name]['vao']])
        GL.glDeleteBuffers(2, [
            self.gl_data['depthmap_buffers'][name]['vbo'],
            self.gl_data['depthmap_buffers'][name]['ebo'],
        ])
        del(self.gl_data['depthmap_buffers'][name])
        del(self.loaded_data['depthmaps'][name])

    def clear_depthmaps(self):
        """
        Deletes all depthmaps.
        """
        for name in list(self.scene_description['depthmaps'].keys()):
            self.remove_depthmap(name)

    def list_depthmaps(self):
        """
        Returns:
        --------
        list :
            All depthmap names in the scene.
        """
        return list(self.scene_description['depthmaps'].keys())

    def depthmap_exists(self, depthmap):
        """
        Parameters:
        -----------
        name : str
        
        Returns:
        --------
        bool
        """
        return depthmap in self.scene_description['depthmaps']

    def get_depthmap(self, depthmap):
        """
        Parameters:
        -----------
        name : str
        
        Returns:
        --------
        str :
            The serialized description of the depthmap.
        """
        return self.scene_description['depthmaps'][depthmap]

    # image_light methods ======================================================
    
    def load_image_light(self,
        name,
        reflect_cubemap,
        irradiance_sh=None,
        irradiance_sh_asset=None,
        offset_matrix = numpy.eye(4),
        blur = 0.,
        diffuse_scale = 1.,
        diffuse_bias = 0.,
        reflect_gamma = 1.,
        reflect_bias = 0.,
        render_background = True,
        set_active = False,
        lock_to_camera = False,
        shadow_map = None,
        shadow_projection = None,
        shadow_pose = None,
        shadow_pcf_radius = 1,
        shadow_color = None,
    ):
        """
        Load an image light.
        
        Loads an image light into memory but does not make it active unless
        set_active=True.  Only one image_light can be active at a time.
        
        Parameters:
        -----------
        name : str
            Name of the image light, must be unique to this scene among other
            image lights
        reflect_cubemap : str
            The name of the cubemap to use for reflections and background.
        irradiance_sh : (9, 3) array-like, optional
            Spherical harmonic irradiance coefficients for diffuse lighting,
            as produced by cubemap_strip_to_sh().
        irradiance_sh_asset : str, optional
            Name of an SH coefficients asset in the asset library.
            Exactly one of irradiance_sh or irradiance_sh_asset must
            be provided.
        offset_matrix : 4x4 array-like, default=numpy.eye(4)
            An offset rotation matrix for the image light.
        blur : float, default=0.
            Blur to apply to the background when the background is visible.
        diffuse_bias : float, default=0.
            A bias added to the diffuse irradiance (artistic control).
        reflect_gamma : float, default=1.
            A gamma correction for the reflect component of the image light.
            Values above one increase the contrast in the reflections.
        reflect_bias : float, default=0.
            A bias for the reflection component of the image light.
        render_background : bool, default=True
            Whether or not to render the reflection maps as a background for
            the scene.
        set_active : bool, default=False
            If true, this image light will become the active image light in
            the scene.
        """
        
        image_light_data = {}
        image_light_data['reflect_cubemap'] = reflect_cubemap

        if irradiance_sh_asset is not None:
            sh_path = self.asset_library['irradiance_sh'][
                irradiance_sh_asset]
            with open(sh_path) as f:
                sh_array = numpy.array(json.load(f), dtype=numpy.float32)
            image_light_data['irradiance_sh_asset'] = irradiance_sh_asset
        elif irradiance_sh is not None:
            sh_array = numpy.array(irradiance_sh, dtype=numpy.float32)
            image_light_data['irradiance_sh'] = irradiance_sh
        else:
            raise SplendorException(
                'Must supply either "irradiance_sh" or '
                '"irradiance_sh_asset" when loading an image light')

        self.loaded_data['irradiance_sh'][name] = sh_array

        # Auto-compute shadow color from SH if not provided
        if shadow_color is not None:
            shadow_color_array = numpy.array(shadow_color, dtype=numpy.float32)
            image_light_data['shadow_color'] = shadow_color
        else:
            from splendor.image_light.diffuse import compute_shadow_color
            shadow_color_array = compute_shadow_color(sh_array)
        self.loaded_data['irradiance_sh'][name + '_shadow_color'] = (
            shadow_color_array)

        image_light_data['offset_matrix'] = numpy.array(offset_matrix)
        image_light_data['blur'] = blur
        image_light_data['render_background'] = render_background
        image_light_data['diffuse_scale'] = diffuse_scale
        image_light_data['diffuse_bias'] = diffuse_bias
        image_light_data['reflect_gamma'] = reflect_gamma
        image_light_data['reflect_bias'] = reflect_bias
        image_light_data['lock_to_camera'] = lock_to_camera
        image_light_data['shadow_map'] = shadow_map
        image_light_data['shadow_projection'] = (
            numpy.array(shadow_projection) if shadow_projection is not None else None)
        image_light_data['shadow_pose'] = (
            numpy.array(shadow_pose) if shadow_pose is not None else None)
        image_light_data['shadow_pcf_radius'] = int(shadow_pcf_radius)
        self.scene_description['image_lights'][name] = image_light_data
        
        self.load_background_mesh()
        
        if set_active:
            self.set_active_image_light(name)
    
    def remove_image_light(self, name):
        """
        Deletes an image light from the scene.
        
        Parameters:
        -----------
        name : str
        """
        del(self.scene_description['image_lights'][name])
        if name in self.loaded_data['irradiance_sh']:
            del(self.loaded_data['irradiance_sh'][name])
        shadow_key = name + '_shadow_color'
        if shadow_key in self.loaded_data['irradiance_sh']:
            del(self.loaded_data['irradiance_sh'][shadow_key])

        # delete the background mesh if there are no image lights left
        if len(self.scene_description['image_lights']) == 0:
            #self.gl_data['mesh_buffers']['BACKGROUND']['vertex_buffer'].delete()
            #self.gl_data['mesh_buffers']['BACKGROUND']['face_buffer'].delete()
            GL.glDeleteVertexArrays(
                1, [self.gl_data['mesh_buffers']['BACKGROUND']['vao']])
            GL.glDeleteBuffers(1, [
                self.gl_data['mesh_buffers']['BACKGROUND']['vbo'],
                self.gl_data['mesh_buffers']['BACKGROUND']['ebo'],
            ])
            del(self.gl_data['mesh_buffers']['BACKGROUND'])

    def clear_image_lights(self):
        """
        Deletes all image lights.
        """
        for image_light in list(self.scene_description['image_lights'].keys()):
            self.remove_image_light(image_light)
        self.set_active_image_light(None)

    def list_image_lights(self):
        """
        Returns:
        --------
        list :
            All image light names in the scene.
        """
        return list(self.scene_description['image_lights'].keys())

    def image_light_exists(self, image_light):
        """
        Parameters:
        -----------
        name : str
        
        Returns:
        --------
        bool
        """
        return image_light in self.scene_description['image_lights']

    def get_image_light(self, image_light):
        """
        Parameters:
        -----------
        name : str

        Returns:
        --------
        str :
            The serialized description of the image light.
        """
        return self.scene_description['image_lights'][image_light]


    # texture methods ==========================================================
    
    def load_texture(
        self,
        name,
        texture_asset=None,
        texture_path=None,
        texture_data=None,
        crop=None,
    ):
        """
        Replace the texture for a material
        
        Parameters:
        -----------
        name : str
            The name of the material to replace the texture
        texture : array-like or str
            Either an asset, path or raw image data
        crop : 4-tuple, optional
            Bottom, left, top, right crop values for the image
        """
        
        # if a texture asset name was provided, load that
        if texture_asset is not None:
            asset_path = self.asset_library['textures'][texture_asset]
            texture = load_image(asset_path)
            self.scene_description['textures'][name] = {
                'texture_asset':texture_asset
            }

        # otherwise if a texture path was provided, load that
        elif texture_path is not None:
            texture = load_image(texture_path)
            self.scene_description['textures'][name] = {
                'texture_path':texture_path
            }

        # otherwise if texture data was provided, load that
        elif texture_data is not None:
            texture = texture_data
            self.scene_description['textures'][name] = {
                'texture_data':texture_data
            }
        
        else:
            raise SplendorException(
                    'Must supply a "texture_asset", "texture_path" or '
                    '"texture_data" argument when loading a texture')
        
        # crop if necessary
        if crop is not None:
            texture = texture[crop[0]:crop[2], crop[1]:crop[3]]
        
        # validate and store the texture
        validate_texture(texture)
        self.loaded_data['textures'][name] = texture
        
        # if an entry for this texture doesn't exist in texture_buffers
        # make one
        self.gl_data['texture_buffers'].setdefault(name, {})
        
        # delete any old texture that exists
        if 'texture' in self.gl_data['texture_buffers'][name]:
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
            GL.glDeleteTextures(
                    [self.gl_data['texture_buffers'][name]['texture']])
        
        # make the new texture
        self.gl_data['texture_buffers'][name]['texture'] = (
            GL.glGenTextures(1))
        
        # copy the texture to the GPU
        texture_buffers = self.gl_data['texture_buffers'][name]
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture_buffers['texture'])
        try:
            if texture.shape[2] == 3:
                gl_color_mode = GL.GL_RGB
            elif texture.shape[2] == 4:
                gl_color_mode = GL.GL_RGBA
            else:
                raise NotImplementedError
            GL.glTexImage2D(
                    GL.GL_TEXTURE_2D, 0, gl_color_mode,
                    texture.shape[1], texture.shape[0], 0,
                    gl_color_mode, GL.GL_UNSIGNED_BYTE, texture)

            GL.glTexParameteri(
                    GL.GL_TEXTURE_2D,
                    GL.GL_TEXTURE_MAG_FILTER,
                    GL.GL_LINEAR,
            )
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER,
                    GL.GL_LINEAR_MIPMAP_LINEAR)
            GL.glGenerateMipmap(GL.GL_TEXTURE_2D)

        finally:
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
    
    def remove_texture(self, name):
        if name in self.gl_data['texture_buffers']:
            GL.glDeleteTextures(
                self.gl_data['texture_buffers'][name]['texture'])
            del(self.gl_data['texture_buffers'][name])
        if name in self.loaded_data['textures']:
            del(self.loaded_data['textures'][name])
        del(self.scene_description['textures'][name])
    
    def list_textures(self):
        return list(self.scene_description['textures'].keys())
    
    def clear_textures(self):
        for name in self.list_textures():
            self.remove_texture(name)
    
    def texture_exists(self, name):
        return name in self.scene_description['textures']
    
    def get_texture(self, name):
        """
        Parameters:
        -----------
        name : str
        
        Returns:
        --------
        array :
            The named loaded texture.
        """
        return self.loaded_data['textures'][name]
    
    # cubemap methods ==========================================================
    
    def load_cubemap(
        self,
        name,
        cubemap_asset=None,
        cubemap_path=None,
        cubemap_data=None,
        crop=None,
        mipmaps=None,
    ):
        """
        Loads a cubemap
        """
        
        # if a cubemap asset name was provided, load that
        if cubemap_asset is not None:
            asset_path = self.asset_library['cubemaps'][cubemap_asset]
            cubemap = load_image(asset_path)
            self.scene_description['cubemaps'][name] = {
                'cubemap_asset':cubemap_asset
            }

        # otherwise if a cubemap path was provided, load that
        elif cubemap_path is not None:
            cubemap = load_image(cubemap_path)
            self.scene_description['cubemaps'][name] = {
                'cubemap_path':cubemap_path
            }

        # otherwise if cubemap data was provided, load that
        elif cubemap_data is not None:
            cubemap = cubemap_data
            self.scene_description['cubemaps'][name] = {
                'cubemap_data':cubemap_data
            }
        
        else:
            raise SplendorException(
                    'Must supply a "cubemap_asset", "cubemap_path" or '
                    '"cubemap_data" argument when loading a cubemap')
        
        # crop if necessary
        if crop is not None:
            cubemap = cubemap[crop[0]:crop[2], crop[1]:crop[3]]
        
        # validate and store the cubemap
        self.loaded_data['cubemaps'][name] = cubemap
        
        # if an entry for this cubemap doesn't exist in cubemap_buffers
        # make one
        self.gl_data['cubemap_buffers'].setdefault(name, {})
        
        # delete any old cubemap that exists
        if 'cubemap' in self.gl_data['cubemap_buffers'][name]:
            GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, 0)
            GL.glDeleteTextures(
                [self.gl_data['cubemap_buffers'][name]['cubemap']])
        
        # make the new cubemap
        self.gl_data['cubemap_buffers'][name]['cubemap'] = (
            GL.glGenTextures(1))
        
        # copy the cubemap to the GPU
        cubemap_buffers = self.gl_data['cubemap_buffers'][name]
        GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, cubemap_buffers['cubemap'])
        try:
            if cubemap.shape[2] == 3:
                gl_color_mode = GL.GL_RGB
            elif cubemap.shape[2] == 4:
                gl_color_mode = GL.GL_RGBA
            else:
                raise NotImplementedError
            
            height, strip_width = cubemap.shape[:2]
            assert strip_width == height * 6
            for i in range(6):
                face_image = cubemap[:,i*height:(i+1)*height]
                validate_texture(face_image)
                GL.glTexImage2D(
                    GL.GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                    0,
                    gl_color_mode,
                    face_image.shape[1],
                    face_image.shape[0],
                    0,
                    gl_color_mode,
                    GL.GL_UNSIGNED_BYTE,
                    face_image,
                )
                if mipmaps is not None:
                    for j, mipmap in enumerate(mipmaps[i]):
                        mipmap = numpy.array(mipmap)
                        validate_texture(mipmap)
                        GL.glTexImage2d(
                            GL.GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                            j+1,
                            gl_color_mode,
                            mipmap.shape[1],
                            mipmap.shape[0],
                            0,
                            gl_color_mode,
                            GL.GL_UNSIGNED_BYTE,
                            mipmap,
                        )

            GL.glTexParameteri(
                GL.GL_TEXTURE_CUBE_MAP,
                GL.GL_TEXTURE_MAG_FILTER,
                GL.GL_LINEAR,
            )
            GL.glTexParameteri(
                GL.GL_TEXTURE_CUBE_MAP,
                GL.GL_TEXTURE_MIN_FILTER,
                #GL.GL_LINEAR_MIPMAP_LINEAR,
                GL.GL_LINEAR,
            )
            if mipmaps is None:
                GL.glGenerateMipmap(GL.GL_TEXTURE_CUBE_MAP)
            else:
                GL.glTexParameteri(
                    GL.GL_TEXTURE_CUBE_MAP,
                    GL.GL_TEXTURE_MAX_LEVEL,
                    len(mipmaps[0]),
                )
            GL.glTexParameteri(
                GL.GL_TEXTURE_CUBE_MAP,
                GL.GL_TEXTURE_WRAP_R,
                GL.GL_CLAMP_TO_EDGE,
            )
            GL.glTexParameteri(
                GL.GL_TEXTURE_CUBE_MAP,
                GL.GL_TEXTURE_WRAP_S,
                GL.GL_CLAMP_TO_EDGE,
            )
            GL.glTexParameteri(
                GL.GL_TEXTURE_CUBE_MAP,
                GL.GL_TEXTURE_WRAP_T,
                GL.GL_CLAMP_TO_EDGE,
            )

        finally:
            GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, 0)
    
    def remove_cubemap(self, name):
        if name in self.gl_data['cubemap_buffers']:
            GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, 0)
            GL.glDeleteTextures(
                [self.gl_data['cubemap_buffers'][name]['cubemap']])
            del(self.gl_data['cubemap_buffers'][name])
        if name in self.loaded_data['cubemaps']:
            del(self.loaded_data['cubemaps'][name])
        del(self.scene_description['cubemaps'][name])
    
    def list_cubemaps(self):
        return list(self.scene_description['cubemaps'].keys())
    
    def clear_cubemaps(self):
        for name in self.list_cubemaps():
            self.remove_cubemap(name)
    
    def cubemap_exists(self, name):
        return name in self.scene_description['cubemaps']
    
    def get_cubemap(self, name):
        """
        Parameters:
        -----------
        name : str
        
        Returns:
        --------
        array :
            The named loaded cubemap.
        """
        return self.loaded_data['cubemaps'][name]
    
    
    # material methods =========================================================
    
    def load_material(self,
            name,
            texture_name = None,
            flat_color = None,
            material_properties_texture = None,
            ambient = 1.,
            metal = 0.,
            rough = 0.3,
            base_reflect = 0.04,
            crop = None):
        """
        Load a material.
        
        Loads a material into memory but it is not used until an instance is
        created that references it.
        
        Parameters:
        -----------
        name : str
            Name of the material, must be unique to this scene among
            other materials
        texture : str or array-like, optional
            Either an asset, path or image data
            (must specify either texture or flat_color)
        flat_color : tuple, optional
            A flat color for this material
            (must specify either texture or flat_color)
        ambient : float, default=1.
            The degree to which this material is affected by the ambient color
            in the scene.
        metal : float, default=0.
            The metal parameter turns down the diffuse component of the light
            and uses the albedo as a reflection coefficient.  A value of 1.0
            will result in a shiny surface that is tinted using the surface
            albedo (texture/flat_color).
        rough : float, default=0.
            Roughness causes reflections to blur and specular highlights to
            be larger and fuzzier.
        base_reflect : float, default=0.04
            The ammount of light the surface reflects when the normal is
            facing the camera.  A value of 1. with 0. metal and 0. roughness
            results in a pure mirror.
        crop : 4-tuple, optional
            Bottom, left, top, right crop values for the texture
        """
        
        material_description = {
            'texture_name':texture_name,
            'flat_color':flat_color,
            'material_properties_texture':material_properties_texture,
            'metal':metal,
            'rough':rough,
            'base_reflect':base_reflect,
            'ambient':ambient,
        }
        
        self.scene_description['materials'][name] = material_description

    def remove_material(self, name):
        """
        Deletes a material from the scene.
        
        Parameters:
        -----------
        name : str
        """
        del(self.scene_description['materials'][name])

    def clear_materials(self):
        """
        Deletes all materials.
        """
        for name in list(self.scene_description['materials'].keys()):
            self.remove_material(name)

    def list_materials(self):
        """
        Returns:
        --------
        list :
            All material names in the scene.
        """
        return list(self.scene_description['materials'].keys())

    def material_exists(self, name):
        """
        Parameters:
        -----------
        name : str
        
        Returns:
        --------
        bool
        """
        return name in self.scene_description['materials']

    def get_material(self, material_name):
        """
        Parameters:
        -----------
        name : str
        
        Returns:
        --------
        str :
            The serialized description of the material.
        """
        return self.scene_description['materials'][material_name]

    def get_material_flat_color(self, name):
        """
        Parameters:
        -----------
        name : str
        
        Returns:
        --------
        str or None :
            The flat color associated with a material or None if not present
        """
        return self.scene_description['materials'][name]['flat_color']
    
    def set_material_flat_color(self, name, color):
        """
        """
        self.scene_description['materials'][name]['flat_color'] = color
    
    def get_material_texture(self, name):
        return self.scene_description['materials'][name]['texture_name']
    
    def get_material_properties_texture(self, name):
        material_data = self.scene_description['materials'][name]
        return material_data['material_properties_texture']
    
    # instance methods =========================================================
    
    def add_instance(self,
            name,
            mesh_name,
            material_name,
            transform = numpy.eye(4),
            mask_color = (0,0,0),
            coord_box = ((0,0,0),(0,0,0)),
            hidden = False):
        """
        Add an instance to the scene.
        
        Each instance is a combination of a mesh and a material with an
        additional transform and mask color.  May also have a coordinate box
        for coordinate rendering.
        
        Parameters:
        -----------
        name : str
            Name of the new instance, must be unique to this scene among
            other instances
        mesh_name : str
            The mesh associated with this instance
        material_name : str
            The material associated with this instance
        transform : 4x4 array-like, default=numpy.eye(4)
            The 3D transform of this mesh in the scene
        mask_color : array-like, default=(0,0,0)
            The color to be applied to the mesh when rendering masks
        coord_box : tuple, default=((0,0,0),(0,0,0))
            Bounding box corners used for coordinate rendering
        hidden : bool, default=False
            If True, this instance will not be rendered
        """

        instance_data = {}
        instance_data['mesh_name'] = mesh_name
        instance_data['material_name'] = material_name
        instance_data['transform'] = numpy.array(transform)
        instance_data['mask_color'] = numpy.array(mask_color)
        instance_data['coord_box'] = numpy.array(coord_box)
        instance_data['hidden'] = hidden
        self.scene_description['instances'][name] = instance_data

    def remove_instance(self, name):
        """    
        Deletes an instance from the scene. 
         
        Parameters: 
        ----------- 
        name : str 
        """
        del(self.scene_description['instances'][name])

    def clear_instances(self):
        """
        Deletes all instances.
        """
        self.scene_description['instances'] = {}

    def get_instance_transform(self, name):
        """
        Gets the 3D transform of an instance
        
        Parameters:
        -----------
        name : str
        
        Returns:
        --------
        4x4 array : the instance's transform
        """
        return self.scene_description['instances'][name]['transform']

    def set_instance_transform(self, name, transform):
        """
        Sets the 3D transform of an instance
        
        Parameters:
        -----------
        name : str
        transform : 4x4 array-like
        """
        self.scene_description['instances'][name]['transform'] = (
                numpy.array(transform))

    def set_instance_material(self, name, material_name):
        """
        Sets the material of an instance
        
        Parameters:
        -----------
        name : str
        material_name : str
        """
        self.scene_description['instances'][name]['material_name'] = (
                material_name)
    
    def set_instance_mesh(self, name, mesh_name):
        """
        Sets the mesh of an instance
        
        Parameters:
        -----------
        name : str
        mesh_name : str
        """
        self.scene_description['instances'][name]['mesh_name'] = mesh_name
    
    def hide_instance(self, name):
        """
        Hides an instance so that it will not render in any render modes
        
        Parameters:
        -----------
        name : str
        """
        self.scene_description['instances'][name]['hidden'] = True
    
    def hide_all_instances(self):
        """
        Hides all instances
        """
        for instance in self.list_instances():
            self.hide_instance(instance)
    
    def show_instance(self, name):
        """
        Makes an instance visible in all render modes
        
        Parameters:
        -----------
        name : str
        """
        self.scene_description['instances'][name]['hidden'] = False
    
    def show_all_instances(self):
        """
        Makes all instances visible
        """
        for instance in self.list_instances():
            self.show_instance(instance)
    
    def get_instance_mesh_name(self, name):
        """
        Returns the mesh associated with an instance
        
        Parameters:
        -----------
        name : str
        
        Returns:
        --------
        str : the name of the associated mesh
        """
        return self.scene_description['instances'][name]['mesh_name']

    def get_instance_material_name(self, name):
        """
        Returns the material associated with an instance
        
        Parameters:
        -----------
        name : str
        
        Returns:
        --------
        str : the name of the associated material
        """
        instance_data = self.scene_description['instances'][name]
        return instance_data['material_name']

    def get_instance_center_bbox(self, instances=None):
        """
        Returns a bounding box of a list of instances
        
        Parameters:
        -----------
        instances : list-like, optional
            A list of instances to compute the bounding box for.  If not
            specified, this uses all instances in the scene.
        
        Returns:
        tuple : ((min_x, min_y, min_z), (max_x, max_y, max_z))
        """
        if instances is None:
            instances = self.scene_description['instances'].keys()
        if len(instances) > 1:
            centers = numpy.stack([
                    self.get_instance_transform(instance)[:3,3]
                    for instance in instances])
        else:
            centers = numpy.array([[0,0,0],[1,1,1]])
        bbox_min = numpy.min(centers, axis=0)
        bbox_max = numpy.max(centers, axis=0)
        return bbox_min, bbox_max

    def set_instance_masks_to_instance_indices(self, instance_indices):
        """
        Use the masks library to assign a unique mask color to a set
        of instances.
        
        Parameters:
        -----------
        instances_indices : dict
            A ditionary mapping instance names to integers.  Each integer will
            be assigned a unique color using the masks module.
        """
        indices = numpy.array(list(instance_indices.values()))
        if indices.shape[0]:
            colors = masks.color_index_to_float(indices)
            for i, color in zip(instance_indices.keys(), colors):
                instance_data = self.scene_description['instances'][i]
                instance_data['mask_color'] = color

    def set_instance_masks_to_mesh_indices(self, mesh_indices, instances=None):
        """
        Use the masks library to assign a mask color to a set of instances
        based on the mesh associated with each instance.
        
        Parameters:
        -----------
        mesh_indices : dict
            A dictionary mapping mesh names to integers.  Each integer will
            be assigned a uniqe color using the masks module.
        instances : list-like, optional
            A list of instances to assign mask colors.  If not specified, all
            instances in the scene will be assigned if the associated mesh name
            exists as a key in mesh_indices.
        """
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
        """
        Returns:
        --------
        list :
            All instance names in the scene.
        """
        return list(self.scene_description['instances'].keys())

    def instance_exists(self, name):
        """
        Parameters:
        -----------
        name : str
        
        Returns:
        --------
        bool
        """
        return name in self.scene_description['instances']

    def instance_hidden(self, name):
        """
        Parameters:
        -----------
        name : str
        
        Returns:
        --------
        bool
        """
        return self.scene_description['instances'][name]['hidden']

    # depthmap_instance methods ================================================
    
    def add_depthmap_instance(self,
            name,
            depthmap_name,
            texture_name,
            transform = numpy.eye(4),
            point_size = 1):
        """
        Add a depthmap instance to the scene.
        
        Each depthmap instance is a combination of a depthmap and a material
        with an additional transform and point_size attribute for display
        purposes.
        
        TODO: Material should be a texture if we break textures out as separate
        assets.
        
        Parameters:
        -----------
        name : str
            Name of the new depthmap instance, must be unique to this scene
            among other depthmap instances
        depthmap_name : str
            The depthmap associated with this instance
        material_name : str
            The material associated with this instance
        transform : 4x4 array-like, default=numpy.eye(4)
            The 3D transform of this mesh in the scene
        point_size : int, default=1
            The 2D size of the rendered points
        """
        depthmap_instance_data = {}
        depthmap_instance_data['depthmap_name'] = depthmap_name
        depthmap_instance_data['texture_name'] = texture_name
        depthmap_instance_data['transform'] = numpy.array(transform)
        depthmap_instance_data['point_size'] = point_size
        self.scene_description['depthmap_instances'][name] = (
                depthmap_instance_data)

    def remove_depthmap_instance(self, depthmap_instance_name):
        """    
        Deletes a depthmap instance from the scene. 
         
        Parameters: 
        ----------- 
        name : str 
        """
        del(self.scene_description['depthmap_instances'][
                depthmap_instance_name])

    def clear_depthmap_instances(self):
        """
        Deletes all depthmap instances.
        """
        self.scene_description['depthmap_instances'] = {}

    def set_depthmap_instance_material(self, name, material_name):
        """
        Sets the material of a depthmap instance
        
        Parameters:
        -----------
        name : str
        material_name : str
        """
        self.scene_description['depthmap_instances'][name][
                'material_name'] = material_name

    def set_depthmap_instance_transform(self, name, transform):
        """
        Sets the 3D transform of a depthmap instance
        
        Parameters:
        -----------
        name : str
        transform : 4x4 array-like
        """
        self.scene_description['depthmap_instances'][name][
                'transform'] = numpy.array(transform)

    def get_depthmap_instance_transform(self, name):
        """
        Gets the 3D transform of a depthmap instance
        
        Parameters:
        -----------
        name : str
        
        Returns:
        --------
        4x4 array : the instance's transform
        """
        return self.scene_description['depthmap_instances'][name]['transform']

    def depthmap_instance_exists(self, name):
        """
        Parameters:
        -----------
        name : str
        
        Returns:
        --------
        bool
        """
        return name in self.scene_description['depthmap_instances']

    # point_light methods ======================================================
    
    def add_point_light(self, name, pose, color):
        """
        Add a point light to the scene.

        Parameters:
        -----------
        name : str
            Name of the new point light, must be unique among point lights.
        pose : array-like, shape (4, 4)
            World-space pose of the light.  The position is taken from
            column 3 (pose[:3, 3]).
        color : array-like
            3-value RGB color of the point light.
        """
        self.scene_description['point_lights'][name] = {
                'pose'  : numpy.array(pose),
                'color' : numpy.array(color)}
    
    def remove_point_light(self, name):
        """    
        Deletes a point light from the scene. 
         
        Parameters: 
        ----------- 
        name : str 
        """
        del(self.scene_description['point_lights'][name])
    
    def clear_point_lights(self):
        """
        Deletes all point lights.
        """
        self.scene_description['point_lights'] = {}
    
    # direction_light methods ==================================================
    
    def add_direction_light(self,
            name,
            pose,
            color,
            shadow_map=None,
            shadow_projection=None,
            shadow_pcf_radius=1):
        """
        Add a directional light to the scene.

        Parameters:
        -----------
        name : str
            Name of the new direction light, must be unique among direction
            lights.
        pose : array-like, shape (4, 4)
            World-space pose of the light.  The light rays travel in the
            direction of the pose's -Z axis (pose[:3, 2]), consistent with
            the convention that a camera looks down its -Z axis.
            Use camera.direction_light_pose(direction, position) to construct
            this from a ray direction vector.
        color : array-like
            3-value RGB color of the light.
        shadow_map : str, optional
            Name of a depth-only sensor to use as this light's shadow map.
        shadow_projection : array-like, shape (4, 4), optional
            Projection matrix for the shadow camera.  Use
            camera.orthographic_matrix() to build one.
        """
        self.scene_description['direction_lights'][name] = {
                'pose'              : numpy.array(pose),
                'color'             : numpy.array(color),
                'shadow_map'        : shadow_map,
                'shadow_projection' : (numpy.array(shadow_projection)
                                       if shadow_projection is not None
                                       else None),
                'shadow_pcf_radius' : int(shadow_pcf_radius),
        }

    def remove_direction_light(self, name):
        """    
        Deletes a direction light from the scene. 
         
        Parameters: 
        ----------- 
        name : str 
        """
        del(self.scene_description['direction_lights'][name])

    def clear_direction_lights(self):
        """
        Deletes all direction lights.
        """
        self.scene_description['direction_lights'] = {}

    # coord frame methods ======================================================

    def _init_coord_frame_buffers(self):
        """
        Allocate VAO/VBO for coord frame line rendering.
        Each coord frame draws 6 lines (±X, ±Y, ±Z) = 12 vertices.
        Each vertex: 3 floats position + 3 floats color = 6 floats.
        The VBO is allocated for 1 frame and updated per-draw via glBufferData.
        """
        vao = GL.glGenVertexArrays(1)
        vbo = GL.glGenBuffers(1)
        GL.glBindVertexArray(vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
        # position: location 0
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(
            0, 3, GL.GL_FLOAT, GL.GL_FALSE,
            6 * 4, ctypes.c_void_p(0))
        # color: location 1
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(
            1, 3, GL.GL_FLOAT, GL.GL_FALSE,
            6 * 4, ctypes.c_void_p(3 * 4))
        GL.glBindVertexArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        self.gl_data['coord_frame_buffers'] = {'vao': vao, 'vbo': vbo}

    # Shared local-space coord frame vertex data (12 vertices × 6 floats).
    # Layout per vertex: px py pz cr cg cb (interleaved).
    # Transform applied via MVP uniform.  axis_length applied as a scale
    # factor inside add_coord_frame by baking a scale into the stored transform.
    # Axes: +X red, +Y green, +Z blue, -X magenta, -Y yellow, -Z cyan.
    _COORD_FRAME_VERTS = numpy.array([
        # px  py  pz    cr cg cb
        0., 0., 0.,   1, 0, 0,   # +X line start
        1., 0., 0.,   1, 0, 0,   # +X line end
        0., 0., 0.,   0, 1, 0,   # +Y line start
        0., 1., 0.,   0, 1, 0,   # +Y line end
        0., 0., 0.,   0, 0, 1,   # +Z line start
        0., 0., 1.,   0, 0, 1,   # +Z line end
        0., 0., 0.,   1, 0, 1,   # -X line start
       -1., 0., 0.,   1, 0, 1,   # -X line end
        0., 0., 0.,   1, 1, 0,   # -Y line start
        0.,-1., 0.,   1, 1, 0,   # -Y line end
        0., 0., 0.,   0, 1, 1,   # -Z line start
        0., 0.,-1.,   0, 1, 1,   # -Z line end
    ], dtype=numpy.float32)

    def add_coord_frame(self, name, transform, axis_length=0.1):
        """
        Add a named coordinate frame to the scene for visualization.

        Parameters
        ----------
        name : str
            Unique name for this coord frame.
        transform : array-like (4x4)
            World-space pose of the frame.
        axis_length : float, default=0.1
            Length of each axis line in world units.
        """
        self.scene_description['coord_frames'][name] = {
            'transform': numpy.array(transform, dtype=numpy.float32),
            'axis_length': float(axis_length),
        }

    def remove_coord_frame(self, name):
        """Remove a named coord frame from the scene."""
        del self.scene_description['coord_frames'][name]

    def clear_coord_frames(self):
        """Remove all coord frames from the scene."""
        self.scene_description['coord_frames'] = {}

    def set_coord_frame_transform(self, name, transform, axis_length=None):
        """Update the transform (and optionally axis_length) of a named coord frame."""
        self.scene_description['coord_frames'][name]['transform'] = (
            numpy.array(transform, dtype=numpy.float32))
        if axis_length is not None:
            self.scene_description['coord_frames'][name]['axis_length'] = (
                float(axis_length))

    def _render_scene_coord_frames(self, camera_data, flip_y=True):
        """Render all scene coord frames. Called from color_render."""
        frames = self.scene_description['coord_frames']
        if not frames:
            return

        view_matrix = camera_data['view_matrix'].astype(numpy.float32)
        projection_matrix = camera_data['projection'].astype(numpy.float32)
        if flip_y:
            flip = numpy.array([
                [1, 0, 0, 0],
                [0,-1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]], dtype=numpy.float32)
            projection_matrix = flip @ projection_matrix

        vp = projection_matrix @ view_matrix

        buffers = self.gl_data['coord_frame_buffers']
        self.shader_library.use_program('lines_shader')
        locations = self.shader_library.get_shader_locations('lines_shader')
        mvp_loc = locations['mvp_matrix']

        GL.glBindVertexArray(buffers['vao'])
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, buffers['vbo'])

        # Upload the shared unit-axis vertex data once
        verts = SplendorRender._COORD_FRAME_VERTS
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER,
            verts.nbytes,
            verts,
            GL.GL_STREAM_DRAW,
        )

        for frame_data in frames.values():
            transform = frame_data['transform']
            axis_length = frame_data['axis_length']
            scaled = transform.copy()
            scaled[:3, :3] *= axis_length
            mvp = (vp @ scaled).astype(numpy.float32)
            GL.glUniformMatrix4fv(mvp_loc, 1, GL.GL_TRUE, mvp)
            GL.glDrawArrays(GL.GL_LINES, 0, 12)

        GL.glBindVertexArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glUseProgram(0)

    # frustum methods ==========================================================

    # NDC corners: near (z=-1) and far (z=1) planes, in order
    # nbl, nbr, ntr, ntl, fbl, fbr, ftr, ftl
    _FRUSTUM_NDC = numpy.array([
        [-1, -1, -1, 1],  # near bottom-left
        [ 1, -1, -1, 1],  # near bottom-right
        [ 1,  1, -1, 1],  # near top-right
        [-1,  1, -1, 1],  # near top-left
        [-1, -1,  1, 1],  # far bottom-left
        [ 1, -1,  1, 1],  # far bottom-right
        [ 1,  1,  1, 1],  # far top-right
        [-1,  1,  1, 1],  # far top-left
    ], dtype=numpy.float32)

    # 12 edges as pairs of corner indices
    _FRUSTUM_EDGES = [
        (0,1),(1,2),(2,3),(3,0),  # near plane
        (4,5),(5,6),(6,7),(7,4),  # far plane
        (0,4),(1,5),(2,6),(3,7),  # connecting
    ]

    @staticmethod
    def _compute_frustum_verts(transform, projection, color):
        inv_proj = numpy.linalg.inv(projection)
        # unproject NDC corners to view space, then to world space
        corners_view = (inv_proj @ SplendorRender._FRUSTUM_NDC.T).T
        corners_view /= corners_view[:, 3:4]  # perspective divide
        corners_world = (transform @ corners_view.T).T
        corners_world = corners_world[:, :3]

        c = numpy.array(color, dtype=numpy.float32)
        verts = []
        for i, j in SplendorRender._FRUSTUM_EDGES:
            verts.append(numpy.concatenate([corners_world[i], c]))
            verts.append(numpy.concatenate([corners_world[j], c]))
        return numpy.array(verts, dtype=numpy.float32).ravel()

    def add_frustum(self, name, transform, projection, color=(1, 1, 1)):
        """
        Add a named camera frustum wireframe to the scene for visualization.

        Parameters
        ----------
        name : str
            Unique name for this frustum.
        transform : array-like (4x4)
            World-space camera pose (camera-to-world).
        projection : array-like (4x4)
            Camera projection matrix.
        color : 3-tuple, default white
            RGB color for all frustum edges.
        """
        self.scene_description['frustums'][name] = {
            'transform': numpy.array(transform, dtype=numpy.float32),
            'projection': numpy.array(projection, dtype=numpy.float32),
            'color': list(color),
        }

    def remove_frustum(self, name):
        """Remove a named frustum from the scene."""
        del self.scene_description['frustums'][name]

    def clear_frustums(self):
        """Remove all frustums from the scene."""
        self.scene_description['frustums'] = {}

    def set_frustum(self, name, transform, projection, color=(1, 1, 1)):
        """Update the transform, projection, and/or color of a named frustum."""
        self.scene_description['frustums'][name] = {
            'transform': numpy.array(transform, dtype=numpy.float32),
            'projection': numpy.array(projection, dtype=numpy.float32),
            'color': list(color),
        }

    def _render_scene_frustums(self, camera_data, flip_y=True):
        """Render all scene frustum wireframes. Called from color_render."""
        frustums = self.scene_description['frustums']
        if not frustums:
            return

        view_matrix = camera_data['view_matrix'].astype(numpy.float32)
        projection_matrix = camera_data['projection'].astype(numpy.float32)
        if flip_y:
            flip = numpy.array([
                [1, 0, 0, 0],
                [0,-1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]], dtype=numpy.float32)
            projection_matrix = flip @ projection_matrix

        vp = (projection_matrix @ view_matrix).astype(numpy.float32)

        buffers = self.gl_data['coord_frame_buffers']
        self.shader_library.use_program('lines_shader')
        locations = self.shader_library.get_shader_locations('lines_shader')
        mvp_loc = locations['mvp_matrix']
        GL.glUniformMatrix4fv(mvp_loc, 1, GL.GL_TRUE, vp)

        GL.glBindVertexArray(buffers['vao'])
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, buffers['vbo'])

        for frustum_data in frustums.values():
            verts = self._compute_frustum_verts(
                frustum_data['transform'],
                frustum_data['projection'],
                frustum_data['color'],
            )
            GL.glBufferData(
                GL.GL_ARRAY_BUFFER, verts.nbytes, verts, GL.GL_STREAM_DRAW)
            GL.glDrawArrays(GL.GL_LINES, 0, 24)

        GL.glBindVertexArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glUseProgram(0)

    # line_set scene object methods ============================================

    def add_line_set(self, name, starts, ends, colors):
        """Add a named set of line segments to the scene."""
        self.scene_description['line_sets'][name] = {
            'starts': numpy.array(starts, dtype=numpy.float32).tolist(),
            'ends': numpy.array(ends, dtype=numpy.float32).tolist(),
            'colors': numpy.array(colors, dtype=numpy.float32).tolist(),
        }

    def remove_line_set(self, name):
        del self.scene_description['line_sets'][name]

    def clear_line_sets(self):
        self.scene_description['line_sets'] = {}

    # point_cloud scene object methods =========================================

    def add_point_cloud(self, name, points, colors, point_size=1):
        """Add a named point cloud to the scene."""
        self.scene_description['point_clouds'][name] = {
            'points': numpy.array(points, dtype=numpy.float32).tolist(),
            'colors': numpy.array(colors, dtype=numpy.float32).tolist(),
            'point_size': point_size,
        }

    def remove_point_cloud(self, name):
        del self.scene_description['point_clouds'][name]

    def clear_point_clouds(self):
        self.scene_description['point_clouds'] = {}

    # box scene object methods =================================================

    def add_box(self, name, transform, color):
        """Add a named wireframe box (transform maps the unit cube)."""
        self.scene_description['boxes'][name] = {
            'transform': numpy.array(transform, dtype=numpy.float32).tolist(),
            'color': list(color),
        }

    def remove_box(self, name):
        del self.scene_description['boxes'][name]

    def clear_boxes(self):
        self.scene_description['boxes'] = {}

    # arrow scene object methods ===============================================

    def add_arrow(self, name, start, end, color, wedge_size=0.1):
        """Add a named arrow to the scene."""
        self.scene_description['arrows'][name] = {
            'start': list(start),
            'end': list(end),
            'color': list(color),
            'wedge_size': wedge_size,
        }

    def remove_arrow(self, name):
        del self.scene_description['arrows'][name]

    def clear_arrows(self):
        self.scene_description['arrows'] = {}

    # scene line overlay rendering =============================================

    def _render_scene_overlays(self, camera_data, flip_y=True):
        """Render all scene line overlays (called from color_render)."""
        view = camera_data['view_matrix'].astype(numpy.float32)
        proj = camera_data['projection'].astype(numpy.float32)
        if flip_y:
            proj = numpy.array(
                [[1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]],
                dtype=numpy.float32) @ proj
        vp = proj @ view

        # Collect all line vertices
        line_verts = []

        for data in self.scene_description['line_sets'].values():
            line_verts.append(self._build_line_set_verts(
                data['starts'], data['ends'], data['colors']))

        for data in self.scene_description['boxes'].values():
            line_verts.append(self._build_box_verts(
                data['transform'], data['color']))

        for data in self.scene_description['arrows'].values():
            v = self._build_arrow_verts(
                data['start'], data['end'], data['color'],
                data.get('wedge_size', 0.1))
            if len(v) > 0:
                line_verts.append(v)

        if line_verts:
            all_verts = numpy.concatenate(line_verts)
            self._draw_primitives(vp, all_verts)

        # Points need a separate draw call (GL_POINTS)
        for data in self.scene_description['point_clouds'].values():
            GL.glPointSize(data.get('point_size', 1))
            verts = self._build_point_cloud_verts(
                data['points'], data['colors'])
            self._draw_primitives(vp, verts, draw_mode=GL.GL_POINTS)

    # render methods ===========================================================
    
    def clear_frame(self):
        """
        Clears the frame.
        """
        GL.glClearColor(*self.scene_description['background_color'])
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

    def finish_frame(self):
        """
        Force opengl to finish rendering before continuing.
        """
        GL.glFinish()

    # color_render methods -----------------------------------------------------

    def _distortion_expanded_projection(self, projection, k1, k2, width, height):
        """
        Return a modified projection matrix with FOV expanded so that the
        undistorted intermediate render covers all pixels needed by the warp
        pass.  Also returns fov_scale = f_intermediate / f_output.
        """
        aspect = width / height
        # Corner of the output image in aspect-corrected camera-space coords
        # (using the output focal lengths: p = ndc / f, corner ndc = (1, 1))
        fx = projection[0, 0]
        fy = projection[1, 1]
        p_corner = numpy.array([1.0 / fx, 1.0 / fy])
        r2 = numpy.dot(p_corner, p_corner)
        r4 = r2 ** 2
        d_corner = 1.0 + k1 * r2 + k2 * r4
        d_corner = max(d_corner, 0.1)
        # For barrel distortion (d_corner < 1): the output corner maps to an
        # undistorted position beyond the natural FOV, so we must widen the
        # intermediate render.  fov_scale = d_corner < 1 widens it just enough.
        #
        # For pincushion distortion (d_corner > 1): the output corner maps to
        # an undistorted position inside the natural FOV, so no narrowing is
        # needed (narrowing would clip scene content near the edges).
        # Keep fov_scale = 1.0.
        fov_scale = float(min(1.0, d_corner))
        expanded = projection.copy()
        expanded[0, 0] = fx * fov_scale
        expanded[1, 1] = fy * fov_scale
        return expanded, fov_scale

    def _radial_warp_pass(self, sensor, camera_data):
        """
        Run the fullscreen warp pass: sample the intermediate texture and write
        the barrel/pincushion-distorted result to the sensor's main FBO.
        """
        sensor_data = self.scene_description['sensors'][sensor]
        sensor_buffers = self.gl_data['sensor_buffers'][sensor]
        width = sensor_data['width']
        height = sensor_data['height']

        projection = camera_data['projection']
        k1 = camera_data['radial_k1']
        k2 = camera_data['radial_k2']
        _, fov_scale = self._distortion_expanded_projection(
            projection, k1, k2, width, height)

        # Resolve MSAA → texture before sampling
        intermediate_fbo = sensor_buffers['intermediate_fbo']
        if intermediate_fbo.anti_alias:
            intermediate_fbo.resolve()

        # Bind the main (output) FBO
        sensor_buffers['main_fbo'].enable()
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        self.shader_library.use_program('radial_warp_shader')
        try:
            locs = self.shader_library.get_shader_locations('radial_warp_shader')

            # Bind the intermediate color texture at unit 5
            GL.glActiveTexture(GL.GL_TEXTURE5)
            GL.glBindTexture(
                GL.GL_TEXTURE_2D,
                intermediate_fbo.texture)
            GL.glUniform1i(locs['intermediate_sampler'], 5)

            # Bind the intermediate depth texture at unit 6
            GL.glActiveTexture(GL.GL_TEXTURE6)
            GL.glBindTexture(
                GL.GL_TEXTURE_2D,
                intermediate_fbo.depth_texture)
            GL.glUniform1i(locs['intermediate_depth_sampler'], 6)

            GL.glUniform1f(locs['radial_k1'], k1)
            GL.glUniform1f(locs['radial_k2'], k2)
            GL.glUniform1f(locs['fx'], float(projection[0, 0]))
            GL.glUniform1f(locs['fy'], float(projection[1, 1]))
            GL.glUniform1f(locs['fov_scale'], fov_scale)

            # Draw fullscreen quad (4 vertices, no VBO).
            # Use GL_ALWAYS so depth test passes (enabling depth writes)
            # without discarding any fragments.
            GL.glBindVertexArray(self.empty_vao)
            GL.glDepthFunc(GL.GL_ALWAYS)
            GL.glDrawArrays(GL.GL_TRIANGLE_FAN, 0, 4)
            GL.glDepthFunc(GL.GL_LESS)
            GL.glBindVertexArray(0)
        finally:
            GL.glUseProgram(0)

    def _render_single_shadow_map(self, shadow_map, shadow_projection, pose,
                                   instances):
        """Render a single depth pass into shadow_map from the given pose."""
        if shadow_map is None or shadow_projection is None or pose is None:
            return
        if not self.sensor_exists(shadow_map):
            return
        shadow_view = numpy.linalg.inv(numpy.array(pose))
        fbo = self.gl_data['sensor_buffers'][shadow_map]['main_fbo']
        fbo.enable()
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
        # Back-face only: natural depth offset prevents acne without peter-panning
        GL.glEnable(GL.GL_CULL_FACE)
        GL.glCullFace(GL.GL_FRONT)
        self.shader_library.use_program('depthmap_shadow_shader')
        try:
            locs = self.shader_library.get_shader_locations('depthmap_shadow_shader')
            GL.glUniformMatrix4fv(
                locs['view_matrix'], 1, GL.GL_TRUE,
                shadow_view.astype(numpy.float32))
            GL.glUniformMatrix4fv(
                locs['projection_matrix'], 1, GL.GL_TRUE,
                numpy.array(shadow_projection, dtype=numpy.float32))
            render_instances = (instances if instances is not None
                                else self.scene_description['instances'].keys())
            for instance_name in render_instances:
                if self.instance_hidden(instance_name):
                    continue
                instance_data = self.scene_description['instances'][instance_name]
                mesh_name = instance_data['mesh_name']
                mesh_buffers = self.gl_data['mesh_buffers'][mesh_name]
                GL.glBindVertexArray(mesh_buffers['vao'])
                GL.glUniformMatrix4fv(
                    locs['model_pose'], 1, GL.GL_TRUE,
                    numpy.array(instance_data['transform'], dtype=numpy.float32))
                mesh = self.loaded_data['meshes'][mesh_name]
                GL.glDrawElements(
                    GL.GL_TRIANGLES, len(mesh['faces']) * 3,
                    GL.GL_UNSIGNED_INT, None)
                GL.glBindVertexArray(0)
        finally:
            GL.glUseProgram(0)
            GL.glCullFace(GL.GL_BACK)
            GL.glDisable(GL.GL_CULL_FACE)

    def render_shadow_maps(self, instances=None):
        """
        Render depth-only passes for all shadow-casting lights.

        Called automatically by color_render() when update_shadow_maps=True.
        Call manually for static scenes where geometry doesn't change each frame.
        """
        for light_data in self.scene_description['direction_lights'].values():
            self._render_single_shadow_map(
                light_data.get('shadow_map'),
                light_data.get('shadow_projection'),
                light_data.get('pose'),
                instances,
            )
        active_il = self.get_active_image_light()
        if active_il:
            il_data = self.get_image_light(active_il)
            self._render_single_shadow_map(
                il_data.get('shadow_map'),
                il_data.get('shadow_projection'),
                il_data.get('shadow_pose'),
                instances,
            )

    def color_render(self,
        camera,
        instances=None,
        depthmap_instances=None,
        sensor=None,
        flip_y=True,
        clear=True,
        finish=True,
        ignore_hidden=False,
        update_shadow_maps=True,
    ):
        """
        Renders instances and depthmap instances using the color program.

        Parameters:
        -----------
        camera : str
            Name of the camera to render from.
        instances : list, optional
            A list of instances to render.  If not specified, all instances will
            be rendered.
        depthmap_instances : list, optional
            A list of depthmap instances to render.  If not specified, all
            depthmap instances will be rendered.
        sensor : str, optional
            Name of the sensor to render into.  If not specified, renders to
            the currently-bound framebuffer (e.g. the screen in interactive
            mode).
        flip_y : bool, default=True
            Whether or not to flip the image in Y when rendering.  This is to
            correct for the difference between rendering to windows and
            framebuffers.
        clear : bool, default=True
            Whether or not to clear the frame before rendering.
        finish : bool, default=True
            Whether or not to finish the frame using glFinish
        """

        camera_data = self.scene_description['cameras'][camera]

        # Determine if we need the two-stage radial distortion pass
        use_distortion = (
            sensor is not None
            and self.scene_description['sensors'][sensor][
                'enable_radial_distortion']
            and (camera_data['radial_k1'] != 0
                 or camera_data['radial_k2'] != 0)
        )

        if update_shadow_maps:
            self.render_shadow_maps(instances=instances)

        if use_distortion:
            # Stage 1: render to intermediate FBO with expanded FOV
            sensor_buffers = self.gl_data['sensor_buffers'][sensor]
            sensor_data = self.scene_description['sensors'][sensor]
            intermediate_projection, _ = self._distortion_expanded_projection(
                camera_data['projection'],
                camera_data['radial_k1'],
                camera_data['radial_k2'],
                sensor_data['width'],
                sensor_data['height'],
            )
            render_camera_data = dict(camera_data)
            render_camera_data['projection'] = intermediate_projection
            sensor_buffers['intermediate_fbo'].enable()
        else:
            render_camera_data = camera_data
            if sensor is not None:
                self._bind_sensor(sensor)

        # clear
        if clear:
            self.clear_frame()

        # render the background
        image_light_name = self.scene_description['active_image_light']
        if image_light_name is not None:
            image_light_data = self.get_image_light(image_light_name)
            if image_light_data['render_background']:
                self.render_background(
                    image_light_name,
                    render_camera_data['view_matrix'],
                    render_camera_data['projection'],
                    flip_y=flip_y,
                )

        # depthmap_instances
        if depthmap_instances is None:
            depthmap_instances = self.scene_description['depthmap_instances']

        self.shader_library.use_program('textured_depthmap_shader')
        try:
            location_data = self.shader_library.get_shader_locations(
                    'textured_depthmap_shader')

            # set the camera's view_matrix
            view_matrix = render_camera_data['view_matrix']
            GL.glUniformMatrix4fv(
                    location_data['view_matrix'],
                    1, GL.GL_TRUE,
                    view_matrix.astype(numpy.float32))

            # set the camera's projection matrix
            projection_matrix = render_camera_data['projection']
            if flip_y:
                projection_matrix = numpy.dot(
                        projection_matrix,
                        numpy.array([
                            [ 1, 0, 0, 0],
                            [ 0,-1, 0, 0],
                            [ 0, 0, 1, 0],
                            [ 0, 0, 0, 1]]))
            GL.glUniformMatrix4fv(
                    location_data['projection_matrix'],
                    1, GL.GL_TRUE,
                    projection_matrix.astype(numpy.float32))

            # radial distortion for point clouds (depthmap shader only)
            GL.glUniform1f(
                location_data['radial_k1'], camera_data['radial_k1'])
            GL.glUniform1f(
                location_data['radial_k2'], camera_data['radial_k2'])

            # render the depthmap instances
            for depthmap_instance_name in depthmap_instances:
                self.color_render_depthmap_instance(depthmap_instance_name)

        finally:
            GL.glUseProgram(0)

        # figure out which programs we need (color/vertex_color)
        if instances is None:
            instances = self.scene_description['instances']
        
        textured_material_properties_shader_instances = {}
        textured_shader_instances = {}
        vertex_color_shader_instances = {}
        flat_color_shader_instances = {}
        for instance in instances:
            if not ignore_hidden and self.instance_hidden(instance):
                continue
            instance_material = self.get_instance_material_name(instance)
            instance_mesh = self.get_instance_mesh_name(instance)
            mesh_color_mode = self.get_mesh_color_mode(instance_mesh)
            if mesh_color_mode == 'textured':
                if (self.get_material_properties_texture(instance_material)
                    is None):
                    shader_instances = textured_shader_instances
                else:
                    shader_instances = (
                        textured_material_properties_shader_instances)
            elif mesh_color_mode == 'vertex_color':
                shader_instances = vertex_color_shader_instances
            elif mesh_color_mode == 'flat_color':
                shader_instances = flat_color_shader_instances
            
            try:
                shader_instances[instance_material][instance_mesh].append(
                        instance)
            except KeyError:
                try:
                    shader_instances[instance_material][instance_mesh] = [
                            instance]
                except KeyError:
                    shader_instances[instance_material] = {
                            instance_mesh:[instance]}
        
        for shader_name, shader_instances in (
            ('textured_material_properties_shader',
                 textured_material_properties_shader_instances),
            ('textured_shader', textured_shader_instances),
            ('vertex_color_shader', vertex_color_shader_instances),
            ('flat_color_shader', flat_color_shader_instances),
        ):

            if len(shader_instances) == 0:
                continue

            # turn on the shader
            self.shader_library.use_program(shader_name)
            
            try:
                location_data = self.shader_library.get_shader_locations(
                    shader_name)
                
                # set the cubemap samplers
                
                if self.get_active_image_light() is not None:
                    if 'irradiance_sh' in location_data:
                        sh_array = self.loaded_data['irradiance_sh'][
                            image_light_name]
                        GL.glUniform3fv(
                            location_data['irradiance_sh'],
                            9,
                            sh_array)
                    if 'reflect_sampler' in location_data:
                        reflect_cubemap = image_light_data['reflect_cubemap']
                        GL.glActiveTexture(GL.GL_TEXTURE3)
                        reflect_data = (
                            self.gl_data['cubemap_buffers'][reflect_cubemap])
                        GL.glBindTexture(
                            GL.GL_TEXTURE_CUBE_MAP,
                            reflect_data['cubemap'],
                        )
                        GL.glUniform1i(location_data['reflect_sampler'], 3)
                
                # set the camera's view matrix
                view_matrix = render_camera_data['view_matrix']
                GL.glUniformMatrix4fv(
                        location_data['view_matrix'],
                        1, GL.GL_TRUE,
                        view_matrix.astype(numpy.float32))

                # set the camera's projection matrix
                projection_matrix = render_camera_data['projection']
                if flip_y:
                    projection_matrix = numpy.dot(
                            projection_matrix,
                            numpy.array([
                                [ 1, 0, 0, 0],
                                [ 0,-1, 0, 0],
                                [ 0, 0, 1, 0],
                                [ 0, 0, 0, 1]]))
                GL.glUniformMatrix4fv(
                        location_data['projection_matrix'],
                        1, GL.GL_TRUE,
                        projection_matrix.astype(numpy.float32))

                
                # set the ambient light's color
                ambient_color = self.scene_description['ambient_color']
                GL.glUniform3fv(
                        location_data['ambient_color'], 1,
                        ambient_color.astype(numpy.float32))
                
                # set the point light data
                GL.glUniform1i(
                        location_data['num_point_lights'],
                        len(self.scene_description['point_lights']))
                point_light_data = numpy.zeros((max_num_lights*2,3))
                for i, light_name in enumerate(
                        self.scene_description['point_lights']):
                    light_data = self.scene_description[
                            'point_lights'][light_name]
                    pose = numpy.array(light_data['pose'])
                    point_light_data[i*2] = light_data['color']
                    point_light_data[i*2+1] = pose[:3, 3]
                GL.glUniform3fv(
                        location_data['point_light_data'], max_num_lights*2,
                        point_light_data.astype(numpy.float32))
                
                # set the direction light data + assign shadow slots
                GL.glUniform1i(
                        location_data['num_direction_lights'],
                        len(self.scene_description['direction_lights']))
                direction_light_data = numpy.zeros((max_num_lights*2,3))
                dir_shadow_slots = numpy.full(max_num_lights, -1,
                                              dtype=numpy.int32)
                shadow_slots = []  # (sensor_name, view_mat, proj_mat, pcf_r)
                for i, light_name in enumerate(
                        self.scene_description['direction_lights']):
                    light_data = self.scene_description[
                            'direction_lights'][light_name]
                    pose = numpy.array(light_data['pose'])
                    direction_light_data[i*2] = light_data['color']
                    direction_light_data[i*2+1] = -pose[:3, 2]
                    smap = light_data.get('shadow_map')
                    sproj = light_data.get('shadow_projection')
                    if (smap is not None and sproj is not None
                            and self.sensor_exists(smap)
                            and len(shadow_slots) < MAX_SHADOW_CASTERS):
                        shadow_slots.append((
                            smap,
                            numpy.linalg.inv(pose),
                            numpy.array(sproj),
                            int(light_data.get('shadow_pcf_radius', 1)),
                        ))
                        dir_shadow_slots[i] = len(shadow_slots) - 1
                GL.glUniform3fv(
                        location_data['direction_light_data'], max_num_lights*2,
                        direction_light_data.astype(numpy.float32))

                # Image light shadow slot
                ibl_shadow_slot = -1
                ibl_shadow_dir = numpy.zeros(3, dtype=numpy.float32)
                if (image_light_name is not None
                        and len(shadow_slots) < MAX_SHADOW_CASTERS):
                    il_data = self.get_image_light(image_light_name)
                    smap = il_data.get('shadow_map')
                    sproj = il_data.get('shadow_projection')
                    spose = il_data.get('shadow_pose')
                    if (smap is not None and sproj is not None
                            and spose is not None
                            and self.sensor_exists(smap)):
                        spose = numpy.array(spose)
                        shadow_slots.append((
                            smap,
                            numpy.linalg.inv(spose),
                            numpy.array(sproj),
                            int(il_data.get('shadow_pcf_radius', 1)),
                        ))
                        ibl_shadow_slot = len(shadow_slots) - 1
                        ibl_shadow_dir = spose[:3, 2].astype(numpy.float32)

                # Upload shadow uniforms
                shadow_view_mats = numpy.zeros(
                    (MAX_SHADOW_CASTERS, 4, 4), dtype=numpy.float32)
                shadow_proj_mats = numpy.zeros(
                    (MAX_SHADOW_CASTERS, 4, 4), dtype=numpy.float32)
                shadow_pcf_radii = numpy.ones(
                    MAX_SHADOW_CASTERS, dtype=numpy.int32)
                for slot, (sname, sv, sp, pcfr) in enumerate(shadow_slots):
                    shadow_view_mats[slot] = sv
                    shadow_proj_mats[slot] = sp
                    shadow_pcf_radii[slot] = pcfr
                    depth_tex = (self.gl_data['sensor_buffers'][sname]
                                 ['main_fbo'].depth_texture)
                    GL.glActiveTexture(GL.GL_TEXTURE4 + slot)
                    GL.glBindTexture(GL.GL_TEXTURE_2D, depth_tex)

                if 'shadow_view_matrices' in location_data:
                    GL.glUniformMatrix4fv(
                        location_data['shadow_view_matrices'],
                        MAX_SHADOW_CASTERS, GL.GL_TRUE, shadow_view_mats)
                if 'shadow_projection_matrices' in location_data:
                    GL.glUniformMatrix4fv(
                        location_data['shadow_projection_matrices'],
                        MAX_SHADOW_CASTERS, GL.GL_TRUE, shadow_proj_mats)
                if 'shadow_pcf_radii' in location_data:
                    GL.glUniform1iv(location_data['shadow_pcf_radii'],
                                    MAX_SHADOW_CASTERS, shadow_pcf_radii)
                if 'direction_light_shadow_slots' in location_data:
                    GL.glUniform1iv(
                        location_data['direction_light_shadow_slots'],
                        max_num_lights, dir_shadow_slots)
                if 'image_light_shadow_slot' in location_data:
                    GL.glUniform1i(
                        location_data['image_light_shadow_slot'], ibl_shadow_slot)
                if 'image_light_shadow_direction' in location_data:
                    GL.glUniform3fv(
                        location_data['image_light_shadow_direction'], 1,
                        ibl_shadow_dir)
                if 'image_light_shadow_color' in location_data:
                    shadow_color = self.loaded_data['irradiance_sh'].get(
                        image_light_name + '_shadow_color',
                        numpy.zeros(3, dtype=numpy.float32))
                    GL.glUniform3fv(
                        location_data['image_light_shadow_color'], 1,
                        shadow_color)
                
                # set the image light parameters
                GL.glUniform1i(location_data['image_light_active'],
                        image_light_name is not None)
                if image_light_name is not None:
                    image_light_data = self.get_image_light(image_light_name)
                    
                    # set the offset matrix
                    offset_matrix = image_light_data['offset_matrix']
                    GL.glUniformMatrix4fv(
                            location_data['image_light_offset_matrix'],
                            1, GL.GL_TRUE,
                            offset_matrix.astype(numpy.float32))
                    
                    GL.glUniform1i(
                            location_data['lock_image_light_to_camera'],
                            image_light_data['lock_to_camera'])
                    
                    image_light_properties = numpy.array([
                            image_light_data['diffuse_scale'],
                            image_light_data['diffuse_bias'],
                            image_light_data['reflect_gamma'],
                            image_light_data['reflect_bias']])
                    GL.glUniform4fv(
                            location_data['image_light_properties'],
                            1, image_light_properties.astype(numpy.float32))
                
                # set the background color
                GL.glUniform3fv(location_data['background_color'], 1,
                        self.get_background_color()[:3].astype(numpy.float32))
                
                for material_name in shader_instances:
                    self.load_material_shader_data(material_name, shader_name)
                    for mesh_name in shader_instances[material_name]:
                        self.load_mesh_color_shader_data(mesh_name, shader_name)
                        instances = shader_instances[material_name][mesh_name]
                        for instance in instances:
                            self.color_render_instance(
                                    instance, shader_name)
                        self.unload_mesh_shader_data(mesh_name)
            
            finally:
                GL.glUseProgram(0)

        # Draw line overlays into whichever FBO is currently bound
        # (intermediate if distortion, main sensor otherwise) so they pass
        # through the same lens distortion as the rest of the scene geometry.
        if self.scene_description['coord_frames']:
            self._render_scene_coord_frames(render_camera_data, flip_y=flip_y)
        if self.scene_description['frustums']:
            self._render_scene_frustums(render_camera_data, flip_y=flip_y)
        self._render_scene_overlays(render_camera_data, flip_y=flip_y)

        # Stage 2: warp intermediate render into main sensor FBO
        if use_distortion:
            self._radial_warp_pass(sensor, camera_data)

        if finish:
            self.finish_frame()

    def load_mesh_color_shader_data(self, mesh_name, shader_name):
        
        # bind mesh buffers
        mesh_buffers = self.gl_data['mesh_buffers'][mesh_name]
        GL.glBindVertexArray(mesh_buffers['vao'])
    
    def unload_mesh_shader_data(self, mesh_name):
        mesh_buffers = self.gl_data['mesh_buffers'][mesh_name]
        #mesh_buffers['face_buffer'].unbind()
        #mesh_buffers['vertex_buffer'].unbind()
        GL.glBindVertexArray(0)
    
    def load_material_shader_data(self, material_name, shader_name):
        material_data = (
                self.scene_description['materials'][material_name])
        
        # get the shader variable locations
        location_data = self.shader_library.get_shader_locations(shader_name)
        
        # set the material properties
        if shader_name == 'textured_material_properties_shader':
            mat_prop_texture = self.get_material_properties_texture(
                material_name)
            mat_prop_texture_buffer = (
                self.gl_data['texture_buffers'][mat_prop_texture]['texture'])
            GL.glActiveTexture(GL.GL_TEXTURE1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, mat_prop_texture_buffer)
            GL.glUniform1i(location_data['material_properties_sampler'], 1)
        else:
            material_properties = numpy.array([
                    material_data['metal'],
                    material_data['rough'],
                    material_data['base_reflect'],
                    material_data['ambient'],
            ])
            
            GL.glUniform4fv(
                    location_data['material_properties'],
                    1, material_properties.astype(numpy.float32))
        
        # apply the albedo based on the shader type
        if shader_name in (
            'textured_shader', 'textured_material_properties_shader'):
            texture = self.get_material_texture(material_name)
            texture_buffer = self.gl_data['texture_buffers'][texture]['texture']
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, texture_buffer)
            GL.glUniform1i(location_data['texture_sampler'], 0)
        
        if shader_name == 'flat_color_shader':
            flat_color = self.get_material_flat_color(material_name)
            GL.glUniform3fv(
                    location_data['flat_color'],
                    1, numpy.array(flat_color, dtype=numpy.float32))
    
    def color_render_instance(self, instance_name, shader_name):
        """
        Renders a single instance.  Assumes load_mesh_shader_data and
        load_mesh_material_data have already been called.
        
        Parameters:
        -----------
        instance_name : str
            The instance to render
        shader_name : str
            The shader being used to render the instance
        """
        # get instance data
        instance_data = self.scene_description['instances'][instance_name]
        instance_mesh = instance_data['mesh_name']
        mesh_data = self.loaded_data['meshes'][instance_mesh]
        num_triangles = len(mesh_data['faces'])
        
        # get the shader variable locations
        location_data = self.shader_library.get_shader_locations(shader_name)
        
        # set the model pose
        GL.glUniformMatrix4fv(
            location_data['model_pose'],
            1, GL.GL_TRUE,
            instance_data['transform'].astype(numpy.float32),
        )

        GL.glDrawElements(
            GL.GL_TRIANGLES,
            num_triangles*3,
            GL.GL_UNSIGNED_INT,
            None,
        )
    
    def color_render_depthmap_instance(self, depthmap_instance_name):
        """
        Render a single depthmap instance using the color program.
        
        Parameters:
        -----------
        depthmap_instance_name : str
            The depthmap instance to render
        """
        depthmap_instance_data = self.scene_description['depthmap_instances'][
                depthmap_instance_name]
        instance_depthmap = depthmap_instance_data['depthmap_name']
        instance_texture = depthmap_instance_data['texture_name']
        depthmap_buffers = self.gl_data['depthmap_buffers'][instance_depthmap]
        texture_buffers = self.gl_data['texture_buffers'][instance_texture]
        depth_data = self.loaded_data['depthmaps'][instance_depthmap]
        location_data = self.shader_library.get_shader_locations(
                'textured_depthmap_shader')

        GL.glUniformMatrix4fv(
                location_data['model_pose'],
                1, GL.GL_TRUE,
                depthmap_instance_data['transform'].astype(numpy.float32))

        depthmap_data = self.scene_description['depthmaps'][instance_depthmap]
        focal_length = numpy.array(
                depthmap_data['focal_length'],
                dtype = numpy.float32)
        width = depthmap_data['width']
        height = depthmap_data['height']
        GL.glUniform2fv(
                location_data['focal_length'],
                1, focal_length)
        GL.glUniform1i(location_data['width'], width)
        GL.glUniform1i(location_data['height'], height)

        GL.glPointSize(depthmap_instance_data['point_size'])

        GL.glBindVertexArray(depthmap_buffers['vao'])
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture_buffers['texture'])
        num_points = depth_data.shape[0] * depth_data.shape[1]
        GL.glDrawElements(
                GL.GL_POINTS,
                num_points,
                GL.GL_UNSIGNED_INT,
                None)
        GL.glBindVertexArray(0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def render_background(self,
        image_light_name,
        view_matrix,
        projection_matrix,
        flip_y=True,
    ):
        """
        Renders the background (reflection map).

        Parameters:
        -----------
        image_light_name : str
            The image light with the reflection map we wish to render.
        view_matrix : 4x4 numpy array
        projection_matrix : 4x4 numpy array
        flip_y : bool, default=True
        """
        self.shader_library.use_program('background_shader')

        mesh_buffers = self.gl_data['mesh_buffers']['BACKGROUND']
        light_data = self.scene_description['image_lights'][image_light_name]
        reflect_cubemap = light_data['reflect_cubemap']
        cubemap_buffers = self.gl_data['cubemap_buffers'][reflect_cubemap]

        location_data = self.shader_library.get_shader_locations(
                'background_shader')

        # set the camera's view_matrix
        if light_data['lock_to_camera']:
            view_matrix = numpy.eye(4)
        GL.glUniformMatrix4fv(
                location_data['view_matrix'],
                1, GL.GL_TRUE,
                view_matrix.astype(numpy.float32))

        # set the camera's projection matrix
        if flip_y:
            projection_matrix = numpy.dot(
                    projection_matrix,
                    numpy.array([
                        [ 1, 0, 0, 0],
                        [ 0,-1, 0, 0],
                        [ 0, 0, 1, 0],
                        [ 0, 0, 0, 1]]))
        GL.glUniformMatrix4fv(
                location_data['projection_matrix'],
                1, GL.GL_TRUE,
                projection_matrix.astype(numpy.float32))

        # set the offset matrix
        offset_matrix = light_data['offset_matrix']
        GL.glUniformMatrix4fv(
                location_data['offset_matrix'],
                1, GL.GL_TRUE,
                offset_matrix.astype(numpy.float32))

        # set the blur
        blur = light_data['blur']
        GL.glUniform1f(location_data['blur'], blur)
        
        GL.glBindVertexArray(mesh_buffers['vao'])

        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(
            GL.GL_TEXTURE_CUBE_MAP,
            cubemap_buffers['cubemap'],
        )
        GL.glUniform1i(location_data['cubemap_sampler'], 0)
        #GL.glTexParameterf(
        #        GL.GL_TEXTURE_CUBE_MAP,
        #        GL.GL_TEXTURE_MIN_LOD,
        #        0)
        # THIS IS WEIRD... THE MIN_LOD LOOKS BETTER FOR THE REFLECTIONS,
        # BUT WORSE HERE.  MAYBE THE RIGHT THING IS TO GENERATE BLURRED
        # MIPMAPS AND DO EXPLICIT LOD LOOKUPS INSTEAD OF BIAS???

        GL.glDrawElements(
                GL.GL_TRIANGLES,
                2*3,
                GL.GL_UNSIGNED_INT,
                None)

    # mask_render methods ------------------------------------------------------
    
    def mask_render(
        self,
        camera,
        instances=None,
        sensor=None,
        flip_y=True,
        clear=True,
        finish=True,
        ignore_hidden=False,
    ):
        """
        Renders instances using the mask program.

        Parameters:
        -----------
        camera : str
            Name of the camera to render from.
        instances : list, optional
            A list of instances to render.  If not specified, all instances will
            be rendered.
        sensor : str, optional
            Name of the sensor to render into.
        flip_y : bool, default=True
        clear : bool, default=True
        finish : bool, default=True
        """

        camera_data = self.scene_description['cameras'][camera]

        if sensor is not None:
            self._bind_sensor(sensor)

        # clear
        if clear:
            self.clear_frame()

        # turn on the shader
        self.shader_library.use_program('mask_shader')

        try:
            location_data = self.shader_library.get_shader_locations(
                    'mask_shader')

            # set the camera's view matrix
            view_matrix = camera_data['view_matrix']
            GL.glUniformMatrix4fv(
                    location_data['view_matrix'],
                    1, GL.GL_TRUE,
                    view_matrix.astype(numpy.float32))

            # set the camera's projection matrix
            projection_matrix = camera_data['projection']
            if flip_y:
                projection_matrix = numpy.dot(
                        projection_matrix,
                        numpy.array([
                            [1,0,0,0],
                            [0,-1,0,0],
                            [0,0,1,0],
                            [0,0,0,1]]))
            GL.glUniformMatrix4fv(
                    location_data['projection_matrix'],
                    1, GL.GL_TRUE,
                    projection_matrix.astype(numpy.float32))
            
            if instances is None:
                instances = self.scene_description['instances'].keys()
            
            # sort the instances
            mesh_instances = {}
            for instance in instances:
                if not ignore_hidden and self.instance_hidden(instance):
                    continue
                instance_mesh = self.get_instance_mesh_name(instance)
                try:
                    mesh_instances[instance_mesh].append(instance)
                except KeyError:
                    mesh_instances[instance_mesh] = [instance]
            
            # render the instances
            for mesh_name in mesh_instances:
                self.load_mesh_mask_shader_data(mesh_name)
                instances = mesh_instances[mesh_name]
                for instance in instances:
                    self.mask_render_instance(instance)
                self.unload_mesh_shader_data(mesh_name)

        finally:
            GL.glUseProgram(0)
        
        if finish:
            GL.glFinish()
    
    def load_mesh_mask_shader_data(self, mesh_name):
        
        # bind mesh buffers
        mesh_buffers = self.gl_data['mesh_buffers'][mesh_name]
        #mesh_buffers['face_buffer'].bind()
        #mesh_buffers['vertex_buffer'].bind()
        GL.glBindVertexArray(mesh_buffers['vao'])
        
        ## get the shader variable locations
        #location_data = self.shader_library.get_shader_locations('mask_shader')
        #
        #GL.glEnableVertexAttribArray(location_data['vertex_position'])
        #stride = self.get_mesh_stride(mesh_name)
        #GL.glVertexAttribPointer(
        #        location_data['vertex_position'],
        #        3, GL.GL_FLOAT, False, stride,
        #        mesh_buffers['vertex_buffer'])
    
    def mask_render_instance(self, instance_name):
        instance_data = self.scene_description['instances'][instance_name]
        location_data = self.shader_library.get_shader_locations('mask_shader')
        GL.glUniformMatrix4fv(
                location_data['model_pose'],
                1, GL.GL_TRUE,
                numpy.array(instance_data['transform'], dtype=numpy.float32))
        mask_color = instance_data['mask_color']
        GL.glUniform3fv(
                location_data['mask_color'],
                1, numpy.array(mask_color, dtype=numpy.float32))
        mesh = self.loaded_data['meshes'][instance_data['mesh_name']]
        GL.glDrawElements(
                GL.GL_TRIANGLES,
                len(mesh['faces'])*3,
                GL.GL_UNSIGNED_INT,
                None)
    
    # coord_render methods -----------------------------------------------------
    
    def coord_render(self,
        camera,
        instances=None,
        sensor=None,
        flip_y=True,
        clear=True,
        finish=True,
        ignore_hidden=False,
    ):
        """
        Renders instances using the coord program.

        Parameters:
        -----------
        camera : str
            Name of the camera to render from.
        instances : list, optional
            A list of instances to render.  If not specified, all instances will
            be rendered.
        sensor : str, optional
            Name of the sensor to render into.
        flip_y : bool, default=True
        clear : bool, default=True
        finish : bool, default=True
        """

        camera_data = self.scene_description['cameras'][camera]

        if sensor is not None:
            self._bind_sensor(sensor)

        # clear
        if clear:
            self.clear_frame()

        # turn on the shader
        self.shader_library.use_program('coord_shader')

        try:
            location_data = self.shader_library.get_shader_locations(
                    'coord_shader')
            view_matrix = camera_data['view_matrix']
            GL.glUniformMatrix4fv(
                    location_data['view_matrix'],
                    1, GL.GL_TRUE,
                    view_matrix.astype(numpy.float32))

            projection_matrix = camera_data['projection']
            if flip_y:
                projection_matrix = numpy.dot(
                        projection_matrix,
                        numpy.array([
                            [1,0,0,0],
                            [0,-1,0,0],
                            [0,0,1,0],
                            [0,0,0,1]]))
            GL.glUniformMatrix4fv(
                    location_data['projection_matrix'],
                    1, GL.GL_TRUE,
                    projection_matrix.astype(numpy.float32))

            # render all instances
            if instances is None:
                instances = self.scene_description['instances']
            for instance_name in instances:
                if not ignore_hidden and self.instance_hidden(instance_name):
                    continue
                self.coord_render_instance(instance_name)

        finally:
            GL.glUseProgram(0)
        
        if finish:
            GL.glFinish()

    def coord_render_instance(self, instance_name):
        """
        Render a single instance using the coord program.
        
        Parameters:
        -----------
        instance_name : str
            The instance to render
        TODO (figure out large scene optimizations first):
        set_mesh_attrib_pointers : bool, default=True
            An optimization.  If the same mesh was rendered previously, there
            is no need to copy certain data to the GPU again.
        """
        
        instance_data = self.scene_description['instances'][instance_name]

        instance_mesh = instance_data['mesh_name']
        coord_box = instance_data['coord_box']
        mesh_buffers = self.gl_data['mesh_buffers'][instance_mesh]
        mesh = self.loaded_data['meshes'][instance_mesh]

        location_data = self.shader_library.get_shader_locations(
                'coord_shader')

        GL.glUniformMatrix4fv(
                location_data['model_pose'],
                1, GL.GL_TRUE,
                numpy.array(instance_data['transform'], dtype=numpy.float32))

        GL.glUniform3fv(
                location_data['box_min'],
                1, numpy.array(coord_box[0], dtype=numpy.float32))

        GL.glUniform3fv(
                location_data['box_max'],
                1, numpy.array(coord_box[1], dtype=numpy.float32))

        GL.glBindVertexArray(mesh_buffers['vao'])
        GL.glDrawElements(
                GL.GL_TRIANGLES,
                len(mesh['faces'])*3,
                GL.GL_UNSIGNED_INT,
                None)
        GL.glBindVertexArray(0)

    # one-shot draw methods ====================================================

    def _compute_vp(self, camera, flip_y=True):
        """Compute view-projection matrix from a camera name."""
        camera_data = self.scene_description['cameras'][camera]
        view = camera_data['view_matrix'].astype(numpy.float32)
        proj = camera_data['projection'].astype(numpy.float32)
        if flip_y:
            proj = numpy.array(
                [[1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]],
                dtype=numpy.float32) @ proj
        return proj @ view

    def _draw_primitives(self, vp, vertices, draw_mode=None):
        """Upload world-space vertices and draw with the lines shader."""
        if draw_mode is None:
            draw_mode = GL.GL_LINES
        vertices = numpy.asarray(vertices, dtype=numpy.float32).ravel()
        num_verts = len(vertices) // 6
        if num_verts == 0:
            return

        buffers = self.gl_data['coord_frame_buffers']
        self.shader_library.use_program('lines_shader')
        locations = self.shader_library.get_shader_locations('lines_shader')
        GL.glUniformMatrix4fv(
            locations['mvp_matrix'], 1, GL.GL_TRUE, vp.astype(numpy.float32))

        GL.glBindVertexArray(buffers['vao'])
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, buffers['vbo'])
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL.GL_STREAM_DRAW)
        GL.glDrawArrays(draw_mode, 0, num_verts)
        GL.glBindVertexArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glUseProgram(0)

    @staticmethod
    def _build_coord_frame_verts(transforms, axis_length=0.1):
        """Build world-space line vertices for coordinate frames."""
        verts = []
        for t in transforms:
            t = numpy.asarray(t, dtype=numpy.float32)
            origin = t[:3, 3]
            axes = t[:3, :3] * axis_length
            colors = [(1,0,0),(0,1,0),(0,0,1),(1,0,1),(1,1,0),(0,1,1)]
            signs = [1, 1, 1, -1, -1, -1]
            for i in range(3):
                for s, ci in [(1, i), (-1, i+3)]:
                    c = colors[ci]
                    tip = origin + axes[:, i] * s
                    verts.extend([*origin, *c, *tip, *c])
        return numpy.array(verts, dtype=numpy.float32)

    @staticmethod
    def _build_frustum_verts(transform, projection, color):
        """Build world-space line vertices for a camera frustum."""
        inv_proj = numpy.linalg.inv(projection)
        corners_ndc = numpy.array([
            [-1,-1,-1,1],[ 1,-1,-1,1],[ 1, 1,-1,1],[-1, 1,-1,1],
            [-1,-1, 1,1],[ 1,-1, 1,1],[ 1, 1, 1,1],[-1, 1, 1,1],
        ], dtype=numpy.float32)
        corners_view = (inv_proj @ corners_ndc.T).T
        corners_view /= corners_view[:, 3:4]
        corners_world = (transform @ corners_view.T).T[:, :3]
        edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
                 (0,4),(1,5),(2,6),(3,7)]
        c = numpy.array(color, dtype=numpy.float32)
        verts = []
        for i, j in edges:
            verts.extend([*corners_world[i], *c, *corners_world[j], *c])
        return numpy.array(verts, dtype=numpy.float32)

    @staticmethod
    def _build_line_set_verts(starts, ends, colors):
        """Build world-space line vertices from parallel arrays."""
        starts = numpy.asarray(starts, dtype=numpy.float32)
        ends = numpy.asarray(ends, dtype=numpy.float32)
        colors = numpy.asarray(colors, dtype=numpy.float32)
        if colors.ndim == 1:
            colors = numpy.broadcast_to(colors, starts.shape)
        n = len(starts)
        verts = numpy.zeros((n * 2, 6), dtype=numpy.float32)
        verts[0::2, :3] = starts
        verts[0::2, 3:] = colors
        verts[1::2, :3] = ends
        verts[1::2, 3:] = colors
        return verts.ravel()

    @staticmethod
    def _build_point_cloud_verts(points, colors):
        """Build world-space point vertices from parallel arrays."""
        points = numpy.asarray(points, dtype=numpy.float32)
        colors = numpy.asarray(colors, dtype=numpy.float32)
        if colors.ndim == 1:
            colors = numpy.broadcast_to(colors, points.shape)
        n = len(points)
        verts = numpy.zeros((n, 6), dtype=numpy.float32)
        verts[:, :3] = points
        verts[:, 3:] = colors
        return verts.ravel()

    @staticmethod
    def _build_box_verts(transform, color):
        """Build world-space wireframe vertices for a box (transform maps unit cube)."""
        t = numpy.asarray(transform, dtype=numpy.float32)
        c = numpy.array(color, dtype=numpy.float32)
        # Unit cube corners
        signs = numpy.array([
            [-1,-1,-1],[ 1,-1,-1],[ 1, 1,-1],[-1, 1,-1],
            [-1,-1, 1],[ 1,-1, 1],[ 1, 1, 1],[-1, 1, 1],
        ], dtype=numpy.float32)
        corners_local = numpy.hstack([signs, numpy.ones((8,1), dtype=numpy.float32)])
        corners_world = (t @ corners_local.T).T[:, :3]
        edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
                 (0,4),(1,5),(2,6),(3,7)]
        verts = []
        for i, j in edges:
            verts.extend([*corners_world[i], *c, *corners_world[j], *c])
        return numpy.array(verts, dtype=numpy.float32)

    @staticmethod
    def _build_arrow_verts(start, end, color, wedge_size=0.1):
        """Build world-space line vertices for an arrow with a wedge tip."""
        start = numpy.asarray(start, dtype=numpy.float32)
        end = numpy.asarray(end, dtype=numpy.float32)
        c = numpy.array(color, dtype=numpy.float32)
        direction = end - start
        length = numpy.linalg.norm(direction)
        if length < 1e-8:
            return numpy.array([], dtype=numpy.float32)
        d = direction / length
        # Find a perpendicular vector
        up = numpy.array([0, 1, 0], dtype=numpy.float32)
        if abs(numpy.dot(d, up)) > 0.99:
            up = numpy.array([1, 0, 0], dtype=numpy.float32)
        perp = numpy.cross(d, up)
        perp = perp / numpy.linalg.norm(perp)
        # Wedge base point
        wedge_len = length * wedge_size
        base = end - d * wedge_len
        w1 = base + perp * wedge_len * 0.4
        w2 = base - perp * wedge_len * 0.4
        verts = [
            *start, *c, *end, *c,   # shaft
            *end, *c, *w1, *c,      # wedge arm 1
            *end, *c, *w2, *c,      # wedge arm 2
        ]
        return numpy.array(verts, dtype=numpy.float32)

    def render_coord_frames(self, camera, transforms, axis_length=0.1,
                            flip_y=True):
        """One-shot: draw coordinate frames at the given transforms."""
        vp = self._compute_vp(camera, flip_y)
        verts = self._build_coord_frame_verts(transforms, axis_length)
        self._draw_primitives(vp, verts)

    def render_frustums(self, camera, transforms, projections,
                        color=(1, 1, 1), flip_y=True):
        """One-shot: draw camera frustum wireframes."""
        vp = self._compute_vp(camera, flip_y)
        verts = []
        transforms = numpy.asarray(transforms)
        projections = numpy.asarray(projections)
        if transforms.ndim == 2:
            transforms = transforms[None]
            projections = projections[None]
        for t, p in zip(transforms, projections):
            verts.append(self._build_frustum_verts(t, p, color))
        if verts:
            self._draw_primitives(vp, numpy.concatenate(verts))

    def render_line_sets(self, camera, starts, ends, colors, flip_y=True):
        """One-shot: draw line segments from starts to ends."""
        vp = self._compute_vp(camera, flip_y)
        verts = self._build_line_set_verts(starts, ends, colors)
        self._draw_primitives(vp, verts)

    def render_point_clouds(self, camera, points, colors, point_size=1,
                            flip_y=True):
        """One-shot: draw points."""
        vp = self._compute_vp(camera, flip_y)
        GL.glPointSize(point_size)
        verts = self._build_point_cloud_verts(points, colors)
        self._draw_primitives(vp, verts, draw_mode=GL.GL_POINTS)

    def render_boxes(self, camera, transforms, colors, flip_y=True):
        """One-shot: draw wireframe boxes (each transform maps the unit cube)."""
        vp = self._compute_vp(camera, flip_y)
        transforms = numpy.asarray(transforms)
        colors = numpy.asarray(colors)
        if transforms.ndim == 2:
            transforms = transforms[None]
        if colors.ndim == 1:
            colors = numpy.broadcast_to(colors, (len(transforms), 3))
        verts = []
        for t, c in zip(transforms, colors):
            verts.append(self._build_box_verts(t, c))
        if verts:
            self._draw_primitives(vp, numpy.concatenate(verts))

    def render_arrows(self, camera, starts, ends, colors, wedge_size=0.1,
                      flip_y=True):
        """One-shot: draw arrows from starts to ends with wedge tips."""
        vp = self._compute_vp(camera, flip_y)
        starts = numpy.asarray(starts, dtype=numpy.float32)
        ends = numpy.asarray(ends, dtype=numpy.float32)
        colors = numpy.asarray(colors, dtype=numpy.float32)
        if starts.ndim == 1:
            starts = starts[None]
            ends = ends[None]
        if colors.ndim == 1:
            colors = numpy.broadcast_to(colors, (len(starts), 3))
        verts = []
        for s, e, c in zip(starts, ends, colors):
            verts.append(self._build_arrow_verts(s, e, c, wedge_size))
        non_empty = [v for v in verts if len(v) > 0]
        if non_empty:
            self._draw_primitives(vp, numpy.concatenate(non_empty))

def print_locations(shader, location_data):
    for k,v in location_data.items():
        print(k)
        print(f'  {v}')
        if k in ('texture_sampler', 'material_properties_sampler'):
            texture_unit = numpy.array([0], dtype=numpy.int32)
            GL.glGetUniformiv(shader, v, texture_unit)
            GL.glActiveTexture(GL.GL_TEXTURE0 + texture_unit[0])
            t = GL.glGetIntegerv(GL.GL_TEXTURE_BINDING_2D)
            print(' ', texture_unit[0], t)
        
        if k in ('reflect_sampler', 'diffuse_sampler'): 
            texture_unit = numpy.array([0], dtype=numpy.int32)
            GL.glGetUniformiv(shader, v, texture_unit)
            GL.glActiveTexture(GL.GL_TEXTURE0 + texture_unit[0])
            t = GL.glGetIntegerv(GL.GL_TEXTURE_BINDING_CUBE_MAP)
            print(' ', texture_unit[0], t)
