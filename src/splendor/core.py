# system
import math
import json
import os

# opengl
from OpenGL import GL
from OpenGL.arrays import vbo

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
from splendor.exceptions import SplendorException
from splendor.primitives import make_primitive

max_num_lights = 8
default_default_view_matrix = numpy.eye(4)
default_default_camera_projection = camera.projection_matrix(
        math.radians(90.), 1.0)

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
            ('direction_light', 'direction_lights'))

    def __init__(self,
            assets=None,
            default_view_matrix=None,
            default_camera_projection=None):
        """
        SplendorRender initialization
        
        Parameters
        ----------
        assets : str or AssetLibrary, optional
            Either a path pointing to an asset library cfg file or an
            AssetLibrary object.  This is used to load assets such as meshes
            and textures by name rather than their full path.  If not provided,
            this will load the splendor-render default asset library.
        default_view_matrix : 4x4 numpy matrix, optional
            The default view matrix for the renderer.  This is the inverse of
            thre 3D pose of the camera object.  Identity if not specified.
        default_camera_projection : 4x4 numpy matrix, optional
            The default projection matrix for the renderer.
            A square projection with a 90 degree fov is used if not specified.
        """
        # asset library
        if isinstance(assets, AssetLibrary):
            self.asset_library = assets
        else:
            self.asset_library = AssetLibrary(assets)

        # default camera settings
        if default_view_matrix is None:
            default_view_matrix = default_default_view_matrix
        self.default_view_matrix = default_view_matrix
        if default_camera_projection is None:
            default_camera_projection = default_default_camera_projection
        self.default_camera_projection = default_camera_projection

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
            'camera':{
                'view_matrix':default_view_matrix,
                'projection':default_camera_projection,
            },
            'image_lights':{},
            'active_image_light':None,
        }

        self.loaded_data = {
            'meshes':{},
            'depthmaps':{},
            'textures':{},
            'cubemaps':{},
        }

        self.gl_data = {
            'mesh_buffers':{},
            'depthmap_buffers':{},
            'texture_buffers':{},
            'cubemap_buffers':{},
        }
        
        self.opengl_init()
        self.shader_library = ShaderLibrary()
    
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
        GL.glEnable(GL.GL_TEXTURE_CUBE_MAP_SEAMLESS)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glDepthFunc(GL.GL_LESS)
        GL.glDepthRange(0.0, 1.0)

        GL.glClearColor(0.,0.,0.,0.)

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

        if 'camera' in scene:
            if 'view_matrix' in scene['camera']:
                self.set_view_matrix(scene['camera']['view_matrix'])
            if 'projection' in scene['camera']:
                self.set_projection(scene['camera']['projection'])

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
        self.reset_camera()

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
    
    def reset_camera(self):
        """
        Resets the camera to the default view matrix and projection.
        """
        self.set_view_matrix(self.default_view_matrix)
        self.set_projection(self.default_camera_projection)

    def set_projection(self, projection_matrix):
        """
        Sets the camera projection matrix.
        
        Parameters:
        -----------
        projection_matrix : 4x4 array-like
        """
        self.scene_description['camera']['projection'] = numpy.array(
                projection_matrix)

    def get_projection(self):
        """
        Get the camera's projection matrix.
        
        Returns:
        --------
        4x4 numpy array
        """
        return self.scene_description['camera']['projection']

    def set_view_matrix(self, view_matrix):
        """
        Sets the view matrix.
        
        Parameters:
        -----------
        view_matrix : 4x4 matrix, 6-element or 9-element azimuthal parameters
            or a dictionary of named azimuthal parameters.
            Azimuthal parameters are:
            [azimuth,
             elevation,
             tilt,
             distance,
             shift_x,
             shift_y,
             center_x (optional),
             center_y (optional),
             center_z (optional)]
        """
        view_matrix = camera.view_matrix(view_matrix)
        self.scene_description['camera']['view_matrix'] = view_matrix

    def get_view_matrix(self):
        """
        Get the view matrix.
        
        Note this is the inverse of the SE3 pose of the camera object.
        
        Returns:
        --------
        view_matrix : 4x4 numpy array
        """
        return self.scene_description['camera']['view_matrix']

    def camera_frame_scene(self, multiplier=3.0, *args, **kwargs):
        bbox = self.get_instance_center_bbox()
        view_matrix = camera.frame_bbox(
                bbox, self.get_projection(), multiplier,
                *args, **kwargs)
        self.set_view_matrix(view_matrix)

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
        color_mode : {"textured", "vertex_color", "flat"}
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
            mesh_buffers['vertex_buffer'] = vbo.VBO(combined_floats)

        elif color_mode == 'vertex_color':
            assert 'vertex_colors' in mesh
            vertex_color_floats = numpy.array(
                    mesh['vertex_colors'], dtype=numpy.float32)
            combined_floats = numpy.concatenate(
                    (vertex_floats, normal_floats, vertex_color_floats), axis=1)
            mesh_buffers['vertex_buffer'] = vbo.VBO(combined_floats)

        elif color_mode == 'flat_color':
            combined_floats = numpy.concatenate(
                    (vertex_floats, normal_floats), axis=1)
            mesh_buffers['vertex_buffer'] = vbo.VBO(combined_floats)

        face_ints = numpy.array(mesh['faces'], dtype=numpy.int32)
        mesh_buffers['face_buffer'] = vbo.VBO(
                face_ints,
                target = GL.GL_ELEMENT_ARRAY_BUFFER)

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
                    target = GL.GL_ELEMENT_ARRAY_BUFFER)
            self.gl_data['mesh_buffers']['BACKGROUND'] = mesh_buffers

    def remove_mesh(self, name):
        """
        Deletes a mesh from the scene.
        
        Parameters:
        -----------
        name : str
        """
        del(self.scene_description['meshes'][name])
        self.gl_data['mesh_buffers'][name]['vertex_buffer'].delete()
        self.gl_data['mesh_buffers'][name]['face_buffer'].delete()
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
            raise SplendorException(
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
                target = GL.GL_ELEMENT_ARRAY_BUFFER)

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
        self.gl_data['depthmap_buffers'][name]['depth_buffer'].delete()
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
            diffuse_cubemap,
            reflect_cubemap,
            offset_matrix = numpy.eye(4),
            blur = 0.,
            diffuse_gamma = 1.,
            diffuse_bias = 0.,
            reflect_gamma = 1.,
            reflect_bias = 0.,
            render_background = True,
            set_active = False):
        """
        Load an image light.
        
        Loads an image light into memory but does not make it active unless
        set_active=True.  Only one image_light can be active at a time.
        
        Parameters:
        -----------
        name : str
            Name of the image light, must be unique to this scene among other
            image lights
        diffuse_cubemap : str
            The name of the cubemap to use for diffuse lighting component.
        reflect_cubemap : str
            The name of the cubemap to use for reflect lighting component.
        offset_matrix : 4x4 array-like, default=numpy.eye(4)
            An offset rotation matrix for the image light.
        blur : float, default=0.
            Blur to apply to the background when the background is visible.
        diffuse_gamma : float, default=1.
            A gamma correction for the diffuse component of the image light.
            Values above one increase the contrast between dark and light
            sides of an object.
        diffuse_bias : float, default=0.
            A bias for the diffuse component of the image light.
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
        image_light_data['diffuse_cubemap'] = diffuse_cubemap
        image_light_data['reflect_cubemap'] = reflect_cubemap
        image_light_data['offset_matrix'] = numpy.array(offset_matrix)
        image_light_data['blur'] = blur
        image_light_data['render_background'] = render_background
        image_light_data['diffuse_gamma'] = diffuse_gamma
        image_light_data['diffuse_bias'] = diffuse_bias
        image_light_data['reflect_gamma'] = reflect_gamma
        image_light_data['reflect_bias'] = reflect_bias
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

        # delete the background mesh if there are no image lights left
        if len(self.scene_description['image_lights']) == 0:
            self.gl_data['mesh_buffers']['BACKGROUND']['vertex_buffer'].delete()
            self.gl_data['mesh_buffers']['BACKGROUND']['face_buffer'].delete()
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
                        Gl.glTexImage2d(
                            Gl.GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
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
                GL.GL_LINEAR_MIPMAP_LINEAR,
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
        for instance_name, index in instance_indices.items():
            instance_data = self.scene_description['instances'][instance_name]
            instance_data['mask_color'] = masks.color_index_to_float(index)

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
    
    def add_point_light(self, name, position, color):
        """
        Add a point light to the scene.
        
        Parameters:
        -----------
        name : str
            Name of the new point light, must be unique to this scene among
            other point lights.
        position : array-like
            3-value array representing the position of the point light.
        color : array-like
            3-value color of the point light
        """
        self.scene_description['point_lights'][name] = {
                'position' : numpy.array(position),
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
            direction,
            color):
        """
        Add a direction light to the scene.
        
        Parameters:
        -----------
        name : str
            Name of the new direction light, must be unique to this scene among
            other direction lights.
        direction : array-like
            3-value array representing the direction of the light.
        color : array-like
            3-value color of the direction light
        """
        self.scene_description['direction_lights'][name] = {
                'direction' : numpy.array(direction),
                'color' : numpy.array(color)}

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
    
    def color_render(self,
            instances = None,
            depthmap_instances = None,
            flip_y = True,
            clear = True,
            finish = True):
        """
        Renders instances and depthmap instances using the color program.
        
        Parameters:
        -----------
        instances : list, optional
            A list of instances to render.  If not specified, all instances will
            be rendered.
        depthmap_instances : list, optional
            A list of depthmap instances to render.  If not specified, all
            depthmap instances will be rendered.
        flip_y : bool, default=True
            Whether or not to flip the image in Y when rendering.  This is to
            correct for the difference between rendering to windows and
            framebuffers.
        clear : bool, default=True
            Whether or not to clear the frame before rendering.
        finish = True
            Whether or not to finish the frame using glFinish
        """
        # clear
        if clear:
            self.clear_frame()
        
        # render the background
        image_light_name = self.scene_description['active_image_light']
        if image_light_name is not None:
            image_light_data = self.get_image_light(image_light_name)
            if image_light_data['render_background']:
                self.render_background(image_light_name, flip_y = flip_y)

            diffuse_cubemap = image_light_data['diffuse_cubemap']
            reflect_cubemap = image_light_data['reflect_cubemap']
            GL.glActiveTexture(GL.GL_TEXTURE2)
            GL.glBindTexture(
                GL.GL_TEXTURE_CUBE_MAP,
                self.gl_data['cubemap_buffers'][diffuse_cubemap]['cubemap'],
            )
            GL.glActiveTexture(GL.GL_TEXTURE3)
            GL.glBindTexture(
                GL.GL_TEXTURE_CUBE_MAP,
                self.gl_data['cubemap_buffers'][reflect_cubemap]['cubemap'],
            )

        # depthmap_instances
        if depthmap_instances is None:
            depthmap_instances = self.scene_description['depthmap_instances']

        self.shader_library.use_program('textured_depthmap_shader')
        try:
            location_data = self.shader_library.get_shader_locations(
                    'textured_depthmap_shader')
            
            # set the camera's view_matrix
            view_matrix = self.scene_description['camera']['view_matrix']
            GL.glUniformMatrix4fv(
                    location_data['view_matrix'],
                    1, GL.GL_TRUE,
                    view_matrix.astype(numpy.float32))

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
            GL.glUniformMatrix4fv(
                    location_data['projection_matrix'],
                    1, GL.GL_TRUE,
                    projection_matrix.astype(numpy.float32))

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
            if self.instance_hidden(instance):
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
                
                # set the camera's view matrix
                view_matrix = self.scene_description['camera']['view_matrix']
                GL.glUniformMatrix4fv(
                        location_data['view_matrix'],
                        1, GL.GL_TRUE,
                        view_matrix.astype(numpy.float32))

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
                    point_light_data[i*2] = light_data['color']
                    point_light_data[i*2+1] = light_data['position']
                GL.glUniform3fv(
                        location_data['point_light_data'], max_num_lights*2,
                        point_light_data.astype(numpy.float32))
                
                # set the direction light data
                GL.glUniform1i(
                        location_data['num_direction_lights'],
                        len(self.scene_description['direction_lights']))
                direction_light_data = numpy.zeros((max_num_lights*2,3))
                for i, light_name in enumerate(
                        self.scene_description['direction_lights']):
                    light_data = self.scene_description[
                            'direction_lights'][light_name]
                    direction_light_data[i*2] = light_data['color']
                    direction_light_data[i*2+1] = light_data['direction']
                GL.glUniform3fv(
                        location_data['direction_light_data'], max_num_lights*2,
                        direction_light_data.astype(numpy.float32))
                
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
                    
                    image_light_properties = numpy.array([
                            image_light_data['diffuse_gamma'],
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

        if finish:
            self.finish_frame()
    
    def load_mesh_color_shader_data(self, mesh_name, shader_name):
        
        # bind mesh buffers
        mesh_buffers = self.gl_data['mesh_buffers'][mesh_name]
        mesh_buffers['face_buffer'].bind()
        mesh_buffers['vertex_buffer'].bind()
        
        # get the shader variable locations
        location_data = self.shader_library.get_shader_locations(shader_name)
        
        # enable the attribute arrays
        GL.glEnableVertexAttribArray(location_data['vertex_position'])
        GL.glEnableVertexAttribArray(location_data['vertex_normal'])
        if shader_name in (
            'textured_shader', 'textured_material_properties_shader'):
            GL.glEnableVertexAttribArray(location_data['vertex_uv'])
        elif shader_name == 'vertex_color_shader':
            GL.glEnableVertexAttribArray(location_data['vertex_color'])
        
        # load the pointers to the vertex, normal, uv and vertex color data
        stride = self.get_mesh_stride(mesh_name)
        GL.glVertexAttribPointer(
                location_data['vertex_position'],
                3, GL.GL_FLOAT, False, stride,
                mesh_buffers['vertex_buffer'])
        GL.glVertexAttribPointer(
                location_data['vertex_normal'],
                3, GL.GL_FLOAT, False, stride,
                mesh_buffers['vertex_buffer']+((3)*4))
        if shader_name in (
            'textured_shader', 'textured_material_properties_shader'):
            GL.glVertexAttribPointer(
                    location_data['vertex_uv'],
                    2, GL.GL_FLOAT, False, stride,
                    mesh_buffers['vertex_buffer']+((3+3)*4))
        elif shader_name == 'vertex_color_shader':
            GL.glVertexAttribPointer(
                    location_data['vertex_color'],
                    3, GL.GL_FLOAT, False, stride,
                    mesh_buffers['vertex_buffer']+((3+3)*4))
    
    def unload_mesh_shader_data(self, mesh_name):
        mesh_buffers = self.gl_data['mesh_buffers'][mesh_name]
        mesh_buffers['face_buffer'].unbind()
        mesh_buffers['vertex_buffer'].unbind()
    
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
                instance_data['transform'].astype(numpy.float32))
        
        GL.glDrawElements(
                GL.GL_TRIANGLES,
                num_triangles*3,
                GL.GL_UNSIGNED_INT,
                None)
    
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

        depthmap_buffers['depth_buffer'].bind()
        depthmap_buffers['index_buffer'].bind()
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture_buffers['texture'])
        try:
            GL.glEnableVertexAttribArray(location_data['vertex_depth'])

            stride = 4
            GL.glVertexAttribPointer(
                    location_data['vertex_depth'],
                    1, GL.GL_FLOAT, False, stride,
                    depthmap_buffers['depth_buffer'])
            depth_data = self.loaded_data['depthmaps'][depthmap_instance_name]
            num_points = depth_data.shape[0] * depth_data.shape[1]
            GL.glDrawElements(
                    GL.GL_POINTS,
                    num_points,
                    GL.GL_UNSIGNED_INT,
                    None)

        finally:
            depthmap_buffers['depth_buffer'].unbind()
            depthmap_buffers['index_buffer'].unbind()
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def render_background(self, image_light_name, flip_y=True):
        """
        Renders the background (reflection map)
        
        Parameters:
        -----------
        image_light_name : str
            The image light with the reflection map we wish to render.
        flip_y : bool, default=True
            Whether or not to flip the image in Y when rendering.  This is to
            correct for the difference between rendering to windows and
            framebuffers.
        """
        self.shader_library.use_program('background_shader')

        mesh_buffers = self.gl_data['mesh_buffers']['BACKGROUND']
        light_data = self.scene_description['image_lights'][image_light_name]
        reflect_cubemap = light_data['reflect_cubemap']
        cubemap_buffers = self.gl_data['cubemap_buffers'][reflect_cubemap]
        
        num_triangles = 2

        location_data = self.shader_library.get_shader_locations(
                'background_shader')

        # set the camera's view_matrix
        view_matrix = self.scene_description['camera']['view_matrix']
        GL.glUniformMatrix4fv(
                location_data['view_matrix'],
                1, GL.GL_TRUE,
                view_matrix.astype(numpy.float32))

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

        mesh_buffers['face_buffer'].bind()
        mesh_buffers['vertex_buffer'].bind()

        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(
            GL.GL_TEXTURE_CUBE_MAP,
            cubemap_buffers['cubemap'],
        )
        #GL.glTexParameterf(
        #        GL.GL_TEXTURE_CUBE_MAP,
        #        GL.GL_TEXTURE_MIN_LOD,
        #        0)
        # THIS IS WEIRD... THE MIN_LOD LOOKS BETTER FOR THE REFLECTIONS,
        # BUT WORSE HERE.  MAYBE THE RIGHT THING IS TO GENERATE BLURRED
        # MIPMAPS AND DO EXPLICIT LOD LOOKUPS INSTEAD OF BIAS???

        try:
            GL.glDrawElements(
                    GL.GL_TRIANGLES,
                    2*3,
                    GL.GL_UNSIGNED_INT,
                    None)

        finally:
            mesh_buffers['face_buffer'].unbind()
            mesh_buffers['vertex_buffer'].unbind()
            GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, 0)

    # mask_render methods ------------------------------------------------------
    
    def mask_render(self, instances=None, flip_y=True, clear=True, finish=True):
        """
        Renders instances using the mask program.
        
        Parameters:
        -----------
        instances : list, optional
            A list of instances to render.  If not specified, all instances will
            be rendered.
        flip_y : bool, default=True
            Whether or not to flip the image in Y when rendering.  This is to
            correct for the difference between rendering to windows and
            framebuffers.
        clear : bool, default=True
            Whether or not to clear the frame before rendering.
        finish = True
            Whether or not to finish the frame using glFinish
        """

        # clear
        if clear:
            self.clear_frame()

        # turn on the shader
        self.shader_library.use_program('mask_shader')

        try:
            location_data = self.shader_library.get_shader_locations(
                    'mask_shader')

            # set the camera's view matrix
            view_matrix = self.scene_description['camera']['view_matrix']
            GL.glUniformMatrix4fv(
                    location_data['view_matrix'],
                    1, GL.GL_TRUE,
                    view_matrix.astype(numpy.float32))

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
            GL.glUniformMatrix4fv(
                    location_data['projection_matrix'],
                    1, GL.GL_TRUE,
                    projection_matrix.astype(numpy.float32))
            
            if instances is None:
                instances = self.scene_description['instances'].keys()
            
            # sort the instances
            mesh_instances = {}
            for instance in instances:
                if self.instance_hidden(instance):
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
        mesh_buffers['face_buffer'].bind()
        mesh_buffers['vertex_buffer'].bind()
        
        # get the shader variable locations
        location_data = self.shader_library.get_shader_locations('mask_shader')
        
        GL.glEnableVertexAttribArray(location_data['vertex_position'])
        stride = self.get_mesh_stride(mesh_name)
        GL.glVertexAttribPointer(
                location_data['vertex_position'],
                3, GL.GL_FLOAT, False, stride,
                mesh_buffers['vertex_buffer'])
    
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
            instances=None,
            flip_y=True,
            clear=True,
            finish=True,
    ):
        """
        Renders instances using the coord program.
        
        Parameters:
        -----------
        instances : list, optional
            A list of instances to render.  If not specified, all instances will
            be rendered.
        flip_y : bool, default=True
            Whether or not to flip the image in Y when rendering.  This is to
            correct for the difference between rendering to windows and
            framebuffers.
        clear : bool, default=True
            Whether or not to clear the frame before rendering.
        finish = True
            Whether or not to finish the frame using glFinish
        """
        
        #clear
        if clear:
            self.clear_frame()

        # turn on the shader
        self.shader_library.use_program('coord_shader')

        try:
            location_data = self.shader_library.get_shader_locations(
                    'coord_shader')
            view_matrix = self.scene_description['camera']['view_matrix']
            GL.glUniformMatrix4fv(
                    location_data['view_matrix'],
                    1, GL.GL_TRUE,
                    view_matrix.astype(numpy.float32))

            projection_matrix = self.scene_description['camera']['projection']
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
        if instance_data['hidden']:
            return

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

        mesh_buffers['face_buffer'].bind()
        mesh_buffers['vertex_buffer'].bind()

        try:
            GL.glEnableVertexAttribArray(location_data['vertex_position'])

            stride = self.get_mesh_stride(instance_mesh)
            GL.glVertexAttribPointer(
                    location_data['vertex_position'],
                    3, GL.GL_FLOAT, False, stride,
                    mesh_buffers['vertex_buffer'])

            GL.glDrawElements(
                    GL.GL_TRIANGLES,
                    len(mesh['faces'])*3,
                    GL.GL_UNSIGNED_INT,
                    None)

        finally:
            mesh_buffers['face_buffer'].unbind()
            mesh_buffers['vertex_buffer'].unbind()

    # misc render methods ------------------------------------------------------
    # TODO Figure out what to do about these.
    
    def render_points(self, points, color, point_size = 1, flip_y = True):
        GL.glPushMatrix()
        try:
            projection_matrix = self.scene_description['camera']['projection']
            if flip_y:
                projection_matrix = numpy.dot(projection_matrix, numpy.array([
                        [1, 0, 0, 0],
                        [0,-1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]]))
            GL.glMultMatrixf(numpy.transpose(numpy.dot(
                    projection_matrix,
                    self.scene_description['camera']['view_matrix'])))

            GL.glColor3f(*color)
            GL.glPointSize(point_size)
            GL.glBegin(GL.GL_POINTS)
            for point in points:
                GL.glVertex3f(*point)
            GL.glEnd()
        finally:
            GL.glPopMatrix()
        GL.glFinish()

    def render_line(self, start, end, color, flip_y = True, finish = True):
        GL.glPushMatrix()
        try:
            projection_matrix = self.scene_description['camera']['projection']
            if flip_y:
                projection_matrix = numpy.dot(projection_matrix, numpy.array([
                        [1, 0, 0, 0],
                        [0,-1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]]))
            GL.glMultMatrixf(numpy.transpose(numpy.dot(
                    projection_matrix,
                    self.scene_description['camera']['view_matrix'])))

            GL.glColor3f(*color)
            GL.glBegin(GL.GL_LINES)
            GL.glVertex3f(*start)
            GL.glVertex3f(*end)
            GL.glEnd()
        finally:
            GL.glPopMatrix()
        if finish:
            self.finish_frame()

    def render_transform(self, transform, axis_length = 0.1, flip_y = True):
        GL.glPushMatrix()
        try:
            projection_matrix = self.scene_description['camera']['projection']
            if flip_y:
                projection_matrix = numpy.dot(projection_matrix, numpy.array([
                        [1, 0, 0, 0],
                        [0,-1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]]))
            GL.glMultMatrixf(numpy.transpose(numpy.dot(numpy.dot(
                    projection_matrix,
                    self.scene_description['camera']['view_matrix']),
                    transform)))

            GL.glColor3f(1., 0., 0.)
            GL.glBegin(GL.GL_LINES)
            GL.glVertex3f(0., 0., 0.)
            GL.glVertex3f(axis_length, 0., 0.)
            GL.glEnd()

            GL.glColor3f(0., 1., 0.)
            GL.glBegin(GL.GL_LINES)
            GL.glVertex3f(0., 0., 0.)
            GL.glVertex3f(0., axis_length, 0.)
            GL.glEnd()

            GL.glColor3f(0., 0., 1.)
            GL.glBegin(GL.GL_LINES)
            GL.glVertex3f(0., 0., 0.)
            GL.glVertex3f(0., 0., axis_length)
            GL.glEnd()

            GL.glColor3f(1., 0., 1.)
            GL.glBegin(GL.GL_LINES)
            GL.glVertex3f(0., 0., 0.)
            GL.glVertex3f(-axis_length, 0., 0.)
            GL.glEnd()

            GL.glColor3f(1., 1., 0.)
            GL.glBegin(GL.GL_LINES)
            GL.glVertex3f(0., 0., 0.)
            GL.glVertex3f(0., -axis_length, 0.)
            GL.glEnd()

            GL.glColor3f(0., 1., 1.)
            GL.glBegin(GL.GL_LINES)
            GL.glVertex3f(0., 0., 0.)
            GL.glVertex3f(0., 0., -axis_length)
            GL.glEnd()

        finally:
            GL.glPopMatrix()

    def render_vertices(self, instance_name, flip_y = True):
        #TODO: add this for debugging purposes
        raise NotImplementedError
