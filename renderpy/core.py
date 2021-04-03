# Sources

# https://github.com/BerkeleyAutomation/meshrender
# (I didn't use this because it has too many dependencies
# and no textured meshes)

# https://github.com/rbmj/SimpleRender
# (I didn't use this because it is too simple for my needs)

# system
import math
import json
import os

# opengl
from OpenGL import GL
from OpenGL.arrays import vbo

# numpy
import numpy

# renderpy
from renderpy.assets import AssetLibrary
import renderpy.camera as camera
import renderpy.masks as masks
from renderpy.shader_library import ShaderLibrary
import renderpy.obj_mesh as obj_mesh
from renderpy.image import load_image, load_depth, validate_texture
import renderpy.json_numpy as json_numpy
from renderpy.exceptions import RenderpyException
from renderpy.primitives import make_primitive

max_num_lights = 8
default_default_camera_pose = numpy.eye(4)
default_default_camera_projection = camera.projection_matrix(
        math.radians(90.), 1.0)

class Renderpy:
    """
    Core rendering functionality.
    
    Contains scene data, methods for manipulating it and for performing
    different rendering operations.
    """
    _global_parameters = (
            'ambient_color', 'background_color', 'active_image_light')
    _asset_types = (
            ('mesh', 'meshes'),
            ('material', 'materials'),
            ('image_light', 'image_lights'),
            ('depthmap', 'depthmaps'))
    _instance_types = (
            ('instance', 'instances'),
            ('depthmap_instance', 'depthmap_instances'),
            ('point_light', 'point_lights'))

    def __init__(self,
            assets=None,
            default_camera_pose=None,
            default_camera_projection=None):
        """
        Renderpy initialization
        
        Parameters
        ----------
        assets : str or AssetLibrary, optional
            Either a path pointing to an asset library cfg file or an
            AssetLibrary object.  This is used to load assets such as meshes
            and textures by name rather than their full path.  If not provided,
            this will load the renderpy default asset library.
        default_camera_pose : 4x4 numpy matrix, optional
            The default camera matrix for the renderer.
            Identity if not specified.
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
                'lut':{},
        }

        self.gl_data = {
                'mesh_buffers':{},
                'depthmap_buffers':{},
                'material_buffers':{},
                'light_buffers':{},
                'lut_buffers':{},
                #'textured_shader':{},
                #'vertex_color_shader':{},
                #'mask_shader':{},
                #'coord_shader':{},
                #'background_shader':{}
        }
        
        self.load_brdf_lut('default_brdf_lut')
        
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

        # meshes, depthmaps, materials, image_lights
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
            if 'pose' in scene['camera']:
                self.set_camera_pose(scene['camera']['pose'])
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

    def set_background_color(self, background_color):
        if len(background_color) == 3:
            background_color = tuple(background_color) + (1,)
        self.scene_description['background_color'] = background_color

    def set_active_image_light(self, image_light):
        self.scene_description['active_image_light'] = image_light

    # camera methods ===========================================================
    
    def reset_camera(self):
        """
        Resets the camera to the default pose and projection.
        """
        self.set_camera_pose(self.default_camera_pose)
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

    def set_camera_pose(self, camera_pose):
        """
        Sets the camera matrix.
        
        Parameters:
        -----------
        camera_matrix : 4x4 array-like, 6-element or 9-element azimuthal pose
            Azimuthal pose is [azimuth, elevation, tilt, distance, x, y]
        """
        camera_matrix = camera.camera_pose_to_matrix(camera_pose)
        self.scene_description['camera']['pose'] = numpy.array(camera_matrix)

    def get_camera_pose(self):
        """
        Get the camera matrix.
        
        Note this is the inverse of the SE3 pose of the camera object.
        
        Returns:
        --------
        camera_matrix : 4x4 numpy array
        """
        return self.scene_description['camera']['pose']

    def camera_frame_scene(self, multiplier=3.0, *args, **kwargs):
        bbox = self.get_instance_center_bbox()
        camera_matrix = camera.frame_bbox(
                bbox, self.get_projection(), multiplier,
                *args, **kwargs)
        self.set_camera_pose(camera_matrix)

    # mesh methods =============================================================
    
    def load_mesh(self,
            name,
            mesh_asset = None,
            mesh_path = None,
            mesh_data = None,
            mesh_primitive = None,
            scale = 1.0,
            #create_uvs = False,
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
            mesh_path = self.asset_library['meshes'][mesh_asset]
            mesh = obj_mesh.load_mesh(mesh_path, scale=scale)
            self.scene_description['meshes'][name] = {
                'mesh_asset':mesh_asset
            }

        # otherwise, load name as an asset path
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
            raise RenderpyException(
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
        color_mode : {"textured", "vertex_color", "flat"}
            Describes the color mode of the surface.  Can be one of:
            "textured" : requires uvs
            "vertex_color" : requires specified vertex colors
            "flat" : the entire surface will be a single flat color
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
            diffuse_texture,
            reflect_texture,
            reflect_mipmaps = None,
            offset_matrix = numpy.eye(4),
            blur = 0.,
            gamma = 1.,
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
        """
        Load an image light.
        
        Loads an image light into memory but does not make it active unless
        set_active=True.  Only one image_light can be active at a time.
        
        Parameters:
        -----------
        name : str
            Name of the image light, must be unique to this scene among other
            image lights
        diffuse_texture : str
            Either an asset name or a path to a diffuse texture strip.  The
            height of the texture strip must be a power of 2 and the width must
            be six times the height, with each sequential square representing
            the px, nx, py, ny, pz, nz face of a cubemap.
        reflect_texture : str
            See diffuse_texture, but for the reflection map.
        TODO: Come back and document the rest of this when we clean up the
            image light parameters
        """
        
        if name in self.gl_data['light_buffers']:
            GL.glDeleteTextures([
                    self.gl_data['light_buffers'][name]['diffuse_texture'],
                    self.gl_data['light_buffers'][name]['reflect_texture']])

        light_buffers = {}
        light_buffers['diffuse_texture'] = GL.glGenTextures(1)
        light_buffers['reflect_texture'] = GL.glGenTextures(1)
        self.gl_data['light_buffers'][name] = light_buffers

        image_light_data = {}
        image_light_data['offset_matrix'] = numpy.array(offset_matrix)
        image_light_data['blur'] = blur
        image_light_data['render_background'] = render_background
        image_light_data['gamma'] = gamma
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
                diffuse_texture,
                reflect_texture,
                reflect_mipmaps)

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
        GL.glDeleteTextures(
                self.gl_data['light_buffers'][name]['diffuse_texture'])
        GL.glDeleteTextures(
                self.gl_data['light_buffers'][name]['reflect_texture'])
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
    
    def replace_texture(self,
            name,
            texture,
            crop = None):
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

        if isinstance(texture, str):
            self.scene_description['materials'][name]['texture'] = texture
            texture = self.asset_library['textures'][texture]
            image = load_image(texture)
        else:
            self.scene_description['materials'][name]['texture'] = -1
            image = numpy.array(texture)

        if crop is not None:
            image = image[crop[0]:crop[2], crop[1]:crop[3]]

        validate_texture(image)
        self.loaded_data['textures'][name] = image

        material_buffers = self.gl_data['material_buffers'][name]
        GL.glBindTexture(GL.GL_TEXTURE_2D, material_buffers['texture'])
        try:
            GL.glTexImage2D(
                    GL.GL_TEXTURE_2D, 0, GL.GL_RGB,
                    image.shape[1], image.shape[0], 0,
                    GL.GL_RGB, GL.GL_UNSIGNED_BYTE, image)

            # GL.GL_NEAREST?
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
    
    def replace_image_light_textures(self,
            name,
            diffuse_texture,
            reflect_texture,
            reflect_mipmaps = None):
        """
        Replace the textures for an image light
        
        Parameters:
        -----------
        name : str
            The name of the material to replace the texture
        diffuse_texture : array-like or str
            Either an asset, path or raw image data for the diffuse texture
        reflect_texture : array-like or str
            Either an asset, path or raw image data for the reflect texture
        TODO: update this once we figure out what we're doing with mipmaps
        """
        
        light_description = self.scene_description['image_lights'][name]
        
        if isinstance(diffuse_texture, str):
            diffuse_texture = self.asset_library['image_lights'][
                    diffuse_texture]
            light_description['diffuse_texture'] = diffuse_texture
            diffuse_image = load_image(diffuse_texture)
        else:
            light_description['diffuse_texture'] = -1
            diffuse_image = diffuse_texture

        if isinstance(reflect_texture, str):
            reflect_texture = self.asset_library['image_lights'][
                    reflect_texture]
            light_description['reflect_texture'] = reflect_texture
            reflect_image = load_image(reflect_texture)
        else:
            light_description['reflect_texture'] = -1
            reflect_image = reflect_texture

        if reflect_mipmaps:
            if isinstance(reflect_mipmaps[0][0], str):
                light_description['reflect_mipmaps'] = reflect_mipmaps
                reflect_mipmaps = [
                        [load_image(mipmap) for mipmap in mipmaps]
                        for mipmaps in reflect_mipmaps]
            else:
                light_description['reflect_mipmaps'] = -1
        else:
            light_description['reflect_mipmaps'] = -1
        
        light_buffers = self.gl_data['light_buffers'][name]
        GL.glBindTexture(
                GL.GL_TEXTURE_CUBE_MAP,
                light_buffers['diffuse_texture'],
        )
        try:
            diffuse_min = float('inf')
            diffuse_max = -float('inf')
            diffuse_image = numpy.array(diffuse_image)
            height, strip_width = diffuse_image.shape[:2]
            assert strip_width == height * 6
            for i in range(6):
                face_image = diffuse_image[:,i*height:(i+1)*height]
                validate_texture(face_image)
                GL.glTexImage2D(
                        GL.GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                        0, GL.GL_RGB,
                        face_image.shape[1], face_image.shape[0],
                        0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, face_image)

                diffuse_intensity = (
                        diffuse_image[:,:,0] * 0.2989 +
                        diffuse_image[:,:,1] * 0.5870 +
                        diffuse_image[:,:,2] * 0.1140)

                diffuse_min = min(diffuse_min, numpy.min(diffuse_intensity))
                diffuse_max = max(diffuse_max, numpy.max(diffuse_intensity))

            light_description['diffuse_min'] = diffuse_min / 255.
            light_description['diffuse_max'] = diffuse_max / 255.

            GL.glTexParameteri(
                    GL.GL_TEXTURE_CUBE_MAP,
                    GL.GL_TEXTURE_MAG_FILTER,
                    GL.GL_LINEAR,
            )
            GL.glTexParameteri(
                    GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_MIN_FILTER,
                    GL.GL_LINEAR_MIPMAP_LINEAR)
            GL.glGenerateMipmap(GL.GL_TEXTURE_CUBE_MAP)
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
            GL.glTexParameteri(
                    GL.GL_TEXTURE_CUBE_MAP,
                    GL.GL_TEXTURE_WRAP_R,
                    GL.GL_CLAMP_TO_EDGE,
            )
        finally:
            GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, 0)

        GL.glBindTexture(
                GL.GL_TEXTURE_CUBE_MAP, light_buffers['reflect_texture'])
        try:
            reflect_image = numpy.array(reflect_image)
            height, strip_width = reflect_image.shape[:2]
            assert strip_width == height * 6
            for i in range(6):
                face_image = reflect_image[:,i*height:(i+1)*height]
                validate_texture(face_image)
                GL.glTexImage2D(
                        GL.GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                        0, GL.GL_RGB,
                        face_image.shape[1], face_image.shape[0],
                        0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, face_image)
                if reflect_mipmaps is not None:
                    for j, mipmap in enumerate(reflect_mipmaps[i]):
                        mipmap = numpy.array(mipmap)
                        validate_texture(mipmap)
                        GL.glTexImage2D(
                                GL.GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                                j+1, GL.GL_RGB,
                                mipmap.shape[1], mipmap.shape[0],
                                0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, mipmap)
            
            GL.glTexParameteri(
                    GL.GL_TEXTURE_CUBE_MAP,
                    GL.GL_TEXTURE_MAG_FILTER,
                    GL.GL_LINEAR,
            )
            GL.glTexParameteri(
                    GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_MIN_FILTER,
                    GL.GL_LINEAR_MIPMAP_LINEAR)
            if reflect_mipmaps is None:
                GL.glGenerateMipmap(GL.GL_TEXTURE_CUBE_MAP)
            else:
                GL.glTexParameteri(
                        GL.GL_TEXTURE_CUBE_MAP,
                        GL.GL_TEXTURE_MAX_LEVEL,
                        len(reflect_mipmaps[0]))
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
            GL.glTexParameteri(
                    GL.GL_TEXTURE_CUBE_MAP,
                    GL.GL_TEXTURE_WRAP_R,
                    GL.GL_CLAMP_TO_EDGE,
            )
        finally:
            GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, 0)

        self.loaded_data['textures'][name + '_diffuse'] = diffuse_image
        self.loaded_data['textures'][name + '_reflect'] = reflect_image
    
    def load_brdf_lut(self, texture):
        if isinstance(texture, str):
            texture = self.asset_library['textures'][texture]
            image = load_image(texture)
        else:
            image = numpy.array(texture)
        
        if 'BRDF' in self.gl_data['lut_buffers']:
            GL.glBindTexture(GL.GL_TEXTURE_2D,0)
            GL.glDeleteTextures(
                    [self.gl_data['lut_buffers']['BRDF']['texture']])
        
        lut_buffers = {}
        lut_buffers['texture'] = GL.glGenTextures(1)
        self.gl_data['lut_buffers']['BRDF'] = lut_buffers
        
        validate_texture(image)
        self.loaded_data['lut'] = {}
        self.loaded_data['lut']['BRDF'] = image
        GL.glBindTexture(GL.GL_TEXTURE_2D, lut_buffers['texture'])
        try:
            GL.glTexImage2D(
                    GL.GL_TEXTURE_2D, 0, GL.GL_RGB,
                    image.shape[1], image.shape[0], 0,
                    GL.GL_RGB, GL.GL_UNSIGNED_BYTE, image)
            
            GL.glTexParameteri(
                    GL.GL_TEXTURE_2D,
                    GL.GL_TEXTURE_WRAP_S,
                    GL.GL_CLAMP_TO_EDGE,
            )
            GL.glTexParameteri(
                    GL.GL_TEXTURE_2D,
                    GL.GL_TEXTURE_WRAP_T,
                    GL.GL_CLAMP_TO_EDGE,
            )
            
            GL.glTexParameteri(
                    GL.GL_TEXTURE_2D,
                    GL.GL_TEXTURE_MIN_FILTER,
                    GL.GL_LINEAR,
            )
            GL.glTexParameteri(
                    GL.GL_TEXTURE_2D,
                    GL.GL_TEXTURE_MAG_FILTER,
                    GL.GL_LINEAR,
            )
            
        finally:
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
    
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
        return self.loaded_data['textures'][texture_name]

    # material methods =========================================================
    
    def load_material(self,
            name,
            texture = None,
            flat_color = None,
            ambient = 1.0,
            #kd = 1.0,
            #ks = 0.5,
            metal = 0.15,
            #shine = 1.0,
            rough = 2.0,
            reflect_color = (0.04, 0.04, 0.04),
            #image_light_kd = 0.85,
            #image_light_ks = 0.15,
            #image_light_blur_reflection = 2.0,
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
        TODO: Fill in the rest of this once material parameters stabilize
        """

        self.scene_description['materials'][name] = {
                #'ka' : ka,
                #'kd' : kd,
                #'ks' : ks,
                #'shine' : shine,
                #'image_light_kd' : image_light_kd,
                #'image_light_ks' : image_light_ks,
                #'image_light_blur_reflection' : image_light_blur_reflection,
                'ambient':ambient,
                'metal':metal,
                'rough':rough,
                'flat_color':flat_color,
                'reflect_color':reflect_color}

        if name in self.gl_data['material_buffers']:
            GL.glBindTexture(GL.GL_TEXTURE_2D,0)
            GL.glDeleteTextures(
                    [self.gl_data['material_buffers'][name]['texture']])

        material_buffers = {}
        material_buffers['texture'] = GL.glGenTextures(1)
        self.gl_data['material_buffers'][name] = material_buffers
        
        if texture is not None:
            self.replace_texture(name, texture, crop)

    def remove_material(self, name):
        """
        Deletes a material from the scene.
        
        Parameters:
        -----------
        name : str
        """
        GL.glDeleteTextures(self.gl_data['material_buffers'][name]['texture'])
        if name in self.loaded_data['textures']:
            del(self.loaded_data['textures'][name])
        del(self.gl_data['material_buffers'][name])
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

    def hide_instance(self, name):
        """
        Hides an instance so that it will not render in any render modes
        
        Parameters:
        -----------
        name : str
        """
        self.scene_description['instances'][name]['hidden'] = True

    def show_instance(self, name):
        """
        Makes an instance visible in all render modes
        
        Parameters:
        -----------
        name : str
        """
        self.scene_description['instances'][name]['hidden'] = False

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
            material_name,
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
        depthmap_instance_data['material_name'] = material_name
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
            image_light_description = (
                    self.scene_description['image_lights'][image_light_name])
            if image_light_description['render_background']:
                self.render_background(image_light_name, flip_y = flip_y)

        # set image light maps
        if image_light_name is not None:
            GL.glActiveTexture(GL.GL_TEXTURE1)
            GL.glBindTexture(
                    GL.GL_TEXTURE_2D,
                    self.gl_data['lut_buffers']['BRDF']['texture'])
            GL.glActiveTexture(GL.GL_TEXTURE2)
            GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, self.gl_data[
                    'light_buffers'][image_light_name]['diffuse_texture'])
            GL.glActiveTexture(GL.GL_TEXTURE3)
            GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, self.gl_data[
                    'light_buffers'][image_light_name]['reflect_texture'])

        # depthmap_instances
        if depthmap_instances is None:
            depthmap_instances = self.scene_description['depthmap_instances']

        self.shader_library.use_program('textured_depthmap_shader')
        try:
            location_data = self.shader_library.get_shader_locations(
                    'textured_depthmap_shader')
            
            # set the camera's pose
            camera_pose = self.scene_description['camera']['pose']
            GL.glUniformMatrix4fv(
                    location_data['camera_matrix'],
                    1, GL.GL_TRUE,
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

        textured_shader_instances = []
        vertex_color_shader_instances = []
        flat_color_shader_instances = []
        for instance in instances:
            instance_mesh = self.get_instance_mesh_name(instance)
            mesh_color_mode = self.get_mesh_color_mode(instance_mesh)
            if mesh_color_mode == 'textured':
                textured_shader_instances.append(instance)
            elif mesh_color_mode == 'vertex_color':
                vertex_color_shader_instances.append(instance)
            elif mesh_color_mode == 'flat_color':
                flat_color_shader_instances.append(instance)

        for shader_name, shader_instances in (
                ('textured_shader', textured_shader_instances),
                ('vertex_color_shader', vertex_color_shader_instances),
                ('flat_color_shader', flat_color_shader_instances)):

            if len(shader_instances) == 0:
                continue

            # turn on the shader
            self.shader_library.use_program(shader_name)

            try:
                location_data = self.shader_library.get_shader_locations(
                        shader_name)

                # set the camera's pose
                camera_pose = self.scene_description['camera']['pose']
                GL.glUniformMatrix4fv(
                        location_data['camera_matrix'],
                        1, GL.GL_TRUE,
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
                
                '''
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
                '''
                # set the image light parameters
                if image_light_name is not None:
                    image_light_data = self.get_image_light(image_light_name)
                    
                    # set the offset matrix
                    offset_matrix = image_light_data['offset_matrix']
                    GL.glUniformMatrix4fv(
                            location_data['image_light_offset_matrix'],
                            1, GL.GL_TRUE,
                            offset_matrix.astype(numpy.float32))
                    
                    '''
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
                    GL.glUniform2fv(
                            location_data['image_light_diffuse_minmax'],
                            1, diffuse_minmax)
                    GL.glUniform3fv(
                            location_data['image_light_diffuse_rescale'],
                            1, diffuse_rescale)
                    GL.glUniform3fv(
                            location_data['image_light_diffuse_tint_lo'],
                            1, diffuse_tint_lo)
                    GL.glUniform3fv(
                            location_data['image_light_diffuse_tint_hi'],
                            1, diffuse_tint_hi)
                    GL.glUniform3fv(
                            location_data['image_light_reflect_tint'],
                            1, reflect_tint)
                    '''

                mesh_instances = {}
                for instance_name in shader_instances:
                    if self.instance_hidden(instance_name):
                        continue
                    mesh_name = self.get_instance_mesh_name(instance_name)
                    try:
                        mesh_instances[mesh_name].append(instance_name)
                    except KeyError:
                        mesh_instances[mesh_name] = [instance_name]

                for mesh_name, instance_names in mesh_instances.items():
                    self.color_render_instance(instance_names[0], shader_name)
                    for instance_name in instance_names[1:]:
                        self.color_render_instance(
                                instance_name,
                                shader_name,
                                set_mesh_attrib_pointers=False)

            finally:
                GL.glUseProgram(0)

        if finish:
            self.finish_frame()

    def color_render_instance(self,
            instance_name,
            shader_name,
            set_mesh_attrib_pointers=True):
        """
        Render a single instance using the color program.
        
        Parameters:
        -----------
        instance_name : str
            The instance to render
        shader_name : str
            {"textured_shader", "vertex_color_shader", "flat_color_shader"}
            The specific version of the color program to use.
        set_mesh_attrib_pointers : bool, default=True
            An optimization.  If the same mesh was rendered previously, there
            is no need to copy certain data to the GPU again.
        """
        
        instance_data = self.scene_description['instances'][instance_name]
        if instance_data['hidden']:
            return

        instance_mesh = instance_data['mesh_name']
        instance_material = instance_data['material_name']
        material_data = (
                self.scene_description['materials'][instance_material])
        image_light_name = self.scene_description['active_image_light']
        if image_light_name is not None:
            gamma = self.get_image_light(image_light_name)['gamma']
        else:
            gamma = 1.0
        '''
        material_properties = numpy.array([
                material_data['ka'],
                material_data['kd'],
                material_data['ks'],
                material_data['shine']])
        image_light_material_properties = numpy.array([
                material_data['image_light_kd'],
                material_data['image_light_ks'],
                material_data['image_light_blur_reflection']])
        '''
        material_properties = numpy.array([
                material_data['ambient'],
                material_data['metal'],
                material_data['rough'],
                gamma])
        mesh_buffers = self.gl_data['mesh_buffers'][instance_mesh]
        material_buffers = self.gl_data['material_buffers'][instance_material]
        mesh_data = self.loaded_data['meshes'][instance_mesh]
        num_triangles = len(mesh_data['faces'])

        location_data = self.shader_library.get_shader_locations(shader_name)

        GL.glUniformMatrix4fv(
                location_data['model_pose'],
                1, GL.GL_TRUE,
                instance_data['transform'].astype(numpy.float32))
        
        '''
        GL.glUniform4fv(
                location_data['material_properties'],
                1, material_properties.astype(numpy.float32))
        GL.glUniform3fv(
                location_data['image_light_material_properties'], 1,
                image_light_material_properties)
        '''
        GL.glUniform4fv(
                location_data['material_properties'],
                1, material_properties.astype(numpy.float32))
        reflect_color = numpy.array(
                material_data['reflect_color']).astype(numpy.float32)
        GL.glUniform3fv(
                location_data['reflect_color'],
                1, reflect_color)
        
        mesh_buffers['face_buffer'].bind()
        mesh_buffers['vertex_buffer'].bind()
        
        if shader_name == 'textured_shader':
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, material_buffers['texture'])
        
        if shader_name == 'flat_color_shader':
            flat_color = self.get_material_flat_color(instance_material)
            GL.glUniform3fv(
                    location_data['flat_color'],
                    1, numpy.array(flat_color, dtype=numpy.float32))
        
        try:
            # SOMETHING BETWEEN HERE...
            if set_mesh_attrib_pointers:
                GL.glEnableVertexAttribArray(location_data['vertex_position'])
                GL.glEnableVertexAttribArray(location_data['vertex_normal'])
                if shader_name == 'textured_shader':
                    GL.glEnableVertexAttribArray(location_data['vertex_uv'])
                elif shader_name == 'vertex_color_shader':
                    GL.glEnableVertexAttribArray(location_data['vertex_color'])

                stride = self.get_mesh_stride(instance_mesh)
                GL.glVertexAttribPointer(
                        location_data['vertex_position'],
                        3, GL.GL_FLOAT, False, stride,
                        mesh_buffers['vertex_buffer'])
                GL.glVertexAttribPointer(
                        location_data['vertex_normal'],
                        3, GL.GL_FLOAT, False, stride,
                        mesh_buffers['vertex_buffer']+((3)*4))
                if shader_name == 'textured_shader':
                    GL.glVertexAttribPointer(
                            location_data['vertex_uv'],
                            2, GL.GL_FLOAT, False, stride,
                            mesh_buffers['vertex_buffer']+((3+3)*4))
                elif shader_name == 'vertex_color_shader':
                    GL.glVertexAttribPointer(
                            location_data['vertex_color'],
                            3, GL.GL_FLOAT, False, stride,
                            mesh_buffers['vertex_buffer']+((3+3)*4))

            # AND HERE TAKES ~40% of the rendering time

            GL.glDrawElements(
                    GL.GL_TRIANGLES,
                    num_triangles*3,
                    GL.GL_UNSIGNED_INT,
                    None)

        finally:
            mesh_buffers['face_buffer'].unbind()
            mesh_buffers['vertex_buffer'].unbind()
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

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
        instance_material = depthmap_instance_data['material_name']
        material_data = (
                self.scene_description['materials'][instance_material])
        depthmap_buffers = self.gl_data['depthmap_buffers'][instance_depthmap]
        material_buffers = self.gl_data['material_buffers'][instance_material]
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
        GL.glBindTexture(GL.GL_TEXTURE_2D, material_buffers['texture'])
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
        light_buffers = self.gl_data['light_buffers'][image_light_name]
        num_triangles = 2

        location_data = self.shader_library.get_shader_locations(
                'background_shader')

        # set the camera's pose
        camera_pose = self.scene_description['camera']['pose']
        GL.glUniformMatrix4fv(
                location_data['camera_matrix'],
                1, GL.GL_TRUE,
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
        GL.glUniformMatrix4fv(
                location_data['projection_matrix'],
                1, GL.GL_TRUE,
                projection_matrix.astype(numpy.float32))

        light_data = self.scene_description['image_lights'][image_light_name]

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
                light_buffers['reflect_texture'])
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

            # set the camera's pose
            camera_pose = self.scene_description['camera']['pose']
            GL.glUniformMatrix4fv(
                    location_data['camera_pose'],
                    1, GL.GL_TRUE,
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
            GL.glUniformMatrix4fv(
                    location_data['projection_matrix'],
                    1, GL.GL_TRUE,
                    projection_matrix.astype(numpy.float32))

            # render all instances
            if instances is None:
                instances = self.scene_description['instances']
            for instance_name in instances:
                self.mask_render_instance(instance_name)

        finally:
            GL.glUseProgram(0)
        
        if finish:
            GL.glFinish()

    def mask_render_instance(self, instance_name):
        """
        Render a single instance using the mask program.
        
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
        mask_color = instance_data['mask_color']
        mesh_buffers = self.gl_data['mesh_buffers'][instance_mesh]
        mesh = self.loaded_data['meshes'][instance_mesh]

        location_data = self.shader_library.get_shader_locations(
                'mask_shader')

        GL.glUniformMatrix4fv(
                location_data['model_pose'],
                1, GL.GL_TRUE,
                numpy.array(instance_data['transform'], dtype=numpy.float32))

        GL.glUniform3fv(
                location_data['mask_color'],
                1, numpy.array(mask_color, dtype=numpy.float32))

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
            camera_pose = self.scene_description['camera']['pose']
            GL.glUniformMatrix4fv(
                    location_data['camera_pose'],
                    1, GL.GL_TRUE,
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
                    self.scene_description['camera']['pose'])))

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
                    self.scene_description['camera']['pose'])))

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
                    self.scene_description['camera']['pose']),
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
