import math

import numpy as np

import OpenGL.GL as GL

from splendor.session import active_render_session
from splendor.named_asset import NamedAsset
from splendor.instance import InstanceTypeList

class Camera(NamedAsset):
    
    _loaded_assets = {}
    
    def __init__(self,
        name=None,
        transform=None,
        view_matrix=None,
        width=256,
        height=256,
        render_color=True,
        render_mask=False,
        render_coord=False,
        render_depth=False,
        include_alpha=False,
        anti_alias=False,
        anti_alias_samples=8,
        radial_distortion_k1=0.,
        radial_distortion_k2=0.,
        #color_format='RGBA8',
    ):
        # register this camera name
        super().__init__(name=name)
        
        # set the camera transform
        assert transform is None or view_matrix is None
        if transform is not None:
            self.transform = transform
        elif view_matrix is not None:
            self.view_matrix = view_matrix
        else:
            self.transform = np.eye(4)
        
        self.width = width
        self.height = height
        self.render_color = render_color
        self.render_mask = render_mask
        self.render_coord = render_coord
        self.render_depth = render_depth
        self.include_alpha = include_alpha
        self.anti_alias = anti_alias
        self.anti_alias_samples = anti_alias_samples
        
        self.radial_distortion_k1 = radial_distortion_k1
        self.radial_distortion_k2 = radial_distortion_k2
        
        #assert color_format in ('RGBA8', 'RGBA32F')
        #self.color_format = color_format
        
        self.initialize_framebuffers()
    
    @property
    def transform(self):
        return self._transform
    
    @transform.setter
    def transform(self, transform):
        assert np.shape(transform) == (4,4)
        self._transform = np.array(transform, dtype=np.float32)
        self._view_matrix = np.linalg.inv(self._transform)
    
    @property
    def view_matrix(self):
        return self._view_matrix
    
    @view_matrix.setter
    def view_matrix(self, view_matrix):
        assert np.shape(transform) == (4,4)
        self._view_matrix = np.array(view_matrix, dtype=np.float32)
        self._transform = np.linalg.inv(self._view_matrix)
    
    @property
    def requires_lens_pass(self):
        return (
            self.radial_distortion_k1 != 0. or
            self.radial_distortion_k2 != 0.
        )
    
    def initialize_framebuffers(self):
        
        # setup the color texture
        if self.render_color:
            self.color_framebuffer = GL.glGenFramebuffers(1)
            self.color_texture = GL.glGenTextures(1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.color_texture)
            if self.include_alpha:
                color_format = GL.GL_RGBA
                zeros = np.zeros((self.height, self.width, 4), dtype=np.uint8)
            else:
                color_format = GL.GL_RGB
                zeros = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
            GL.glTexImage2D(
                GL.GL_TEXTURE_2D,
                0,
                color_format,
                self.width,
                self.height,
                0,
                color_format,
                GL.GL_UNSIGNED_BYTE,
                zeros,
            )
            GL.glTexParameteri(
                GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(
                GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(
                GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
            GL.glTexParameteri(
                GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
            
            # attach the color texture to the color framebuffer
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.color_framebuffer)
            GL.glFramebufferTexture2D(
                GL.GL_FRAMEBUFFER,
                GL.GL_COLOR_ATTACHMENT0,
                GL.GL_TEXTURE_2D,
                self.color_texture,
                0,
            )
            
            # setup the depth texture
            self.depth_texture = GL.glGenTextures(1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.depth_texture)
            GL.glTexImage2D(
                GL.GL_TEXTURE_2D,
                0,
                GL.GL_DEPTH_COMPONENT,
                self.width,
                self.height,
                0,
                GL.GL_DEPTH_COMPONENT,
                GL.GL_FLOAT,
                0,
            )
            GL.glTexParameteri(
                GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
            GL.glTexParameteri(
                GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
            GL.glTexParameteri(
                GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
            GL.glTexParameteri(
                GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
            
            # attach the depth texture to the color framebuffer
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.color_framebuffer)
            GL.glFramebufferTexture2D(
                GL.GL_FRAMEBUFFER,
                GL.GL_DEPTH_ATTACHMENT,
                GL.GL_TEXTURE_2D,
                self.depth_texture,
                0,
            )
    
    @property
    def width(self):
        return self._width
    
    @width.setter
    def width(self, width):
        self._framebuffer_dirty = True
        self._width = width
    
    @property
    def height(self):
        return self._height
    
    @height.setter
    def height(self, height):
        self._framebuffer_dirty = True
        self._height = height
    
    @property
    def aspect_ratio(self):
        return self.width / self.height
    
    @property
    def anti_alias(self):
        return self._anti_alias
    
    @anti_alias.setter
    def anti_alias(self, anti_alias):
        self._framebuffer_dirty = True
        self._anti_alias = anti_alias
    
    @property
    def projection_matrix(self):
        raise NotImplementedError
    
    @staticmethod
    def deserialize(name, camera_type, **kwargs):
        if camera_type == 'pinhole':
            return PinholeCamera(name, **kwargs)
        elif camera_type == 'orthographic':
            return OrthographicCaemra(name, **kwargs)
        elif camera_type == 'intrinsics':
            return IntrinsicsCamera(name, **kwargs)
        elif caemra_type == 'projection_matrix':
            return ProjectionMatrixCamera(name, **kwargs)
    
    def render(self,
        instances=None,
        lights=None,
        background_color=(0,0,0,0),
    ):
        images = []
        if self.render_color:
            color_image = self._render_color(
                instances=instances,
                lights=lights,
                background_color=background_color,
            )
            images.append(color_image)
        
       # if self.depth_render:
       #     # if we have already rendered another image, simply pull the depth
       #     # data off the existing framebuffer
       #     if self.render_color:
       #         pass
       #     else:
       #         pass
        
        return images
            
    def _render_color(self,
        instances=None,
        lights=None,
        background_color=(0,0,0,0),
    ):
        # get the shader library
        session = active_render_session()
        shader_library = session.shader_library
        
        # bind the framebuffer
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.color_framebuffer)
        
        # TODO: Multisampling
        GL.glDisable(GL.GL_MULTISAMPLE)
        
        # viewport/scissor
        GL.glViewport(0, 0, self.width, self.height)
        GL.glScissor(0, 0, self.width, self.height)
        
        # get the background color
        if len(background_color) == 3:
            background_color = tuple(*background_color, 0)
        GL.glClearColor(*background_color)
        
        # clear
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        
        # FOR EACH INSTANCE TYPE
        if not isinstance(instances, InstanceTypeList):
            instances = InstanceTypeList(instances)
        
        for instance_type_list in instances:
            
            if not len(instance_type_list):
                continue
            
            # get the shader name and variable locations
            shader_name = instance_type_list[0].shader_name
            shader_library.use_program(shader_name)
            shader_locations = shader_library.get_shader_locations(shader_name)
            
            # set the view and projection matrices
            GL.glUniformMatrix4fv(
                shader_locations['view_matrix'],
                1,
                GL.GL_TRUE,
                self.view_matrix,
            )
            GL.glUniformMatrix4fv(
                shader_locations['projection_matrix'],
                1,
                GL.GL_TRUE,
                self.projection_matrix,
            )
            
            # set the lights (move elsewhere)
            ambient_color = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            GL.glUniform3fv(
                shader_locations['ambient_color'], 1, ambient_color)
            #GL.glUniform1i(shader_locations['num_point_lights'], 0)
            #GL.glUniform1i(shader_locations['num_direction_lights'], 0)
            #GL.glUniform1i(shader_locations['image_light_active'], 0)
            GL.glUniform3fv(
                shader_locations['background_color'], 1, background_color[:3])
            
            # render each instance
            for instance in instance_type_list:
                instance.render(shader_locations)
                
                GL.glPushMatrix()
                GL.glMultMatrixf(np.transpose(np.dot(
                    self.projection_matrix,
                    self.view_matrix)))
                GL.glColor3f(1,0,0)
                GL.glBegin(GL.GL_LINES)
                GL.glVertex3f(0,0,1)
                GL.glVertex3f(1,1,1)
                GL.glEnd()
                GL.glPopMatrix()
            
        # finish rendering
        GL.glFinish()
        
        # read the pixels
        if self.include_alpha:
            channel_format = GL.GL_RGBA
            num_channels = 4
        else:
            channel_format = GL.GL_RGB
            num_channels = 3
        
        if True:
            gl_dtype = GL.GL_UNSIGNED_BYTE
            np_dtype = np.uint8
        else:
            gl_dtype = GL.GL_FLOAT
            np_dtype = np.float32
        
        
        pixels = GL.glReadPixels(
            0, 0, self.width, self.height, channel_format, gl_dtype)
        image = np.frombuffer(pixels, dtype=np_dtype).reshape(
            self.height, self.width, num_channels)
        
        return image

class PinholeCamera(Camera):
    def __init__(self,
        name=None,
        transform=None,
        view_matrix=None,
        width=256,
        height=256,
        fov=math.radians(90.),
        near_clip=0.05,
        far_clip=5000,
        render_color=True,
        render_depth=False,
        render_mask=False,
        render_coord=False,
        include_alpha=False,
        anti_alias=False,
        anti_alias_samples=8,
        radial_distortion_k1=0.,
        radial_distortion_k2=0.,
    ):
        super().__init__(
            name=name,
            transform=transform,
            view_matrix=view_matrix,
            width=width,
            height=height,
            render_color=render_color,
            render_depth=render_depth,
            render_mask=render_mask,
            render_coord=render_coord,
            include_alpha=include_alpha,
            anti_alias=anti_alias,
            anti_alias_samples=anti_alias_samples,
            radial_distortion_k1=radial_distortion_k1,
            radial_distortion_k2=radial_distortion_k2,
        )
        
        self.fov = fov
        self.width = width
        self.height = height
        self.near_clip = near_clip
        self.far_clip = far_clip
    
    
    @property
    def fov(self):
        return self._fov
    
    @fov.setter
    def fov(self, fov):
        self._projection_matrix_dirty = True
        self._fov = fov
    
    @property
    def width(self):
        return super().width
    
    @Camera.width.setter
    def width(self, width):
        self._framebuffer_dirty = True
        self._projection_matrix_dirty = True
        self._width = width
    
    @property
    def height(self):
        return super().height
    
    @height.setter
    def height(self, height):
        self._framebuffer_dirty = True
        self._projection_matrix_dirty = True
        self._height = height
    
    @property
    def near_clip(self):
        return self._near_clip
    
    @near_clip.setter
    def near_clip(self, near_clip):
        self._projection_matrix_dirty = True
        self._near_clip = near_clip
    
    @property
    def far_clip(self):
        return self._far_clip
    
    @far_clip.setter
    def far_clip(self, far_clip):
        self._projection_matrix_dirty = True
        self._far_clip = far_clip
    
    @property
    def projection_matrix(self):
        if self._projection_matrix_dirty:
            x_limit = self.near_clip * math.tan(self.fov * 0.5)
            y_limit = x_limit / self.aspect_ratio
            
            nc = self.near_clip
            fc = self.far_clip
            self._projection_matrix = np.array([
                [nc/x_limit, 0, 0, 0],
                [0, nc/y_limit, 0, 0],
                [0, 0, -(fc + nc) / (fc - nc), -2 * fc * nc / (fc - nc)],
                [0, 0, -1, 0]
            ])
            self._projection_matrix.flags.writeable = False
            self._projection_matrix_dirty = False
        
        return self._projection_matrix

class OrthographicCamera(Camera):
    pass

class IntrinsicsCamera(Camera):
    pass

class ProjectionMatrixCamera(Camera):
    pass
