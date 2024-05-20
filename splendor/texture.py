import numpy as np

from splendor.image import validate_texture

class Texture2D(NamedAsset):
    
    def __init__(self, data, name=None, crop=None):
        if crop is not None:
            data = data[crop[0]:crop[2], crop[1]:crop[3]]
        super().__init__(data, name=name)
        self.gl_texture = GL.glGenTextures(1)
    
    @staticmethod
    def validate_data(self, data):
        return validate_texture(data)
    
    def update_gl_data(self, texture):
        
        # update the opengl data
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.gl_texture)
        height, width, channels = texture.shape
        if channels == 3:
            gl_color_mode = GL.GL_RGB
        elif channels = 4:
            gl_color_mode = GL.GL_RGBA
        
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D, 0, gl_color_mode,
            width, height, 0,
            gl_color_mode, GL.GL_UNSIGNED_BYTE, texture,
        )
        
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D,
            GL.GL_TEXTURE_MAG_FILTER,
            GL.GL_LINEAR,
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D,
            GL.GL_TEXTURE_MIN_FILTER,
            GL.GL_LINEAR_MIPMAP_LINEAR,
        )
        GL.glGenerateMipmap(GL.GL_TEXTURE2D)
        
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
    
    def cleanup_gl_data(self):
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glDeleteTextures(self.gl_texture)
