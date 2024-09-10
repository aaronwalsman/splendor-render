import numpy as np

import OpenGL.GL as GL

from splendor.named_asset import NamedAsset
from splendor.image import load_image, validate_texture

class Texture2D(NamedAsset):
    
    _loaded_assets = {}
    
    def __init__(self,
        name=None,
        asset=None,
        path=None,
        texture=None,
        crop=None,
    ):
        # if no name was specified, but an asset or path was provided, change
        # the default name
        if name is None:
            if asset is not None:
                name, _ = os.path.splitext(os.path.basename(asset))
            elif path is not None:
                name, _ = os.path.splitext(os.path.basename(path))
        
        # register a unique name for this texture
        super().__init__(name=name)
        
        # initialize the texture
        self.gl_texture = GL.glGenTextures(1)
        
        # load and validate the texture
        self.reload(asset=asset, path=path, texture=texture, crop=crop)
    
    def reload(self,
        asset=None,
        path=None,
        texture=None,
        crop=None,
    ):
        # make sure only one source flag was specified
        source_flags = [name for name, data in
            (('asset', asset), ('path', path), ('texture', texture))
            if data is not None
        ]
        assert len(source_flags) == 1, (
            f'expected exactly one of "asset", "path" or "texture"'
            ' to be supplied to the Mesh constructor, got: {source_flags}.'
        )
        self.source_type = source_flags[0]
        self.asset = asset
        self.path = path
        
        # load the texture from the provided asset or path if necessary
        if self.source_type == 'asset':
            asset_path = AssetLibrary['texture'][asset]
            texture = load_image(asset_path)
        elif self.source_type == 'path':
            texture = load_image(path)
        
        if crop is not None:
            texture = texture[crop[1]:crop[3], crop[0]:crop[2]]
        
        self.texture = texture
    
    @property
    def texture(self):
        return self._texture
    
    @texture.setter
    def texture(self, texture):
        validated_texture = validate_texture(texture)
        self._texture = texture
        self._update_gl_data()
    
    def _update_gl_data(self):
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.gl_texture)
        height, width, channels = self.texture.shape
        if channels == 3:
            gl_color_mode = GL.GL_RGB
        elif channels == 4:
            gl_color_mode = GL.GL_RGBA
        
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D, 0, gl_color_mode,
            width, height, 0,
            gl_color_mode, GL.GL_UNSIGNED_BYTE, self.texture,
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
        GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
        
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
    
    def _cleanup_gl_data(self):
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glDeleteTextures(self.gl_texture)
