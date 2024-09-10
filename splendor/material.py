import numpy as np

import OpenGL.GL as GL

from splendor.named_asset import NamedAsset
from splendor.texture import Texture2D

class SurfaceMaterial(NamedAsset):
    
    #_SUPPORTED_ALBEDOS = ('FLAT', 'VERTEX_COLOR', 'TEXTURE')
    
    _loaded_assets = {}
    
    _active_material = None
    
    def __init__(self,
        name=None,
        texture=None,
        flat_color=None,
        #albedo='FLAT',
        material_properties_texture=None,
        ambient=1.,
        metal=0.,
        rough=0.03,
        reflect=0.04,
    ):
        # register this material name
        super().__init__(name=name)
        
        if texture is not None:
            assert isinstance(texture, Texture2D)
        #assert albedo in self._SUPPORTED_ALBEDOS
        if material_properties_texture is not None:
            assert isinstance(material_properties_texture, Texture2D)
        
        self.texture = texture
        self.flat_color = flat_color
        self.material_properties_texture = material_properties_texture
        self.ambient = ambient
        self.metal = metal
        self.rough = rough
        self.reflect = reflect
    
    @property
    def texture(self):
        return self._texture
    
    @texture.setter
    def texture(self, texture):
        assert texture is None
        self._texture = texture
        self._dirty = True
    
    @property
    def flat_color(self):
        return self._flat_color
    
    @flat_color.setter
    def flat_color(self, color):
        assert np.shape(color) == (3,)
        self._flat_color = np.array(color, dtype=np.float32)
        self._flat_color.setflags(write=False)
        self._dirty = True
    
    @property
    def material_properties_texture(self):
        return self._material_properties_texture
    
    @material_properties_texture.setter
    def material_properties_texture(self, material_properties_texture):
        assert material_properties_texture is None
        self._material_properties_texture = material_properties_texture
        self._dirty = True
    
    @property
    def ambient(self):
        return self._ambient
    
    @ambient.setter
    def ambient(self, ambient):
        self._ambient = ambient
        self._dirty = True
    
    @property
    def metal(self):
        return self._metal
    
    @metal.setter
    def metal(self, metal):
        self._metal = metal
        self._dirty = True
    
    @property
    def rough(self):
        return self._rough
    
    @rough.setter
    def rough(self, rough):
        self._rough = rough
        self._dirty = True
    
    @property
    def reflect(self):
        return self._reflect
    
    @reflect.setter
    def reflect(self, reflect):
        self._reflect = reflect
        self._dirty = True
    
    @property
    def is_textured(self):
        return self.texture is not None
    
    @property
    def is_material_properties_textured(self):
        return self.material_properties_texture is not None
    
    def activate(self, shader_locations):
        if self._active_material is not self or self._dirty:
            print('loading material')
            material_properties = np.array(
                [self.metal, self.rough, self.reflect, self.ambient],
                dtype=np.float32,
            )
            #GL.glUniform1i(shader_locations['is_textured'], self.is_textured)
            if not self.is_textured:
                print('setting flat color')
                GL.glUniform3fv(
                    shader_locations['flat_color'], 1, self.flat_color)
            if not self.is_material_properties_textured:
                print('setting material properties')
                GL.glUniform4fv(
                    shader_locations['material_properties'],
                    1,
                    material_properties,
                )
            
            self._active_material = self
            self._dirty = False
