import numpy as np

from splendor.Texture import Texture2D

class SurfaceMaterial(NamedAsset):
    
    def __init__(self,
        name=None,
        texture=None,
        flat_color=None,
        material_properties_texture=None,
        ambient=1.,
        metal=0.,
        rough=0.03,
        base_reflect=0.04,
    ):
        
        super().__init__(data, name=name)
        
