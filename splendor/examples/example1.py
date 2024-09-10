'''
from splendor import (
    EGLContext,
    Scene,
    Mesh,
    Material,
    Texture,
    MeshInstance,
    PinholeCamera,
)
'''

import math

import numpy as np

#from splendor.contexts.egl import EGLContext
from splendor.session import RenderSession
from splendor.mesh import Mesh
from splendor.texture import Texture2D
from splendor.material import SurfaceMaterial
from splendor.instance import MeshInstance
from splendor.camera import PinholeCamera
from splendor.image import save_image

def example():
    # intialize the session
    #context = EGLContext()
    session = RenderSession(context_type='EGL')
    
    with session:
    
        # load the assets we are going to use
        cereal_mesh = Mesh(
            name='cereal_mesh',
            vertices=[[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]],
            normals=[[0,0,1],[0,0,1],[0,0,1],[0,0,1]],
            uvs=[[0,0],[1,0],[1,1],[0,1]],
            faces=[[0,1,2],[0,2,3]],
        )
        
        #cereal_texture = Texture2D(
        #    name='cereal_texture', path='./cereal_texture.png')
        
        cereal_material = SurfaceMaterial(
            name='cereal_material',
            flat_color=(1,1,0),
        )
        
        # add a new cereal box instance
        cereal_transform = np.array([
            [ 1, 0, 0,   0],
            [ 0, 1, 0,   0],
            [ 0, 0, 1,  -5],
            [ 0, 0, 0,   1],
        ])
        instance = MeshInstance(
            name='cereal_instance',
            mesh=cereal_mesh,
            material=cereal_material,
            transform=cereal_transform,
        )
        
        # add a camera
        camera = PinholeCamera(
            fov=math.radians(30.),
        )
        
        # render a new image
        image, = camera.render(
            instances=(instance,),
            background_color=(0,0,0,0),
        )
        
        save_image(image, './tmp.png')
        
        # save the scene so it can be used again later
        #scene.save('./example1_scene.json')

if __name__ == '__main__':
    example()
