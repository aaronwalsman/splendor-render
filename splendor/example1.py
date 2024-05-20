from splendor import EGLRenderer, Scene, Mesh, Material, Texture

def example():
    # intialize the renderer
    renderer = EGLRenderer()
    
    # load the assets we are going to use
    cereal_mesh = asset_library['mesh']['cereal']
    texture = asset_library['texture']['cereal_rgb']
    material_property_texture = asset_library['material']['cereal_mat']
    material = SurfaceMaterial(SOMETHING)
    
    # build a new scene
    scene = Scene()
    
    # set the scene properties
    scene.set_background_color((1,0,0))
    scene.set_ambient_color((1,1,1))
    
    # add a new cereal box instance
    cereal_transform = np.array([
        [ 1, 0, 0,   0],
        [ 0, 1, 0,   0],
        [ 0, 0, 1, -10],
        [ 0, 0, 0,   1],
    ])
    instance = scene.add_instance(
        mesh=cereal_mesh,
        material=cereal_material,
        transform=cereal_transform,
    )
    
    # add a camera
    camera = scene.add_camera(
        transform=np.eye(4),
        horizontal_fov=math.radians(90.)
    )
    
    # render a new image
    image = renderer.color_render(scene, camera)
    
    # save the scene so it can be used again later
    renderer.save('./example1_scene.json')
