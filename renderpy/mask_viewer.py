import renderpy.glut as drpy_glut
import renderpy.core as core
import renderpy.camera as camera
import renderpy.masks as masks
import renderpy.primitives as primitives
from renderpy.image import load_image

def start_viewer(file_path):
    
    image = load_image(file_path)
    height, width, _ = image.shape
    
    drpy_glut.initialize_glut()
    
    mask_window = drpy_glut.GlutWindowWrapper(
            'Mask', width, height)
    renderer = core.Renderpy()
    
    rectangle = primitives.mesh_grid(
            axes = (0,1),
            x_divisions = 0,
            y_divisions = 0,
            x_extents = [-width/200., width/200.],
            y_extents = [-height/200., height/200.],
            depth = -width/200.)
    
    renderer.load_mesh('rectangle_mesh', mesh_data=rectangle)
    renderer.load_material('rectangle_mat', texture=file_path)
    renderer.add_instance('rectangle', 'rectangle_mesh', 'rectangle_mat')
    
    renderer.set_ambient_color((1,1,1))
    
    def render():
        mask_window.set_active()
        mask_window.enable_window()
        renderer.color_render(flip_y=False)
    
    def mouse_button(button, button_state, x, y):
        if button_state == 0:
            color = image[y,x]
            print('Index at (%i,%i): %i'%(
                    x, y, masks.color_byte_to_index(color)))
    
    mask_window.start_main_loop(
            glutDisplayFunc = render,
            glutIdleFunc = render,
            glutMouseFunc = mouse_button)
