import splendor.contexts.glut as glut
import splendor.core as core
import splendor.camera as camera
import splendor.masks as masks
import splendor.primitives as primitives
from splendor.image import load_image

def start_viewer(file_path):
    
    image = load_image(file_path)
    height, width, _ = image.shape
    
    glut.initialize()
    
    mask_window = glut.GlutWindowWrapper(
            'Mask', width, height)
    renderer = core.SplendorRender()
    
    rectangle = primitives.mesh_grid(
            axes = (0,1),
            x_divisions = 0,
            y_divisions = 0,
            x_extents = [-width / 200., width / 200.],
            y_extents = [-height / 200., height / 200.],
            depth = -width / 200.)
    
    renderer.load_mesh('rectangle_mesh', mesh_data=rectangle)
    renderer.load_texture('rectangle_texture', texture_path=file_path)
    renderer.load_material('rectangle_mat', texture_name='rectangle_texture')
    renderer.add_instance('rectangle', 'rectangle_mesh', 'rectangle_mat')
    
    renderer.set_ambient_color((1, 1, 1))
    
    def render():
        mask_window.set_active()
        mask_window.enable_window()
        renderer.color_render(flip_y=False)
    
    def mouse_button(button, button_state, x, y):
        if button_state == 0:
            color = image[y, x]
            print(f'Index at ({x}, {y}): {masks.color_byte_to_index(color)}')
    
    mask_window.register_callbacks(
            glutDisplayFunc = render,
            glutIdleFunc = render,
            glutMouseFunc = mouse_button)
    
    glut.start_main_loop()
