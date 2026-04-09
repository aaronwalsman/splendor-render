from splendor.contexts.glfw import GLFWContext
import splendor.core as core
import splendor.camera as camera
import splendor.masks as masks
import splendor.primitives as primitives
from splendor.image import load_image


def start_viewer(file_path):

    image = load_image(file_path)
    height, width, _ = image.shape

    with GLFWContext(width=width, height=height, title='Mask Viewer') as ctx:
        renderer = core.SplendorRender()

        rectangle = primitives.mesh_grid(
            axes=(0, 1),
            x_divisions=0,
            y_divisions=0,
            x_extents=[-width / 200., width / 200.],
            y_extents=[-height / 200., height / 200.],
            depth=-width / 200.)

        renderer.load_mesh('rectangle_mesh', mesh_data=rectangle)
        renderer.load_texture('rectangle_texture', texture_path=file_path)
        renderer.load_material('rectangle_mat', texture_name='rectangle_texture')
        renderer.add_instance('rectangle', 'rectangle_mesh', 'rectangle_mat')
        renderer.set_ambient_color((1, 1, 1))

        cursor = {'x': 0., 'y': 0.}

        def cursor_pos_callback(window, x, y):
            cursor['x'] = x
            cursor['y'] = y

        def mouse_button_callback(window, button, action, mods):
            import glfw as _glfw
            if action == _glfw.PRESS:
                x = int(cursor['x'])
                y = int(cursor['y'])
                if 0 <= x < width and 0 <= y < height:
                    color = image[y, x]
                    print(f'Index at ({x}, {y}): '
                          f'{masks.color_byte_to_index(color)}')

        ctx.set_cursor_pos_callback(cursor_pos_callback)
        ctx.set_mouse_button_callback(mouse_button_callback)

        while not ctx.should_close():
            ctx.poll_events()
            fbw, fbh = ctx.framebuffer_size()
            renderer.viewport_scissor(0, 0, fbw, fbh)
            renderer.color_render(flip_y=False)
            ctx.swap_buffers()
