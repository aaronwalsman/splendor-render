import splendor.contexts.egl as egl
import splendor.core as core
from splendor.image import save_image, save_depth
from splendor.frame_buffer import FrameBufferWrapper

def render_scene(
    scene,
    width,
    height,
    assets = None,
    output_file = None,
    anti_alias = True,
    anti_alias_samples = 8,
    render_mode = 'color',
    device = None,
):
    
    egl.initialize_plugin()
    egl.initialize_device(device)
    
    framebuffer = FrameBufferWrapper(
            width, height, anti_alias, anti_alias_samples)
    framebuffer.enable()
    renderer = core.SplendorRender(assets=assets)
    renderer.load_scene(scene, clear_scene=True)
        
    if render_mode == 'color' or render_mode == 'depth':
        renderer.color_render(flip_y=True)
    elif render_mode == 'mask':
        renderer.mask_render(flip_y=True)
    else:
        raise NotImplementedError
    
    image = framebuffer.read_pixels(
            read_depth=(render_mode=='depth'),
            projection=renderer.get_projection())
    
    if output_file is not None:
        if render_mode == 'depth':
            save_depth(image, output_file)
        else:
            save_image(image, output_file)
    
    return image
