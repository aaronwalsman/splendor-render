#!/usr/bin/env python
import time
import sys
import argparse
import numpy
from renderpy import buffer_manager
from renderpy import core
from renderpy.image import save_image, save_depth

def render_scene(
        scene,
        width,
        height,
        output_file=None,
        frame_name='render',
        anti_alias=True,
        render_mode='color'):
    
    manager = buffer_manager.initialize_shared_buffer_manager(width, height)
    try:
        manager.add_frame(
                frame_name, width, height, anti_aliasing=anti_alias)
    except buffer_manager.FrameExistsError:
        pass
    
    manager.enable_frame(frame_name)
    renderer = core.Renderpy()
    
    renderer.load_scene(scene, clear_existing=True)
    if render_mode == 'color' or render_mode == 'depth':
        renderer.color_render(flip_y=True)
    elif render_mode == 'mask':
        renderer.mask_render(flip_y=True)
    else:
        raise NotImplementedError
    
    image = manager.read_pixels(frame_name, read_depth=(render_mode=='depth'))
    if output_file is not None:
        if render_mode == 'depth':
            save_depth(image, output_file)
        else:
            save_image(image, output_file)
    
    return image

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Render a Scene')
    parser.add_argument('scene', type=str,
            help='json file containing scene data')
    parser.add_argument('output', type=str,
            help='destination path for rendered image')
    parser.add_argument('--dim', type=str, default='512x512',
            help='dimensions of the output file in WIDTHxHEIGHT format')
    parser.add_argument('--anti-alias', dest='anti_alias', action='store_true',
            help='turn anti-alias on (default is on)')
    parser.add_argument(
            '--no-anti-alias', dest='anti_alias', action='store_false',
            help='turn anti-alias off')
    parser.set_defaults(anti_alias=True)
    parser.add_argument('--render-mode', type=str, default='color',
            help='should be either "color", "mask" or "depth"')
    
    args = parser.parse_args()
    width, height = (int(wh) for wh in args.dim.split('x'))
    render_scene(
            args.scene,
            width,
            height, 
            output_file = args.output,
            anti_alias = args.anti_alias,
            render_mode = args.render_mode)
