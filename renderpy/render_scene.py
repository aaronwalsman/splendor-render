#!/usr/bin/env python
import time
import sys
import argparse
import numpy
import PIL.Image as Image
import buffer_manager
import core

def render_scene(
        scene,
        width,
        height,
        output_file=None,
        frame_name='render',
        anti_aliasing=True):
    
    manager = buffer_manager.initialize_shared_buffer_manager(width, height)
    try:
        manager.add_frame(
                frame_name, width, height, anti_aliasing=anti_aliasing)
    except buffer_manager.FrameExistsError:
        pass
    
    renderer = core.Renderpy()
    
    renderer.load_scene(scene, clear_existing=True)
    renderer.color_render(flip_y=True)
    
    image = manager.read_pixels(frame_name)
    if output_file is not None:
        pil_image = Image.fromarray(image)
        pil_image.save(output_file)
    
    return image

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Render a Scene')
    parser.add_argument('scene', type=str,
            help='json file containing scene data')
    parser.add_argument('output', type=str,
            help='destination path for rendered image')
    parser.add_argument('--dim', type=str, default='512x512',
            help='dimensions of the output file in WIDTHxHEIGHT format')
    parser.add_argument('--anti-aliasing', type=bool, default=True,
            help='use anti-aliasing')
    
    args = parser.parse_args()
    width, height = (int(wh) for wh in args.dim.split('x'))
    render_scene(
            args.scene,
            width,
            height, 
            output_file = args.output,
            anti_aliasing = args.anti_aliasing)
