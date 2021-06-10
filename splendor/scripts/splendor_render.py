#!/usr/bin/env python
import argparse

import splendor.render as render

parser = argparse.ArgumentParser(description='Render a Scene')
parser.add_argument('scene', type=str,
        help='json file containing scene data')
parser.add_argument('output', type=str,
        help='destination path for rendered image')
parser.add_argument('--assets', type=str, default=None,
        help='assets.cfg file specifying asset paths')
parser.add_argument('--resolution', type=str, default='512x512',
        help='dimensions of the output file in WIDTHxHEIGHT format')
parser.add_argument('--anti-alias-samples', type=int, default = 8,
        help='number of multisamples used for anti-aliasing, '
            'set to 0 to turn off anti-aliasing')
parser.add_argument('--render-mode', type=str, default='color',
        help='should be either "color", "mask" or "depth"')
parser.add_argument('--device', type=int, default=0,
        help='which device to use for rendering using EGL')

def main():
    args = parser.parse_args()

    width, height = (int(wh) for wh in args.resolution.lower().split('x'))
    anti_alias = args.anti_alias_samples != 0

    render.render_scene(
            args.scene,
            width,
            height, 
            assets = args.assets,
            output_file = args.output,
            anti_alias = anti_alias,
            anti_alias_samples = args.anti_alias_samples,
            render_mode = args.render_mode,
            device = args.device)
