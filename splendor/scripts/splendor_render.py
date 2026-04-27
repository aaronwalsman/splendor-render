#!/usr/bin/env python
"""CLI for rendering a scene to an image file."""
import argparse

import splendor.render as render

parser = argparse.ArgumentParser(description='Render a Scene')
parser.add_argument('scene', type=str,
        help='json file containing scene data')
parser.add_argument('output', type=str,
        help='destination path for rendered image')
parser.add_argument('--assets', type=str, default=None,
        help='assets.cfg file specifying asset paths')
parser.add_argument('--camera', type=str, default='main',
        help='name of the camera defined in the scene to render from')
parser.add_argument('--sensor', type=str, default='output',
        help='name of the sensor to render into; if not defined in the scene '
             'one will be created from --resolution and --anti-alias-samples')
parser.add_argument('--resolution', type=str, default=None,
        help='resolution of the output image in WIDTHxHEIGHT format '
             '(ignored if --sensor is already defined in the scene)')
parser.add_argument('--anti-alias-samples', type=int, default=None,
        help='number of MSAA samples; 0 disables anti-aliasing '
             '(ignored if --sensor is already defined in the scene)')
parser.add_argument('--render-mode', type=str, default='color',
        help='one of: color, mask, depth')
parser.add_argument('--device', type=int, default=0,
        help='which EGL device to use for rendering')

def main():
    """Parse CLI args and render a scene."""
    args = parser.parse_args()

    resolution = args.resolution
    anti_alias_samples = args.anti_alias_samples

    render.render_scene(
        args.scene,
        assets=args.assets,
        output_file=args.output,
        camera=args.camera,
        sensor=args.sensor,
        resolution=resolution,
        anti_alias_samples=anti_alias_samples,
        render_mode=args.render_mode,
        device=args.device,
    )
