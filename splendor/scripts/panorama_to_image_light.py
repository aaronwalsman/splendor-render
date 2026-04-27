#!/usr/bin/env python
"""Convert a panorama image to a reflection cubemap and SH irradiance coefficients."""
import os
import json
import argparse

from splendor.assets import AssetLibrary
from splendor.image_light.panorama import panorama_to_strip
from splendor.image_light.diffuse import cubemap_strip_to_sh
from splendor.image import load_image, save_image

parser = argparse.ArgumentParser()
parser.add_argument('panorama', type=str)
parser.add_argument('output', type=str,
    help='Output prefix. Produces <output>_ref.png and <output>_sh.json.')
parser.add_argument('--reflect-size', type=int, default=512)
parser.add_argument('--filter', type=str, default='linear')
parser.add_argument('--assets', type=str, default=None)
parser.add_argument('--device', type=int, default=None)

def main():
    """Run the panorama to image light conversion pipeline."""
    args = parser.parse_args()

    asset_library = AssetLibrary(args.assets)
    panorama_path = asset_library['panoramas'][args.panorama]
    panorama_image = load_image(panorama_path)

    reflect_image = panorama_to_strip(
        panorama_image, args.reflect_size, args.filter, args.device)

    irradiance_sh = cubemap_strip_to_sh(reflect_image)

    save_image(reflect_image, args.output + '_ref.png')

    with open(args.output + '_sh.json', 'w') as f:
        json.dump(irradiance_sh.tolist(), f)

    print(f'Wrote {args.output}_ref.png')
    print(f'Wrote {args.output}_sh.json')

if __name__ == '__main__':
    main()
