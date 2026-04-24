#!/usr/bin/env python
import json
import argparse

from splendor.assets import AssetLibrary
from splendor.image_light.diffuse import cubemap_strip_to_sh
from splendor.image import load_image

parser = argparse.ArgumentParser(
    description='Compute SH irradiance coefficients from a cubemap strip.')
parser.add_argument('reflect', type=str,
    help='Cubemap strip image (H x 6H, faces px/nx/py/ny/pz/nz).')
parser.add_argument('out', type=str,
    help='Output .json file path.')
parser.add_argument('--assets', type=str, default=None)

def main():
    args = parser.parse_args()

    asset_library = AssetLibrary(args.assets)
    reflect_path = asset_library['image_lights'][args.reflect]
    reflect = load_image(reflect_path)[:, :, :3]

    sh = cubemap_strip_to_sh(reflect)

    with open(args.out, 'w') as f:
        json.dump(sh.tolist(), f)

    print(f'Wrote {args.out}')

if __name__ == '__main__':
    main()
