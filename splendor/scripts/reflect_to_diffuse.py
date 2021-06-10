#!/usr/bin/env python
import argparse

import numpy

from splendor.assets import AssetLibrary
from splendor.image_light.diffuse import reflect_to_diffuse
from splendor.image import intensity, even_intensity, load_image, save_image

parser = argparse.ArgumentParser()
parser.add_argument('reflect', type=str)
parser.add_argument('out', type=str)
parser.add_argument('--size', type=int, default=64)
parser.add_argument('--intensity-gamma', type=float, default=1)
parser.add_argument('--samples', type=int, default=100000)
parser.add_argument('--assets', type=str, default=None)
parser.add_argument('--debug', type=str, default=None)
parser.add_argument('--device', type=int, default=None)

def main():
    args = parser.parse_args()

    asset_library = AssetLibrary(args.assets)
    reflect_path = asset_library['image_lights'][args.reflect]
    reflect = load_image(reflect_path)
    reflect_intensity = (
            (even_intensity(reflect) / 255.) ** args.intensity_gamma * 255.)

    diffuse = reflect_to_diffuse(
            args.size,
            reflect,
            reflect_intensity,
            args.samples,
            args.debug,
            args.device)

    save_image(diffuse, args.out)
