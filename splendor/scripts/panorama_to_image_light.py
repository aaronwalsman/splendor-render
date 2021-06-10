#!/usr/bin/env python
import os
import argparse

from splendor.assets import AssetLibrary
from splendor.image_light.panorama import panorama_to_strip
from splendor.image_light.diffuse import reflect_to_diffuse
from splendor.image import even_intensity, load_image, save_image

parser = argparse.ArgumentParser()
parser.add_argument('panorama', type=str)
parser.add_argument('output', type=str)
parser.add_argument('--reflect-size', type=int, default=512)
parser.add_argument('--diffuse-size', type=int, default=64)
parser.add_argument('--filter', type=str, default='linear')
parser.add_argument('--intensity-gamma', type=float, default=1)
parser.add_argument('--samples', type=int, default=100000)
parser.add_argument('--assets', type=str, default=None)
parser.add_argument('--debug', type=str, default=None)
parser.add_argument('--device', type=int, default=None)

def main():
    args = parser.parse_args()

    asset_library = AssetLibrary(args.assets)
    panorama_path = asset_library['panoramas'][args.panorama]
    panorama_image = load_image(panorama_path)
    reflect_image = panorama_to_strip(
        panorama_image, args.reflect_size, args.filter, args.device)
    reflect_intensity = (
        (even_intensity(reflect_image) / 255.) ** args.intensity_gamma * 255.)
    diffuse_image = reflect_to_diffuse(
        args.diffuse_size,
        reflect_image,
        reflect_intensity,
        args.samples,
        args.debug,
        args.device,
    )

    save_image(reflect_image, args.output + '_ref.png')
    save_image(diffuse_image, args.output + '_dif.png')
