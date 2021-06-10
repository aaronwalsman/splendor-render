#!/usr/bin/env python
import os
import argparse

from splendor.assets import AssetLibrary
import splendor.image_light.panorama as panorama
from splendor.image import load_image, save_image

parser = argparse.ArgumentParser()
parser.add_argument('panorama', type=str)
parser.add_argument('output', type=str)
parser.add_argument('--size', type=int, default=512)
parser.add_argument('--filter', type=str, default='linear')
parser.add_argument('--assets', type=str, default=None)
parser.add_argument('--device', type=int, default=None)

def main():
    args = parser.parse_args()

    asset_library = AssetLibrary(args.assets)
    panorama_path = asset_library['panoramas'][args.panorama]
    panorama_image = load_image(panorama_path)
    out_image = panorama.panorama_to_strip(
            panorama_image, args.size, args.filter, args.device)
    save_image(out_image, args.output)
