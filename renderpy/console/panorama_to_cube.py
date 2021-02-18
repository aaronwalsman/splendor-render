#!/usr/bin/env python
import os
import argparse

import renderpy.image_light.panorama as panorama
import renderpy.image as image

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('panorama', type=str)
    parser.add_argument('output', type=str)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--filter', type=str, default='linear')

    args = parser.parse_args()

    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    panorama_image = image.load_image(args.panorama)
    out_images = panorama.panorama_to_cube(panorama_image, args.size, args.filter)
    for cube_face in out_images:
        face_path = os.path.join(args.output, '%s_ref.png'%cube_face)
        image.save_image(out_images[cube_face], face_path)
