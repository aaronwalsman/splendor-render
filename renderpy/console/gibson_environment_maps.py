#!/usr/bin/env python

# system
import sys
import os
import argparse
import re

# tqdm
import tqdm

# numpy
import numpy

# local
import renderpy.image
import renderpy.panorama_to_cube as p2c
import renderpy.reflection_to_diffuse as r2d

def fix_black_bars(image):
    row = 0
    fix = False
    while numpy.sum(image[row]) == 0.0:
        row += 1
        fix = True

    if fix:
        image[:row,:,:] = image[row]

    row = -1
    fix = False
    while numpy.sum(image[row]) == 0.0:
        row -= 1
        fix = True

    if fix:
        image[row+1:,:,:] = image[row]

def run():
    parser = argparse.ArgumentParser(description =
            'Convert Matterport panoramas to reflective and diffuse cube maps')
    parser.add_argument('gibson_location', type=str,
            help = 'The location of the gibson data')
    parser.add_argument('destination', type=str,
            help = 'The destination for the cube maps')
    parser.add_argument('--reflection-resolution', type=int, default=512,
            help = 'The resolution of the reflection cube map')
    parser.add_argument('--diffuse-resolution', type=int, default=128,
            help = 'The resolution of the diffuse cube map')

    args = parser.parse_args()

    try:
        with open(os.path.join(args.gibson_location, 'manifest')) as manifest:
            houses = [line.strip() for line in manifest.readlines()]

    except FileNotFoundError:
        raise Exception(
                'Could not find "manifest" file in the gibson directory\n' +
                args.gibson_location)

    for i, house in enumerate(houses):
        print('='*80)
        print(house, '(%i/%i)'%(i+1, len(houses)))

        pano_rgb = os.path.join(args.gibson_location, house, 'pano', 'rgb')
        if not os.path.isdir(pano_rgb):
            print('Can not find panorama directory for %s (%s)'%(house, pano_rgb))
            continue

        pano_files = os.listdir(pano_rgb)
        for pano_file in tqdm.tqdm(pano_files):

            pano_index = re.search('[0-9]+', pano_file)[0]

            #pano = numpy.array(
            #        imageio.imread(os.path.join(pano_rgb, pano_file)))[:,:,:3]
            pano = renderpy.image.load_image(os.path.join(pano_rgb, pano_file))
            fix_black_bars(pano)
            reflection_images = p2c.panorama_to_cube(
                    pano, args.reflection_resolution)
            reflection_dir = os.path.join(
                    args.destination, house + '_' + pano_index + '_ref')
            if not os.path.isdir(reflection_dir):
                os.makedirs(reflection_dir)
            for cube_face in reflection_images:
                image_path = os.path.join(reflection_dir, cube_face + '.png')
                #imageio.imsave(image_path, reflection_images[cube_face])
                renderpy.image.save_image(reflection_images[cube_face], image_path)

            diffuse_images = r2d.reflection_to_diffuse(
                    reflection_images, args.diffuse_resolution)
            diffuse_dir = os.path.join(
                    args.destination, house + '_' + pano_index + '_dif')
            if not os.path.isdir(diffuse_dir):
                os.makedirs(diffuse_dir)
            for cube_face in diffuse_images:
                image_path = os.path.join(diffuse_dir, cube_face + '.png')
                imageio.imsave(image_path, diffuse_images[cube_face])

        break
