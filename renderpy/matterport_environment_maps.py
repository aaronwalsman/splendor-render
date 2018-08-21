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

# imageio
import imageio

# local
import renderpy.resize_image as ri
import renderpy.reflection_to_diffuse as r2d

parser = argparse.ArgumentParser(description = 
        'Convert Matterport panoramas to reflective and diffuse cube maps')
parser.add_argument('matterport_location', type=str,
        help = 'The location of the matterport data')
parser.add_argument('destination', type=str,
        help = 'The destination for the cube maps')
parser.add_argument('--reflection-resolution', type=int, default=-1,
        help = 'The resolution of the reflection cube map')
parser.add_argument('--diffuse-resolution', type=int, default=128,
        help = 'The resolution of the diffuse cube map')

args = parser.parse_args()

houses_path = os.path.join(args.matterport_location, 'v1', 'scans')
house_hashes = os.listdir(houses_path)

for i, house in enumerate(house_hashes):
    print('='*80)
    print(house, '(%i/%i)'%(i+1, len(house_hashes)))
    
    house_path = os.path.join(
            houses_path, house, 'matterport_skybox_images')
    skybox_reflection_images = [image for image in os.listdir(house_path)
            if image[-4:] == '.jpg']
    house_positions = {skybox_reflection_image.split('_')[0]
            for skybox_reflection_image in skybox_reflection_images}
    
    for house_position in tqdm.tqdm(house_positions):
        index_mapping = {0:'py', 1:'pz', 2:'px', 3:'nz', 4:'nx', 5:'ny'}
        reflection_images = {}
        for index in range(6):
            image = numpy.array(imageio.imread(os.path.join(
                    house_path, house_position + '_skybox%i_sami.jpg'%index)))
            if args.reflection_resolution != -1:
                image = ri.resize_image(
                        image,
                        args.reflection_resolution,
                        args.reflection_resolution)
            reflection_images[index_mapping[index]] = image
        
        diffuse_images = r2d.reflection_to_diffuse(
                reflection_images, args.diffuse_resolution)
        output_path = os.path.join(args.destination, house_position)
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        for cube_face in diffuse_images:
            diffuse_path = os.path.join(output_path, cube_face + '_dif.jpg')
            imageio.imsave(
                    diffuse_path,
                    diffuse_images[cube_face],
                    quality=100)
            reflection_path = os.path.join(output_path, cube_face + '_ref.jpg')
            imageio.imsave(
                    reflection_path,
                    reflection_images[cube_face],
                    quality=100)

print('='*80)
