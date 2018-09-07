#!/usr/bin/env python

import time
import os

import tqdm

import torchvision.transforms as transforms
T = transforms.ToTensor()

import numpy

import imageio
from PIL import Image
import cv2

base_path = '/home/awalsman/Development/pexel_textures'
images = [os.path.join(base_path, f) for f in os.listdir(base_path)][:30]

iterate = tqdm.tqdm(images)
iterate.set_description('PIL')
for image in iterate:
    img = Image.open(image)
    img.load()
    #img = T(img)

iterate = tqdm.tqdm(images)
iterate.set_description('imageio')
for image in iterate:
    img = imageio.imread(image)

iterate = tqdm.tqdm(images)
iterate.set_description('cv2')
for image in iterate:
    img = cv2.imread(image)

