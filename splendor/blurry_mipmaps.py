#!/usr/bin/env python

import math

from PIL import Image, ImageFilter
import numpy

def blurry_mipmaps(image, stop):
    if isinstance(image, str):
        image = Image.open(image)
    
    if not isinstance(image, Image.Image):
        if numpy.max(image) <= 1.0:
            image = numpy.uint8(image * 255)
        image = Image.fromarray(image)
    
    mipmaps = []
    
    while image.width > stop and image.height > stop:
        image = image.resize((image.width//2, image.height//2), Image.ANTIALIAS)
        image = image.filter(ImageFilter.GaussianBlur(radius=2))
        mipmaps.append(image)
    
    return mipmaps

# standard cubemap image neighbors and orientations
# +1 = 90 degrees clockwise, -1 = 90 degrees counterclockwise, 2 = 180 degrees
cube_neighbors = {
    'px' : {
        'l' : ('pz', 0),
        'r' : ('nz', 0),
        't' : ('py', 3),
        'b' : ('ny', 1)},
    'nx' : {
        'l' : ('nz', 0),
        'r' : ('pz', 0),
        't' : ('py', 1),
        'b' : ('ny', 3)},
    'py' : {
        'l' : ('nx', 3),
        'r' : ('px', 1),
        't' : ('nz', 2),
        'b' : ('pz', 0)},
    'ny' : {
        'l' : ('nx', 1),
        'r' : ('px', 3),
        't' : ('pz', 0),
        'b' : ('nz', 2)},
    'pz' : {
        'l' : ('nx', 0),
        'r' : ('px', 0),
        't' : ('py', 0),
        'b' : ('ny', 0)},
    'nz' : {
        'l' : ('px', 0),
        'r' : ('nx', 0),
        't' : ('py', 2),
        'b' : ('ny', 2)},
}

def blurry_cube_mipmaps(images, start, stop, radius=2):
    current = start
    cube_mipmaps = []
    while current > stop:
        images = blurry_cube_mipmap_step(images, radius=radius)
        cube_mipmaps.append(images)
        current /= 2
    
    return cube_mipmaps

def blurry_cube_mipmap_step(images, radius=2):
    r = math.ceil(radius)
    loaded_images = {}
    for cube_face in images:
        image = images[cube_face]
        if isinstance(image, str):
            image = Image.open(image)

        if not isinstance(image, Image.Image):
            if numpy.max(image) <= 1.0:
                image = numpy.uint8(image * 255)
            image = Image.fromarray(image)
        
        loaded_images[cube_face] = image
    
    result = {}
    for cube_face in loaded_images:
        image = loaded_images[cube_face]
        width = image.width
        height = image.height
        image = numpy.array(image)
        expanded_image = numpy.zeros(
                (height+2*r, width+2*r, 3))
        expanded_image[
                r:height+r,
                r:width+r,:] = image
        
        expanded_image[0:r,0:r,:] = image[0,0,:]
        expanded_image[-r:,0:r,:] = image[-1,0,:]
        expanded_image[0:r,-r:,:] = image[0,-1,:]
        expanded_image[-r:,-r:,:] = image[-1,-1,:]
        
        neighbors = cube_neighbors[cube_face]
        
        def rotated_image(image, index):
            image = numpy.array(image).copy()
            if index:
                image = numpy.rot90(image, index)
            return image
        
        # left
        left_neighbor, left_rotate = neighbors['l']
        left_image = rotated_image(loaded_images[left_neighbor], left_rotate)
        expanded_image[
                r:height+r,
                0:r,:] = left_image[:,-r:,:]
        
        # right
        right_neighbor, right_rotate = neighbors['r']
        right_image = rotated_image(loaded_images[right_neighbor], right_rotate)
        expanded_image[
                r:height+r,
                width+r:,:] = right_image[:,:r,:]
        
        # top
        top_neighbor, top_rotate = neighbors['t']
        top_image = rotated_image(loaded_images[top_neighbor], top_rotate)
        expanded_image[
                0:r,
                r:height+r,:] = top_image[-r:,:,:]
        
        # btm
        btm_neighbor, btm_rotate = neighbors['b']
        btm_image = rotated_image(loaded_images[btm_neighbor], btm_rotate)
        expanded_image[
                -r:,
                r:width+r,:] = btm_image[:r,:,:]
        
        image = Image.fromarray(expanded_image.astype(numpy.uint8))
        image = image.filter(ImageFilter.GaussianBlur(radius=2))
        image = image.crop(box=(r,r,width+r,height+r))
        image = image.resize((image.width//2, image.height//2), Image.ANTIALIAS)
        
        result[cube_face] = image
    
    return result


if __name__ == '__main__':
    
    '''
    for cube_face in 'px', 'nx', 'py', 'ny', 'pz', 'nz':
        test_image = Image.open('example_background/%s_ref.jpg'%cube_face)
        mipmaps = blurry_mipmaps(test_image, 8)
        for i, mipmap in enumerate(mipmaps):
            mipmap.save('example_background/%s_ref_%i.jpg'%(cube_face, i+1))
    '''
    
    images = {cube_face:'example_background/%s_ref.jpg'%cube_face
            for cube_face in ('px', 'nx', 'py', 'ny', 'pz', 'nz')}
    
    cube_mipmaps = blurry_cube_mipmaps(images, 1024, 8, radius=1)
    for i, mipmaps in enumerate(cube_mipmaps):
        for cube_face in mipmaps:
            mipmaps[cube_face].save(
                    'example_background/%s_ref_%i.jpg'%(cube_face, i+1))

