import os

import numpy
import PIL.Image as Image

def load_image(path):
    '''
    This is a stub so that image loading can be easily changed in one place.
    '''
    #return numpy.array(Image.open(path).convert(mode))
    return numpy.array(Image.open(os.path.expanduser(path)))

def save_image(image, path):
    '''
    This is a stub so that image saving can be easily changed in one place.
    image : a numpy uint8 array with shape HxWx3 with RGB channel order.
    '''
    image = Image.fromarray(image)
    image.save(os.path.expanduser(path))

def load_depth(path):
    '''
    This is a stub so that depth saving can be easily changed in one place.
    Returns a numpy array with shape HxWx1.
    '''
    with open(os.path.expanduser(path), 'rb') as f:
        return numpy.load(f)

def save_depth(depth, path):
    '''
    This is a stub so that depth saving can be easily changed in one place.
    depth : a numpy float array with shape HxWx1.
    '''
    with open(os.path.expanduser(path), 'wb') as f:
        numpy.save(f, depth)

def resize_image(image, width, height):
    image = Image.fromarray(image).resize((width, height), Image.BILINEAR)
    return numpy.array(image)

def intensity(image):
    return (0.2989 * image[...,0] +
            0.5870 * image[...,1] +
            0.1141 * image[...,2])

def even_intensity(image):
    return (0.3333 * image[...,0] +
            0.3334 * image[...,1] +
            0.3333 * image[...,2])

def validate_texture(image):
    if image.shape[0] not in [1,2,4,8,16,32,64,128,256,512,1024,2048,4096]:
        raise ValueError('Image height must be a power of 2 '
                'less than or equal to 4096 (Got %i)'%(image.shape[0]))
    if image.shape[1] not in [1,2,4,8,16,32,64,128,256,512,1024,2048,4096]:
        raise ValueError('Image width must be a power of 2 '
                'less than or equal to 4096 (Got %i)'%(image.shape[1]))
