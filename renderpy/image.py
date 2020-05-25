import numpy
import PIL.Image as Image

def load_image(path):
    '''
    This is a stub so that image loading can be easily changed in one place.
    Returns a numpy uint8 array with shape HxWx3 with RGB channel order.
    '''
    return numpy.array(Image.open(path).convert('RGB'))

def save_image(image, path):
    '''
    This is a stub so that image saving can be easily changed in one place.
    image : a numpy uint8 array with shape HxWx3 with RGB channel order.
    '''
    image = Image.fromarray(image)
    image.save(path)

def load_depth(path):
    '''
    This is a stub so that depth saving can be easily changed in one place.
    Returns a numpy uint8 array with shape HxWx1.
    '''
    with open(path, 'rb') as f:
        return numpy.load(f)

def save_depth(depth, path):
    '''
    This is a stub so that depth saving can be easily changed in one place.
    depth : a numpy float array with shape HxWx1.
    '''
    with open(path, 'wb') as f:
        numpy.save(f, depth)
