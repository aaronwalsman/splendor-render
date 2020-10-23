#import numpy.random
#numpy.random.seed(1234)

import numpy

CELL_WIDTH = 8
CELL_OFFSET = CELL_WIDTH//2
NUM_CELLS = 256//CELL_WIDTH
NUM_MASKS = NUM_CELLS**3

#color_index_to_scramble = numpy.random.permutation(NUM_MASKS-1)+1
#color_index_to_scramble = numpy.concatenate(([0], color_index_to_scramble))
# generate a scramble/descramble lookup so that nearby indices will not be
# similar to each other
a = 7019 # a medium-sized prime
color_index_to_scramble = numpy.array(
        [(i*a)%(NUM_MASKS) for i in range(NUM_MASKS)])
color_scramble_to_index = numpy.zeros(
        NUM_MASKS, dtype=color_index_to_scramble.dtype)
color_scramble_to_index[color_index_to_scramble] = numpy.arange(NUM_MASKS)

def color_float_to_byte(f):
    return numpy.round(f*255).astype(numpy.uint8)

def color_byte_to_float(b):
    return numpy.array(b).astype(numpy.float) / 255

def color_index_to_byte(index):
    scramble = color_index_to_scramble[index]
    r = scramble % NUM_CELLS * CELL_WIDTH + CELL_OFFSET
    g = (scramble // NUM_CELLS) % NUM_CELLS * CELL_WIDTH + CELL_OFFSET
    b = (scramble // NUM_CELLS**2) % NUM_CELLS * CELL_WIDTH + CELL_OFFSET
    rgb = numpy.stack((r,g,b), axis=-1).astype(numpy.uint8)
    return rgb

def color_byte_to_index(byte):
    byte = numpy.array(byte).astype(numpy.int)//CELL_WIDTH
    r = byte[..., 0]
    g = byte[..., 1]
    b = byte[..., 2]
    scramble = r + g * NUM_CELLS + b * NUM_CELLS**2
    return color_scramble_to_index[scramble]

def color_index_to_float(index):
    byte_color = color_index_to_byte(index)
    return color_byte_to_float(byte_color)

'''
def test_a_thing():
    #a = numpy.random.randint(0, NUM_MASKS, size=(256,256))
    a = 12
    import time
    t0 = time.time()
    b = color_index_to_byte(a)
    #noise = numpy.random.randint(-1,2, size=(512,512,3))
    #b = b + noise
    t1 = time.time()
    i = color_byte_to_index(b)
    t2 = time.time()
    
    print(t1 - t0)
    print(t2 - t1)
    
    print(b.shape)
    print(numpy.all(i == a))
'''
