
# numpy
import numpy

# imageio
import imagio

def sample_rectangle(panorama, depth, x_min, y_min, x_max, y_max, resolution):
    x_range = numpy.arange(x_min, x_max, (x_max - x_min)/resolution[0])
    pixel_width = x_range[1] - x_range[0]
    x_range += pixel_width * 0.5
    y_range = numpy.arange(y_min, y_max, (y_max - y_min)/resolution[1])
    pixel_height = y_range[1] - y_range[0]
    y_range += pixel_height * 0.5
    
    x_coord, y_coord = numpy.meshgrid(x_range, y_range)
    
    direction = 
