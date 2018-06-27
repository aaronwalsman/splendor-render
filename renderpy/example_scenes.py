# system
import math

# numpy
import numpy

# local
import renderpy
import camera

def first_test(width, height):
    r = renderpy.Renderpy()
    
    proj = camera.projection_matrix(math.radians(90), width/height)
    r.set_projection(proj)
    
    r.load_mesh(
            'test_cube',
            './example_meshes/cube.obj',
            './example_meshes/spinner_tex.png')

    r.load_mesh(
            'test_sphere',
            './example_meshes/sphere.obj',
            './example_meshes/candy_color2.png')

    r.add_instance('cube1', 'test_cube', numpy.array(
            [[0.707,0,-0.707,0],[0,1,0,0],[0.707,0,0.707,0],[0,0,0,1]]))
    r.add_instance('sphere1', 'test_sphere', numpy.array(
            [[0.707,0,-0.707,0],[0,1,0,0],[-0.707,0,-0.707,-4],[0,0,0,1]]),
            mask_color = numpy.array([1,1,1]))
    
    r.add_direction_light(
            name = 'light_main',
            direction = numpy.array([0.5,0,-0.866]),
            color = numpy.array([1,1,1]))

    r.set_ambient_color(numpy.array([0.2, 0.2, 0.2]))
    
    return r
