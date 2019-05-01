#!/usr/bin/env python
import time
import math
import argparse

import numpy

import renderpy.buffer_manager as buffer_manager
import renderpy.core as core
import renderpy.example_scenes as example_scenes


if __name__ == '__main__':
    
    # args
    parser = argparse.ArgumentParser(description='Render while spinning.')
    parser.add_argument(
            '--scene', type=str,
            help='An image light directory')
    parser.add_argument(
            '--width', type=int, help = 'window width', default=512)
    parser.add_argument(
            '--height', type=int, help = 'window height', default=512)
    
    args = parser.parse_args()
    
    # initialization
    width = args.width
    height = args.height

    manager = buffer_manager.initialize_shared_buffer_manager(width)
    manager.add_frame('A', width, height)
    manager.enable_frame('A')

    rendererA = core.Renderpy()
    rendererA.load_scene(args.scene)

    theta = [0.0]
    translate = numpy.array([[1,0,0,0],[0,1,0,0],[0,0,1,10],[0,0,0,1]])
    e = math.radians(-30)
    elevate = numpy.array([
            [1, 0, 0, 0],
            [0, math.cos(e), -math.sin(e), 0],
            [0, math.sin(e), math.cos(e), 0],
            [0, 0, 0, 1]])

    t0 = time.time()
    rendered_frames = 0
    while True:
        tmp_r = math.pi * 1.5
        rotate = numpy.array([
                [math.cos(theta[0]), 0, -math.sin(theta[0]), 0],
                [0, 1, 0, 0],
                [math.sin(theta[0]), 0,  math.cos(theta[0]), 0],
                [0, 0, 0, 1]])

        c = numpy.linalg.inv(
                numpy.dot(rotate, numpy.dot(elevate, translate)))
        rendererA.set_camera_pose(c)

        blur = (rendered_frames % 10000)/10000. * 8

        theta[0] += math.pi * 2 / 5000.

        manager.show_window()
        manager.enable_window()
        rendererA.color_render(flip_y=False)
        
        rendered_frames +=1
        if rendered_frames % 100 == 0:
            print('hz: %.04f'%(rendered_frames / (time.time() - t0)))

        save_increment = 1000

