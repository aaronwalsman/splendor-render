#!/usr/bin/env python
import argparse

import splendor.viewer as viewer

parser = argparse.ArgumentParser(description='Splendor Viewer')
parser.add_argument('file_path', type=str,
        help='json file containing scene data')
parser.add_argument('--resolution', type=str, default='512x512',
        help='resolution of the image in WIDTHxHEIGHT format')
parser.add_argument('--poll-frequency', type=int, default = 1024,
        help='frequency with which to check for scene updates')
parser.add_argument('--anti-alias-samples', type=int, default = 8,
        help='number of multisamples used for anti-aliasing, '
            'set to 0 to turn off anti-aliasing')
parser.add_argument('--assets', type=str, default=None)
parser.add_argument('--fps', action='store_true',
        help='print fps')

def main():
    args = parser.parse_args()

    width, height = (int(wh) for wh in args.resolution.lower().split('x'))
    anti_alias = args.anti_alias_samples != 0

    viewer.start_viewer(
        args.file_path,
        width,
        height,
        args.poll_frequency,
        anti_alias,
        args.anti_alias_samples,
        args.assets,
        args.fps,
    )
