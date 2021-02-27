#!/usr/bin/env python
import argparse

import renderpy.mask_viewer as mask_viewer

def run():
    parser = argparse.ArgumentParser(description='Renderpy Mask Viewer')
    parser.add_argument('file_path', type=str,
            help='json file containing scene data')

    args = parser.parse_args()

    mask_viewer.start_viewer(args.file_path)
