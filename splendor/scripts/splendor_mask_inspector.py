#!/usr/bin/env python
import argparse

import splendor.mask_viewer as mask_viewer

parser = argparse.ArgumentParser(description='Splendor Mask Viewer')
parser.add_argument('file_path', type=str,
        help='json file containing scene data')

def main():
    args = parser.parse_args()
    mask_viewer.start_viewer(args.file_path)
