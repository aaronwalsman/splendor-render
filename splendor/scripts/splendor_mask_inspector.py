#!/usr/bin/env python
"""CLI for launching the interactive mask inspector."""
import argparse

import splendor.mask_viewer as mask_viewer

parser = argparse.ArgumentParser(description='Splendor Mask Viewer')
parser.add_argument('file_path', type=str,
        help='json file containing scene data')

def main():
    """Parse CLI args and launch the interactive mask inspector."""
    args = parser.parse_args()
    mask_viewer.start_viewer(args.file_path)
