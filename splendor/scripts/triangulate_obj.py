#!/usr/bin/env python
"""CLI for triangulating an OBJ mesh file."""
import sys
from splendor.obj_mesh import triangulate_obj

def main():
    """Parse CLI args and triangulate an OBJ file."""
    triangulate_obj(sys.argv[1], sys.argv[2])
