#!/usr/bin/env python
import sys
from splendor.obj_mesh import triangulate_obj

def main():
    triangulate_obj(sys.argv[1], sys.argv[2])
