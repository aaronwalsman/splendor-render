import math
import copy

import numpy

def merge_meshes(meshes):
    merged_mesh = copy.deepcopy(meshes[0])
    for mesh in meshes[1:]:
        vertex_offset = len(merged_mesh['vertices'])
        merged_mesh['vertices'] = numpy.concatenate(
                (merged_mesh['vertices'], mesh['vertices']), axis=0)
        merged_mesh['normals'] = numpy.concatenate(
                (merged_mesh['normals'], mesh['normals']), axis=0)
        merged_mesh['uvs'] = numpy.concatenate(
                (merged_mesh['uvs'], mesh['uvs']), axis=0)
        merged_mesh['faces'] = numpy.concatenate(
                (merged_mesh['faces'], mesh['faces'] + vertex_offset), axis=0)
    
    return merged_mesh

def mesh_grid(
        axes,
        x_divisions,
        y_divisions,
        x_extents = None,
        y_extents = None,
        x_spacing = None,
        y_spacing = None,
        depth = 0,
        flip_normals = False,
        flatten = True):

    x_vertices = x_divisions+2
    y_vertices = y_divisions+2

    if x_extents is not None:
        x_range = x_extents[1] - x_extents[0]
        x_spacing = [(i/(x_vertices-1))*x_range + x_extents[0]
                for i in range(x_vertices)]
    elif x_spacing is not None:
        x_range = x_spacing[-1] - x_spacing[0]
    else:
        raise Exception(
                'Either x_range or x_spacing argument must be supplied')

    if y_extents is not None:
        y_range = y_extents[1] - y_extents[0]
        y_spacing = [(i/(y_vertices-1))*y_range + y_extents[0]
                for i in range(y_vertices)]
    elif y_spacing is not None:
        y_range = y_spacing[-1] - y_spacing[0]
    else:
        raise Exception(
                'Either y_range or y_spacing argument must be supplied')

    x_quads = x_divisions+1
    y_quads = y_divisions+1
    total_vertices = x_vertices * y_vertices
    vertices = numpy.zeros((4, x_vertices, y_vertices))
    normals = numpy.zeros((4, x_vertices, y_vertices))
    uvs = numpy.zeros((3, x_vertices, y_vertices))
    total_faces = x_quads * y_quads * 2
    faces = numpy.zeros((3, total_faces), dtype=int)
    normal_axis, = {0,1,2} - set(axes)
    uv_scale = 1./max(x_range, y_range)

    for i in range(x_vertices):
        vertices[axes[0], i, :] = x_spacing[i]
        u = (x_spacing[i] - x_spacing[0]) * uv_scale
        uvs[0, i, :] = u

    for i in range(y_vertices):
        vertices[axes[1], :, i] = y_spacing[i]
        v = (y_spacing[i] - y_spacing[0]) * uv_scale
        uvs[1, :, i] = v

    vertices[normal_axis] = depth
    vertices[3] = 1
    uvs[2] = 1
    
    if flip_normals:
        normals[normal_axis,:] = 1
    else:
        normals[normal_axis,:] = -1

    # faces
    for i in range(x_quads):
        for j in range(y_quads):

            # make two faces
            ai = i
            aj = j
            bi = i+1
            bj = j
            ci = i
            cj = j+1
            di = i+1
            dj = j+1

            face_index_0 = (i * y_quads + j)*2
            face_index_1 = face_index_0 + 1
            if flip_normals:
                faces[:,face_index_0] = [
                        ci * y_vertices + cj,
                        bi * y_vertices + bj,
                        ai * y_vertices + aj]
                faces[:,face_index_1] = [
                        ci * y_vertices + cj,
                        di * y_vertices + dj,
                        bi * y_vertices + bj]
            else:
                faces[:,face_index_0] = [
                        ai * y_vertices + aj,
                        bi * y_vertices + bj,
                        ci * y_vertices + cj]
                faces[:,face_index_1] = [
                        bi * y_vertices + bj,
                        di * y_vertices + dj,
                        ci * y_vertices + cj]

    if flatten:
        vertices = vertices.reshape(4, total_vertices)
        normals = normals.reshape(4, total_vertices)
        uvs = uvs.reshape(3, total_vertices)

    return {'vertices':vertices.T,
            'normals':normals.T,
            'uvs':uvs.T,
            'faces':faces.T}

def disk(
        radius = 1,
        inner_radius = 0.,
        theta_extents = None,
        radial_resolution = 16,
        flip_normals = False):
    
    inner_radius_ratio = inner_radius / (inner_radius + radius)
    
    if theta_extents is None:
        partial = False
        theta_extents = (0, math.pi * 2)
        radial_vertices = radial_resolution
    else:
        partial = True
        radial_vertices = radial_resolution + 1
    
    if inner_radius:
        num_faces = radial_resolution * 2
        inner_vertices = radial_vertices
    else:
        num_faces = radial_resolution
        inner_vertices = 1
    
    total_vertices = radial_vertices + inner_vertices
    vertices = numpy.zeros((4, total_vertices))
    normals = numpy.zeros((4, total_vertices))
    uvs = numpy.zeros((3, total_vertices))
    faces = numpy.zeros((3, num_faces), dtype=numpy.long)
    
    theta_range = theta_extents[1] - theta_extents[0]
    vertices[1,:] = 0.
    vertices[3,:] = 1.
    
    if not inner_radius:
        uvs[:,total_vertices-1] = [0.5, 0.5, 1]
    uvs[2,:] = 1.
    
    for i in range(radial_vertices):
        # make one radial vertex
        theta = float(i) / radial_resolution
        theta = theta * theta_range + theta_extents[0]
        vertices[0,i] = math.cos(theta) * radius
        vertices[2,i] = math.sin(theta) * radius
        
        uvs[0,i] = (math.cos(theta) + 1) * 0.5
        uvs[1,i] = (math.sin(theta) + 1) * 0.5
        
        if inner_radius:
            # make one inner vertex
            vertices[0,radial_vertices+i] = math.cos(theta) * inner_radius
            vertices[2,radial_vertices+i] = math.sin(theta) * inner_radius
            
            uvs[0,i] = (math.cos(theta) + 1) * 0.5 * inner_radius_ratio
            uvs[1,i] = (math.sin(theta) + 1) * 0.5 * inner_radius_ratio
            
            # make two faces
            a = i
            b = i+1
            c = radial_vertices + i
            d = radial_vertices + i + 1
            if partial:
                if i != radial_vertices-1:
                    if flip_normals:
                        faces[:,i] = [a,b,c]
                        faces[:,radial_resolution+i] = [b,d,c]
                    else:
                        faces[:,i] = [c,b,a]
                        faces[:,radial_resolution+i] = [c,d,b]
        
        else:
            # make one face
            a = i
            b = i+1
            c = radial_vertices
            if partial:
                if i != radial_vertices-1:
                    if flip_normals:
                        faces[:,i] = [a,b,c]
                    else:
                        faces[:,i] = [c,b,a]
            else:
                b = b % radial_vertices
                if flip_normals:
                    faces[:,i] = [a,b,c]
                else:
                    faces[:,i] = [c,b,a]
    
    if flip_normals:
        normals[:,:] = [[0],[-1], [0], [0]]
    else:
        normals[:,:] = [[0], [1], [0], [0]]
    
    return {'vertices':vertices.T,
            'normals':normals.T,
            'uvs':uvs.T,
            'faces':faces.T}

def cube(
        x_extents = (-1,1),
        y_extents = (-1,1),
        z_extents = (-1,1),
        x_divisions = 0,
        y_divisions = 0,
        z_divisions = 0,
        bezel = None):
    
    raise NotImplementedError

def barrel(
        height_extents = (-1, 1),
        radius = 1,
        theta_extents = (0, math.pi*2),
        height_divisions = 0,
        radial_resolution = 16):
    
    mesh = mesh_grid(
            axes = [0,1],
            x_divisions = radial_resolution-1,
            y_divisions = height_divisions,
            x_extents = [0,1],
            y_extents = height_extents,
            x_spacing = None,
            y_spacing = None,
            depth = 0,
            flip_normals = True,
            flatten = True)
    
    columns = radial_resolution + 1
    theta_range = theta_extents[1] - theta_extents[0]
    column_vertices = height_divisions + 2
    for c in range(columns):
        theta = c / radial_resolution * theta_range + theta_extents[0]
        mesh['vertices'][c*column_vertices:(c+1)*column_vertices,0] = (
                math.cos(theta) * radius)
        mesh['vertices'][c*column_vertices:(c+1)*column_vertices,2] = (
                math.sin(theta) * radius)
        mesh['normals'][c*column_vertices:(c+1)*column_vertices,0] = (
                math.cos(theta))
        mesh['normals'][c*column_vertices:(c+1)*column_vertices,2] = (
                math.sin(theta))
    
    return mesh

def multi_cylinder(
        start_height = 0,
        sections = ((1, 1),),
        radial_resolution = 16):
    
    previous_height = start_height
    barrel_segments = []
    for radius, height in sections:
        barrel_segment = barrel(
                height_extents = (previous_height, height),
                radius = radius,
                radial_resolution = radial_resolution)
        barrel_segments.append(barrel_segment)
        previous_height = height
    
    return merge_meshes(barrel_segments)

def cylinder(
        height_extents = (-1, 1),
        radius = 1,
        theta_extents = (0, math.pi*2),
        height_divisions = 0,
        radial_resolution = 16,
        make_bottom_cap = True,
        make_top_cap = True):
   
    pass
