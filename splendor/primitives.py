import math
import copy

import numpy

def make_primitive(shape, **kwargs):
    primitive_functions = {
        'mesh_grid' : mesh_grid,
        'disk' : disk,
        'cube' : cube,
        'barrel' : barrel,
        'multi_cylinder' : multi_cylinder,
        'sphere' : sphere}
    
    return primitive_functions[shape](**kwargs)

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
    flatten = True,
    uv_max = None,
    flip_u=False,
    flip_v=False,
):
    
    # how many vertices will there be in x and y
    x_vertices = x_divisions+2
    y_vertices = y_divisions+2
    
    # get the x_range (x_max - x_min) and the spacing for each x_vertex
    if x_extents is not None:
        x_range = x_extents[1] - x_extents[0]
        x_spacing = [(i/(x_vertices-1))*x_range + x_extents[0]
                for i in range(x_vertices)]
    elif x_spacing is not None:
        x_range = x_spacing[-1] - x_spacing[0]
    else:
        raise Exception(
                'Either x_extents or x_spacing argument must be supplied')

    # get the x_range (x_max - x_min) and the spacing for each x_vertex
    if y_extents is not None:
        y_range = y_extents[1] - y_extents[0]
        y_spacing = [(i/(y_vertices-1))*y_range + y_extents[0]
                for i in range(y_vertices)]
    elif y_spacing is not None:
        y_range = y_spacing[-1] - y_spacing[0]
    else:
        raise Exception(
                'Either y_extents or y_spacing argument must be supplied')
    
    x_quads = x_divisions+1
    y_quads = y_divisions+1
    total_vertices = x_vertices * y_vertices
    vertices = numpy.zeros((4, x_vertices, y_vertices))
    normals = numpy.zeros((4, x_vertices, y_vertices))
    uvs = numpy.zeros((3, x_vertices, y_vertices))
    total_faces = x_quads * y_quads * 2
    faces = numpy.zeros((3, total_faces), dtype=int)
    normal_axis, = {0,1,2} - set(axes)
    if uv_max is None:
        uv_max = max(x_range, y_range)
    uv_scale = 1./uv_max
    
    max_u = (x_spacing[-1] - x_spacing[0])
    for i in range(x_vertices):
        vertices[axes[0], i, :] = x_spacing[i]
        if flip_u:
            u = (max_u - (x_spacing[i] - x_spacing[0])) * uv_scale
        else:
            u = (x_spacing[i] - x_spacing[0]) * uv_scale
        uvs[0, i, :] = u
    
    max_v = (y_spacing[-1] - y_spacing[0])
    for i in range(y_vertices):
        vertices[axes[1], :, i] = y_spacing[i]
        if flip_v:
            v = (max_v - (y_spacing[i] - y_spacing[0])) * uv_scale
        else:
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
    flip_face=False
    if tuple(axes) in ((2,1), (1,0), (0,2)):
        flip_face=True
    for i in range(x_quads):
        for j in range(y_quads):

            # make two faces
            ai = i
            aj = j
            if flip_face:
                bi = i+1
                bj = j
                ci = i
                cj = j+1
            else:
                bi = i
                bj = j+1
                ci = i+1
                cj = j
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
        vertices = vertices.reshape(4, total_vertices).T
        normals = normals.reshape(4, total_vertices).T
        uvs = uvs.reshape(3, total_vertices).T
    
    return {'vertices':vertices,
            'normals':normals,
            'uvs':uvs,
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

def compute_bezel_spacing(extents, divisions, bezel):
    primary_vertices = divisions + 2
    total_vertices = primary_vertices + 2
    total_range = extents[1] - extents[0]
    primary_range = total_range - bezel * 2
    primary_start = extents[0] + bezel
    spacing = [float(extents[0])] + [
        (i/(primary_vertices-1)) * primary_range + primary_start
        for i in range(primary_vertices)
    ] + [float(extents[1])]
    
    return spacing

def rectangle(
    x_extents = (-1,1),
    y_extents = (-1,1),
    x_divisions = 0,
    y_divisions = 0,
    depth = 0,
    axes = (0,1),
    flip_normals = False,
    bezel = None,
    uv_max = None,
    flip_u=False,
    flip_v=False,
):
    if not bezel:
        grid = mesh_grid(
            axes=axes,
            x_extents=x_extents,
            y_extents=y_extents,
            x_divisions=x_divisions,
            y_divisions=y_divisions,
            depth=depth,
            flip_normals=flip_normals,
            flatten=True,
            uv_max=uv_max,
            flip_u=flip_u,
            flip_v=flip_v,
        )
    
    else:
        x_spacing = compute_bezel_spacing(x_extents, x_divisions, bezel)
        y_spacing = compute_bezel_spacing(y_extents, y_divisions, bezel)
        grid = mesh_grid(
            axes=axes,
            x_spacing=x_spacing,
            y_spacing=y_spacing,
            x_divisions=x_divisions+2,
            y_divisions=y_divisions+2,
            depth=depth,
            flip_normals=flip_normals,
            flatten=False,
            uv_max=uv_max,
            flip_u=flip_u,
            flip_v=flip_v,
        )
        
        half = 1./(2.**0.5)
        third = 1./(3.**0.5)
        
        normal_axis, = {0,1,2} - set(axes)
        
        grid['normals'][axes[1],1:-1, 0] = -half
        grid['normals'][axes[1],1:-1,-1] =  half
        grid['normals'][axes[0], 0,1:-1] = -half
        grid['normals'][axes[0],-1,1:-1] =  half
        grid['normals'][normal_axis,1:-1, 0] *= half
        grid['normals'][normal_axis,1:-1,-1] *= half
        grid['normals'][normal_axis, 0,1:-1] *= half
        grid['normals'][normal_axis,-1,1:-1] *= half
        
        grid['normals'][axes, 0, 0] = [-third,-third]
        grid['normals'][axes,-1, 0] = [ third,-third]
        grid['normals'][axes, 0,-1] = [-third, third]
        grid['normals'][axes,-1,-1] = [ third, third]
        grid['normals'][normal_axis, 0, 0] *= third
        grid['normals'][normal_axis,-1, 0] *= third
        grid['normals'][normal_axis, 0,-1] *= third
        grid['normals'][normal_axis,-1,-1] *= third
        
        if flip_normals:
            direction = -1
        else:
            direction = 1
        edge = 1. - half
        corner = 1. - 0.546918160678027
        grid['vertices'][axes[1],1:-1, 0] += edge * bezel
        grid['vertices'][axes[1],1:-1,-1] -= edge * bezel
        grid['vertices'][axes[0], 0,1:-1] += edge * bezel
        grid['vertices'][axes[0],-1,1:-1] -= edge * bezel
        grid['vertices'][normal_axis,1:-1, 0] += edge * bezel * direction
        grid['vertices'][normal_axis,1:-1,-1] += edge * bezel * direction
        grid['vertices'][normal_axis, 0,1:-1] += edge * bezel * direction
        grid['vertices'][normal_axis,-1,1:-1] += edge * bezel * direction
        
        grid['vertices'][axes, 0, 0] += [ corner * bezel, corner * bezel]
        grid['vertices'][axes,-1, 0] += [-corner * bezel, corner * bezel]
        grid['vertices'][axes, 0,-1] += [ corner * bezel,-corner * bezel]
        grid['vertices'][axes,-1,-1] += [-corner * bezel,-corner * bezel]
        grid['vertices'][normal_axis, 0, 0] += corner*bezel*direction
        grid['vertices'][normal_axis,-1, 0] += corner*bezel*direction
        grid['vertices'][normal_axis, 0,-1] += corner*bezel*direction
        grid['vertices'][normal_axis,-1,-1] += corner*bezel*direction
        
        total_vertices = (x_divisions+4) * (y_divisions+4)
        grid['vertices'] = grid['vertices'].reshape(4, total_vertices).T
        grid['normals'] = grid['normals'].reshape(4, total_vertices).T
        grid['uvs'] = grid['uvs'].reshape(3, total_vertices).T
    
    return grid

def cube(
        x_extents = (-1,1),
        y_extents = (-1,1),
        z_extents = (-1,1),
        x_divisions = 0,
        y_divisions = 0,
        z_divisions = 0,
        bezel = None):
    
    x_range = x_extents[1] - x_extents[0]
    y_range = y_extents[1] - y_extents[0]
    z_range = z_extents[1] - z_extents[0]
    uv_max = max(y_range+z_range*2, x_range*2+z_range*2)
    
    neg_x = rectangle(
        x_extents = z_extents,
        y_extents = y_extents,
        x_divisions = y_divisions,
        y_divisions = z_divisions,
        depth = x_extents[0],
        axes = (2,1),
        flip_normals=False,
        bezel=bezel,
        uv_max=uv_max,
    )
    neg_x_uv_transform = numpy.array([
        [1, 0, x_range/uv_max],
        [0, 1, z_range/uv_max],
        [0, 0, 1]
    ])
    neg_x['uvs'] = (neg_x_uv_transform @ neg_x['uvs'].T).T
    
    pos_x = rectangle(
        x_extents = z_extents,
        y_extents = y_extents,
        x_divisions = y_divisions,
        y_divisions = z_divisions,
        depth = x_extents[1],
        axes = (2,1),
        flip_normals=True,
        bezel=bezel,
        uv_max=uv_max,
        flip_u=True
    )
    pos_x_uv_transform = numpy.array([
        [1, 0, (x_range*2+z_range)/uv_max],
        [0, 1, z_range/uv_max],
        [0, 0, 1]
    ])
    pos_x['uvs'] = (pos_x_uv_transform @ pos_x['uvs'].T).T
    
    neg_y = rectangle(
        x_extents = x_extents,
        y_extents = z_extents,
        x_divisions = z_divisions,
        y_divisions = x_divisions,
        depth = y_extents[0],
        axes = (0,2),
        flip_normals=False,
        bezel=bezel,
        uv_max=uv_max,
        flip_v=True
    )
    
    pos_y = rectangle(
        x_extents = x_extents,
        y_extents = z_extents,
        x_divisions = z_divisions,
        y_divisions = x_divisions,
        depth = y_extents[1],
        axes = (0,2),
        flip_normals=True,
        bezel=bezel,
        uv_max=uv_max,
        flip_u=True
    )
    pos_y_uv_transform = numpy.array([
        [1, 0, 0],
        [0, 1, (y_range+z_range)/uv_max],
        [0, 0, 1]
    ])
    pos_y['uvs'] = (pos_y_uv_transform @ pos_y['uvs'].T).T
    
    neg_z = rectangle(
        x_extents = x_extents,
        y_extents = y_extents,
        x_divisions = x_divisions,
        y_divisions = y_divisions,
        depth = z_extents[0],
        axes = (0,1),
        flip_normals=False,
        bezel=bezel,
        uv_max=uv_max,
        flip_u=True,
    )
    neg_z_uv_transform = numpy.array([
        [1, 0, 0],
        [0, 1, z_range/uv_max],
        [0, 0, 1]
    ])
    neg_z['uvs'] = (neg_z_uv_transform @ neg_z['uvs'].T).T
    
    pos_z = rectangle(
        x_extents = x_extents,
        y_extents = y_extents,
        x_divisions = x_divisions,
        y_divisions = y_divisions,
        depth = z_extents[1],
        axes = (0,1),
        flip_normals=True,
        bezel=bezel,
        uv_max=uv_max,
    )
    pos_z_uv_transform = numpy.array([
        [1, 0, (x_range + z_range)/uv_max],
        [0, 1, z_range/uv_max],
        [0, 0, 1]
    ])
    pos_z['uvs'] = (pos_z_uv_transform @ pos_z['uvs'].T).T
    
    return merge_meshes([neg_x, pos_x, neg_y, pos_y, neg_z, pos_z])

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

def cylinder(
    start_height,
    end_height,
    radius,
    radial_resolution=16,
    start_cap=False,
    end_cap=False,
):
    return multi_cylinder(
        start_height=start_height,
        sections=((radius, end_height),),
        radial_resolution=radial_resolution,
        start_cap=start_cap,
        end_cap=end_cap)

def multi_cylinder(
    start_height = 0,
    sections = ((1, 1),),
    radial_resolution = 16,
    start_cap = False,
    middle_caps = False,
    end_cap = False,
):
    
    previous_height = start_height
    barrel_segments = []
    for i, (radius, height) in enumerate(sections):
        barrel_segment = barrel(
                height_extents = (previous_height, height),
                radius = radius,
                radial_resolution = radial_resolution)
        barrel_segments.append(barrel_segment)
        previous_height = height
    
    caps = []
    if start_cap:
        cap = disk(
                radius = sections[0][0],
                radial_resolution = radial_resolution,
                flip_normals = False)
        cap['vertices'][:,1] = start_height
        caps.append(cap)
    if end_cap:
        cap = disk(
                radius = sections[-1][0],
                radial_resolution = radial_resolution,
                flip_normals = True)
        cap['vertices'][:,1] = previous_height
        caps.append(cap)
    
    return merge_meshes(barrel_segments + caps)

def sphere(
        height_extents = (-1, 1),
        radius = 1,
        theta_extents = (0, math.pi*2),
        height_divisions = 16,
        radial_resolution = 16):
    
    mesh = barrel(
        height_extents = height_extents,
        radius = 1,
        theta_extents = theta_extents,
        height_divisions = height_divisions,
        radial_resolution = radial_resolution)
    
    row_vertices = height_divisions + 2
    for r in range(row_vertices):
        theta = r/(row_vertices-1) * math.pi
        xz = math.sin(theta)
        y = math.cos(theta)
        
        mesh['vertices'][r::row_vertices,0] *= xz
        mesh['vertices'][r::row_vertices,1] = y
        mesh['vertices'][r::row_vertices,2] *= xz
        
        mesh['normals'][r::row_vertices,0] = mesh['vertices'][r::row_vertices,0]
        mesh['normals'][r::row_vertices,1] = mesh['vertices'][r::row_vertices,1]
        mesh['normals'][r::row_vertices,2] = mesh['vertices'][r::row_vertices,2]
        
    
    return mesh
