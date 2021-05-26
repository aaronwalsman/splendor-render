import os
import shutil

class MeshError(Exception):
    pass

def copy_rename_obj(
        obj_path,
        mtl_path,
        tex_path,
        destination,
        new_name):
    
    '''
    Copies an obj file from one location to another along with it's mtl
    and texture file, and changes internal paths appropriately.
    '''
    
    mtl_file_name = os.path.basename(mtl_path)
    tex_file_name = os.path.basename(tex_path)
    tex_file_extension = tex_file_name.split('.')[-1]
    tex_new_name = new_name + '.' + tex_file_extension
    
    with open(obj_path, 'r') as f:
        obj_data = f.read()
        obj_data = obj_data.replace(
                'mtllib %s'%mtl_file_name,
                'mtllib %s.mtl'%new_name)
        obj_data = obj_data.replace(
                'mtllib ./%s'%mtl_file_name,
                'mtllib %s.mtl'%new_name)
        with open(os.path.join(destination, new_name + '.obj'), 'w') as g:
            g.write(obj_data)
    
    with open(mtl_path, 'r') as f:
        mtl_data = f.read()
        mtl_data = mtl_data.replace(tex_file_name, tex_new_name)
        with open(os.path.join(destination, new_name + '.mtl'), 'w') as g:
            g.write(mtl_data)
    
    shutil.copy2(tex_path, os.path.join(destination, tex_new_name))

def load_mesh(mesh_path, strict=False, scale=1.0):
    
    '''
    Loads an obj file to a mesh dictionary.  This does not support all obj
    features, but supports the ones necessary for splendor-render.
    '''
    
    try:
        obj_vertices = []
        obj_normals = []
        obj_uvs = []
        obj_faces = []
        obj_vertex_colors = []
        
        mesh = {
            'vertices':[],
            'normals':[],
            'uvs':[],
            'vertex_colors':[],
            'faces':[]}
        
        vertex_face_mapping = {}
        
        with open(mesh_path) as f:
            for line in f:
                tokens = line.strip().split()
                if not tokens:
                    continue
                if tokens[0] == 'v':
                    # add a vertex
                    if len(tokens) != 4 and len(tokens) != 7:
                        if strict:
                            raise MeshError(
                                    'Vertex must have exactly three '
                                    'or six elements')
                    obj_vertices.append(
                            [float(xyz) * scale for xyz in tokens[1:4]])
                    if len(tokens) > 4:
                        obj_vertex_colors.append(
                                [float(rgb) for rgb in tokens[4:7]])
                
                if tokens[0] == 'vt':
                    # add a uv
                    if len(tokens) != 3 and len(tokens) != 4:
                        raise MeshError(
                                'UV must have two or three elements')
                    obj_uvs.append([float(uv) for uv in tokens[1:3]])
                
                if tokens[0] == 'vn':
                    # add a normal
                    if len(tokens) != 4:
                        raise MeshError(
                                'Normal must have exactly three elements')
                    obj_normals.append([float(xyz) for xyz in tokens[1:]])
                
                if tokens[0] == 'f':
                    # add a face
                    #if len(tokens) != 4:
                    #    raise MeshError(
                    #            'Only triangle meshes are supported')
                    part_groups = tokens[1:]
                    if len(part_groups) < 3:
                        raise MeshError(
                                'A face must contain at least three vertices')
                    # triangulate
                    for i in range(len(part_groups)-2):
                        face = []
                        face_id = len(obj_faces)
                        triangle_part_groups = (
                            part_groups[0],
                            part_groups[i+1],
                            part_groups[i+2])
                        for j, part_group in enumerate(triangle_part_groups):
                            face_parts = part_group.split('/')
                            if len(face_parts) == 1:
                                face_parts = face_parts * 3
                            if len(face_parts) != 3:
                                raise MeshError(
                                        'Each face must contain an vertex, '
                                        'uv and normal')
                            if face_parts[1] == '':
                                face_parts[1] = 0
                            face_parts = [int(part)-1 for part in face_parts]
                            face.append(face_parts)
                            
                            vertex, uv, normal = face_parts
                            vertex_face_mapping.setdefault(vertex, [])
                            vertex_face_mapping[vertex].append((face_id,j))
                        obj_faces.append(face)
        
        # break up the mesh so that all vertices have the same uv and normal
        for vertex_id, vertex in enumerate(obj_vertices):
            if vertex_id not in vertex_face_mapping:
                if strict:
                    raise MeshError('Vertex %i is not used in any faces'%i)
                else:
                    continue
            
            # find out how many splits need to be made by going through all
            # faces this vertex is used in and finding which normals and uvs
            # are associated with it
            face_combo_lookup = {}
            for face_id, corner_id in vertex_face_mapping[vertex_id]:
                corner = obj_faces[face_id][corner_id]
                combo = corner[1], corner[2]
                face_combo_lookup.setdefault(combo, [])
                face_combo_lookup[combo].append((face_id, corner_id))
            
            for combo in face_combo_lookup:
                uv_id, normal_id = combo
                new_vertex_id = len(mesh['vertices'])
                # fix the mesh faces
                for face_id, corner_id in face_combo_lookup[combo]:
                    obj_faces[face_id][corner_id] = [
                            new_vertex_id, new_vertex_id, new_vertex_id]
                
                mesh['vertices'].append(vertex)
                if len(obj_uvs):
                    mesh['uvs'].append(obj_uvs[uv_id])
                mesh['normals'].append(obj_normals[normal_id])
                if len(obj_vertex_colors):
                    mesh['vertex_colors'].append(obj_vertex_colors[vertex_id])
        
        mesh['faces'] = [[corner[0] for corner in face] for face in obj_faces]
    
    except:
        print('Failed to load %s'%mesh_path)
        raise
    
    return mesh

def write_obj(mesh, obj_path, texture_name=None):
    
    '''
    Write a mesh dictionary to an obj file.
    '''
    
    v = mesh['vertices']
    n = mesh['normals']
    u = mesh['uvs']
    f = mesh['faces']

    obj_name = os.path.basename(obj_path)
    base_path, _ = os.path.splitext(obj_path)
    mtl_path = base_path + '.mtl'
    mtl_name = os.path.basename(mtl_path)
    if texture_name is None:
        texture_path = base_path + '.png'
        texture_name = os.path.basename(texture_path)
    with open(obj_path, 'w') as obj:
        obj.write('mtllib %s\n'%mtl_name)
        for i in range(v.shape[0]):
            obj.write('vn %f %f %f\n'%(n[i,0], n[i,1], n[i,2]))
            obj.write('v %f %f %f\n'%(v[i,0], v[i,1], v[i,2]))
        for i in range(u.shape[0]):
            obj.write('vt %f %f\n'%(u[i,0], u[i,1]))
        #for i in range(n.shape[0]):
        #    obj.write('vn %f %f %f\n'%(n[i,0], n[i,1], n[i,2]))
        for i in range(f.shape[0]):
            obj.write('f %i/%i/%i %i/%i/%i %i/%i/%i\n'%(
                    f[i,0]+1, f[i,0]+1, f[i,0]+1,
                    f[i,1]+1, f[i,1]+1, f[i,1]+1,
                    f[i,2]+1, f[i,2]+1, f[i,2]+1))

    with open(mtl_path, 'w') as mtl:
        mtl.write('newmtl singleShader\n')
        mtl.write('illum 4\n')
        mtl.write('Kd 1.00 1.00 1.00\n')
        mtl.write('Ka 0.00 0.00 0.00\n')
        mtl.write('Tf 1.00 1.00 1.00\n')
        mtl.write('Ni 1.00\n')
        mtl.write('map_Kd %s\n'%texture_name)

def triangulate_obj(in_path, out_path):
    with open(os.path.expanduser(in_path)) as f:
        lines = []
        for line in f:
            tokens = line.strip().split()
            if not tokens:
                lines.append(line)
                continue
            if tokens[0] == 'f':
                if len(tokens) == 4:
                    lines.append(line)
                else:
                    for i in range(len(tokens)-3):
                        lines.append('f %s %s %s\n'%(
                                tokens[1], tokens[i+2], tokens[i+3]))
            else:
                lines.append(line)

    out = ''.join(lines)
    with open(os.path.expanduser(out_path), 'w') as f:
        f.write(out)

