
class MeshError(Exception):
    pass

def load_mesh(mesh_path):
    
    obj_vertices = []
    obj_normals = []
    obj_uvs = []
    obj_faces = []
    
    mesh = {
        'vertices':[],
        'normals':[],
        'uvs':[],
        'faces':[]}
    
    #vertex_uv_mapping = {}
    #vertex_normal_mapping = {}
    vertex_face_mapping = {}
    
    with open(mesh_path) as f:
        for line in f:
            tokens = line.strip().split()
            if not tokens:
                continue
            if tokens[0] == 'v':
                # add a vertex
                if len(tokens) != 4:
                    raise MeshError('Vertex must have exactly three elements')
                obj_vertices.append([float(xyz) for xyz in tokens[1:]])
            
            if tokens[0] == 'vt':
                # add a uv
                if len(tokens) != 3:
                    raise MeshError('UV must have exactly two elements')
                obj_uvs.append([float(uv) for uv in tokens[1:]])
            
            if tokens[0] == 'vn':
                # add a normal
                if len(tokens) != 4:
                    raise MeshError('Normal must have exactly three elements')
                obj_normals.append([float(xyz) for xyz in tokens[1:]])
            
            if tokens[0] == 'f':
                # add a face
                if len(tokens) != 4:
                    raise MeshError('Only triangle meshes are supported')
                face = []
                face_id = len(obj_faces)
                for i, part_group in enumerate(tokens[1:]):
                    face_parts = part_group.split('/')
                    if len(face_parts) != 3:
                        raise MeshError('Each face must contain an vertex, '
                                'uv and normal')
                    face_parts = [int(face_part)-1 for face_part in face_parts]
                    face.append(face_parts)
                    
                    vertex, uv, normal = face_parts
                    vertex_face_mapping.setdefault(vertex, [])
                    vertex_face_mapping[vertex].append((face_id,i))
                obj_faces.append(face)
                
                #vertex_uv_mapping.setdefault(vertex, [])
                #vertex_normal_mapping.setdefault(vertex, [])
                #vertex_uv_mapping[vertex].append(uv)
                #vertex_normal_mapping[vertex].append(normal)
    
    # break up the mesh so that all vertices have the same uv and normal
    for vertex_id, vertex in enumerate(obj_vertices):
        if vertex_id not in vertex_face_mapping:
            raise MeshError('Vertex %i is not used in any faces'%i)
        
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
            mesh['uvs'].append(obj_uvs[uv_id])
            mesh['normals'].append(obj_normals[normal_id])
    
    mesh['faces'] = [[corner[0] for corner in face] for face in obj_faces]
    
    return mesh
