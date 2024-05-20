import numpy as np

from OpenGL.arrays import vbo

from splendor.named_asset import NamedAsset

def load_mesh_asset(name, asset_name, albedo='FLAT'):
    asset_path = asset_library['mesh'][asset_name]
    return Mesh(data, name=asset_name, albedo=albedo)


class Mesh(NamedAsset):
    
    ASSET_CLASS_NAME = 'mesh'
    SUPPORTED_ALBEDOS = ('FLAT', 'VERTEX_COLOR', 'TEXTURED')
    
    def __init__(self,
        name=None,
        mesh=None,
        albedo='FLAT',
    ):
        # verify/set albedo mode
        #assert albedo in self.SUPPORTED_ALBEDOS, (
        #    f'albedo must be in {self.SUPPORTED_ALBEDOS}')
        #self.albedo = albedo
        
        # set internal data and register the name
        super().__init__(
            name=name,
            mesh=mesh,
            albedo=albedo,
        )
        
        # initialize vertex/face buffers
        self.vertex_buffer = vbo.VBO(np.zeros(0, dtype=np.float32))
        self.face_buffer = vbo.VBO(
            np.zeros(0, dtype=np.int32),
            target=GL.GL_ELEMENT_ARRAY_BUFFER
        )
    
    @staticmethod
    def validate_data(data):
        validated = {}
        
        # vertices
        # assert vertices exist
        assert 'vertices' in data, 'mesh must have "vertices"'
        vertices = data['vertices']
        # assert Nx3 shape
        assert len(vertices.shape) == 2 and vertices.shape[1] == 3, (
            'mesh vertices must have shape Nx3')
        # convert to float32
        validated['vertices'] = np.array(vertices, dtype=np.float32)
        
        # faces
        # assert faces exist
        assert 'faces' in data, 'mesh must have "faces"'
        faces = data['faces']
        # assert Nx3 shape
        assert len(faces.shape) == 2 and faces.shape[1] == 3, (
            'mesh faces must have shape Nx3')
        
        # normals
        # assert normals exist
        assert 'normals' in data, 'mesh must have normals'
        normals = data['normals']
        # assert Nx3 shape
        assert len(normals.shape) == 2 and normals.shape[1] == 3, (
            'mesh normals must have shape Nx3')
        # convert to float32
        validated['normals'] = np.array(normals, dtype=np.float32)
        
        # uvs
        if 'uvs' in data:
            uvs = data['uvs']
            # assert Nx2 shape
            assert len(uvs.shape) == 2 and uvs.shape[1] == 2, (
                'mesh uvs must have shape Nx2')
            # convert to float32
            validated['uvs'] = np.array(uvs, dtype=np.float32)
        
        # vertex color
        if 'vertex_color' in data:
            vertex_color = data['vertex_color']
            # assert Nx3 shape
            assert (len(vertex_color.shape) == 2 and
                vertex_color.shape[1] == 3), (
                'mesh vertex_color must have shape Nx3')
            # convert to float32
            validated['vertex_color'] = np.array(
                vertex_color, dtype=np.float32)
        
        return validated
    
    def update_gl_data(self):
        if self.albedo == 'FLAT':
            combined_floats = np.concatenate(
                self.data['vertices'],
                self.data['normals'],
                axis=1
            )
        elif self.albedo == 'VERTEX_COLOR':
            combined_floats = np.concatenate(
                self.data['vertices'],
                self.data['normals'],
                self.data['vertex_color'],
                axis=1,
            )
        elif self.albedo == 'TEXTURED':
            combined_floats = np.concatenate(
                self.data['vertices'],
                self.data['normals'],
                self.data['uvs'],
                axis=1,
            )
        
        self.vertex_buffer.set_array(combined_floats)
        self.face_buffer.set_array(self.data['faces'])
    
    def cleanup_gl_data(self):
        self.vertex_buffer.delete()
        self.face_buffer.delete()
    
    @property
    def vertex_stride(self):
        if self.albedo == 'FLAT':
            return (3+3) * 4
        elif self.albedo == 'VERTEX_COLOR':
            return (3+3+3) * 4
        elif self.albedo == 'TEXTURED':
            return (3+3+2) * 4
