import numpy as np

from OpenGL.arrays import vbo

from splendor.named_asset import NamedAsset

def load_mesh_asset(name, asset_name):
    asset_path = 

class Mesh(NamedAsset):
    
    SUPPORTED_ALBEDOS = ('FLAT', 'VERTEX_COLOR', 'TEXTURED')
    
    def __init__(self,
        data,
        name=None,
        #asset=None,
        #path=None,
        mesh_data=None,
        #primitive=None,
        albedo='FLAT',
    ):
        super().__init__(data, name=name)
        
        '''
        specified_flags = [
            name if value is not None
            for name, value in (
                ('asset', asset),
                ('path', path),
                ('mesh_data', mesh_data),
                ('primitive', primitive),
            )
        ]
        assert len(specified_flags) == 1, (
            f'please specify exactly one data source, got: {specified_flags}')
        
        self.asset = asset
        self.path = path
        self.mesh_data = mesh_data
        self.primitive = primitive
        
        if asset is not None:
            asset_path = asset_library['meshes'][asset]
            self.mesh_data = load_mesh_data(asset_path)
        if path is not None:
            self.mesh_data = load_mesh_data(path)
        if mesh_data is not None:
            self.mesh_data = self.validate(mesh_data)
        if primitive is not None:
            self.primitive = 
        '''
        
        assert albedo in self.SUPPORTED_ALBEDOS, (
            f'albedo must be in {self.SUPPORTED_ALBEDOS}')
        self.albedo = albedo
    
    @staticmethod
    def validate_data(mesh_data):
        validated = {}
        
        # vertices
        # assert vertices exist
        assert 'vertices' in mesh_data, 'mesh must have "vertices"'
        vertices = mesh_data['vertices']
        # assert Nx3 shape
        assert len(vertices.shape) == 2 and vertices.shape[1] == 3, (
            'mesh vertices must have shape Nx3')
        # convert to float32
        validated['vertices'] = np.array(vertices, dtype=np.float32)
        
        # faces
        # assert faces exist
        assert 'faces' in mesh_data, 'mesh must have "faces"'
        faces = mesh_data['faces']
        # assert Nx3 shape
        assert len(faces.shape) == 2 and faces.shape[1] == 3, (
            'mesh faces must have shape Nx3')
        
        # normals
        # assert normals exist
        assert 'normals' in mesh_data, 'mesh must have normals'
        normals = mesh_data['normals']
        # assert Nx3 shape
        assert len(normals.shape) == 2 and normals.shape[1] == 3, (
            'mesh normals must have shape Nx3')
        # convert to float32
        validated['normals'] = np.array(normals, dtype=np.float32)
        
        # uvs
        if 'uvs' in mesh_data:
            uvs = mesh_data['uvs']
            # assert Nx2 shape
            assert len(uvs.shape) == 2 and uvs.shape[1] == 2, (
                'mesh uvs must have shape Nx2')
            # convert to float32
            validated['uvs'] = np.array(uvs, dtype=np.float32)
        
        # vertex color
        if 'vertex_color' in mesh_data:
            vertex_color = mesh_data['vertex_color']
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
        
        self.vertex_buffer = vbo.VBO(combined_floats)
        self.face_buffer = vbo.VBO(
            self.data['faces'],
            target=GL.GL_ELEMENT_ARRAY_BUFFER
        )
    
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
