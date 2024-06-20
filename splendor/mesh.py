import numpy as np

import OpenGL.GL as GL
from OpenGL.arrays import vbo

from splendor.named_asset import NamedAsset

class Mesh(NamedAsset):
    
    SUPPORTED_ALBEDOS = ('FLAT', 'VERTEX_COLOR', 'TEXTURE')
    
    _loaded_assets = {}
    
    _active_mesh = None
    
    def __init__(self,
        name=None,
        asset=None,
        path=None,
        #mesh_data=None,
        vertices=None,
        normals=None,
        uvs=None,
        vertex_colors=None,
        faces=None,
        #albedo='FLAT',
    ):
        
        # if no name was specified, but an asset or path was provided, change
        # the default name
        if name is None:
            if asset is not None:
                name, _ = os.path.splitext(os.path.basename(asset))
            elif path is not None:
                name, _ = os.path.splitext(os.path.basename(path))
        
        # register a unique name for this mesh
        super().__init__(name=name)
        
        # initialize vertex/face buffers
        self.vertex_buffer = vbo.VBO(np.zeros(0, dtype=np.float32))
        self.face_buffer = vbo.VBO(
            np.zeros(0, dtype=np.int32),
            target=GL.GL_ELEMENT_ARRAY_BUFFER
        )
        
        # load and validate the mesh_data
        self.load_mesh(
            asset=asset,
            path=path,
            vertices=vertices,
            normals=normals,
            uvs=uvs,
            vertex_colors=vertex_colors,
            faces=faces,
        )
    
    def load_mesh(self,
        asset=None,
        path=None,
        vertices=None,
        normals=None,
        uvs=None,
        vertex_colors=None,
        faces=None,
    ):
        # make sure the arguments are coherent
        if asset is not None:
            assert path is None, (
                'both "asset" and "path" were specified')
            assert (
                vertices is None and
                normals is None and
                uvs is None and
                vertex_colors is None and
                faces is None
            ), 'both "asset" and raw mesh data were specified'
            self.source_type = 'asset'
        elif path is not None:
            assert (
                vertices is None and
                normals is None and
                uvs is None and
                vertex_colors is None and
                faces is None
            ), 'both "path" and raw mesh data were specified'
            self.source_type = 'path'
        else:
            self.source_type = 'raw'
        
        self.source_type = source_flags[0]
        self.asset = asset
        self.path = path
        
        # load the mesh_data from the provided asset or path if necessary
        if self.source_type in ('asset', 'path'):
            if self.source_type == 'asset':
                asset_path = AssetLibrary['mesh'][asset]
                mesh_data = load_mesh_data(asset_path)
            elif self.source_type == 'path':
                mesh_data = load_mesh_data(path)
            
            vertices = mesh_data.get('vertices', None)
            normals = mesh_data.get('normals', none)
            uvs = mesh_data.get('uvs', none)
            vertex_colors = mesh_data.get('vertex_colors', none)
            faces = mesh_data.get('faces', none)
        
        # validate vertices
        # assert vertices exist
        assert vertices is not None, 'mesh must have "vertices"'
        # convert to float32
        vertices = np.array(vertices, dtype=np.float32)
        # assert Nx3 shape
        assert len(vertices.shape) == 2 and vertices.shape[1] == 3, (
            'mesh vertices must have shape Nx3')
        # make read only
        vertices.setflags(write=False)
        self._vertices = vertices
        
        # validate faces
        # assert faces exist
        assert faces is not None, 'mesh must have "faces"'
        # convert to int
        faces = np.array(faces, dtype=np.int32)
        # assert Nx3 shape
        assert len(faces.shape) == 2 and faces.shape[1] == 3, (
            'mesh faces must have shape Nx3')
        # make read only
        faces.setflags(write=False)
        self._faces = faces
        
        # validate normals
        # assert normals exist
        assert normals is not None, 'mesh must have "normals"'
        # convert to float32
        normals = np.array(normals, dtype=np.float32)
        # assert Nx3 shape
        assert len(normals.shape) == 2 and normals.shape[1] == 3, (
            'mesh normals must have shape Nx3')
        # make read only
        normals.setflags(write=False)
        self._normals = normals
        
        # validate uvs
        if uvs is not None:
            # convert to float32
            uvs = np.array(uvs, dtype=np.float32)
            # assert Nx2 shape
            assert len(uvs.shape) == 2 and uvs.shape[1] == 2, (
                'mesh uvs must have shape Nx2')
            # make read only
            uvs.setflags(write=False)
            self._uvs = uvs
        else:
            self._uvs = None
        
        # vertex color
        if vertex_color is not None:
            # convert to float32
            vertex_color = np.array(vertex_color, dtype=np.float32)
            # assert Nx3 shape
            assert (len(vertex_color.shape) == 2 and
                vertex_color.shape[1] == 3), (
                'mesh vertex_color must have shape Nx3')
            # make read only
            vertex_color.setflags(write=False)
            self._vertex_color = vertex_color
        else:
            self._vertex_color = None
        
    
    '''
    @property
    def mesh_data(self):
        return self._mesh_data
    
    @mesh_data.setter
    def mesh_data(self, mesh_data):
        validated_mesh_data = self.validate_mesh(mesh_data)
        self._mesh_data = validated_mesh_data
        self._update_gl_data()
    '''
    '''
    @property
    def albedo(self):
        return self._albedo
    
    @albedo.setter
    def albedo(self, albedo):
        assert albedo in self.SUPPORTED_ALBEDOS, (
            f'albedo "{albedo}" not {self.SUPPORTED_ALBEDOS}')
        self._albedo = albedo
    '''
    
    '''
    @staticmethod
    def validate_mesh(mesh_data):
        validated = {}
        
        # vertices
        # assert vertices exist
        assert 'vertices' in mesh_data, 'mesh must have "vertices"'
        # convert to float32
        vertices = np.array(mesh_data['vertices'], dtype=np.float32)
        # assert Nx3 shape
        assert len(vertices.shape) == 2 and vertices.shape[1] == 3, (
            'mesh vertices must have shape Nx3')
        # make read only
        vertices.setflags(write=False)
        validated['vertices'] = vertices
        
        # faces
        # assert faces exist
        assert 'faces' in mesh_data, 'mesh must have "faces"'
        # convert to int
        faces = np.array(mesh_data['faces'], dtype=np.int32)
        # assert Nx3 shape
        assert len(faces.shape) == 2 and faces.shape[1] == 3, (
            'mesh faces must have shape Nx3')
        # make read only
        faces.setflags(write=False)
        validated['faces'] = faces
        
        # normals
        # assert normals exist
        assert 'normals' in mesh_data, 'mesh must have normals'
        # convert to float32
        normals = np.array(mesh_data['normals'], dtype=np.float32)
        # assert Nx3 shape
        assert len(normals.shape) == 2 and normals.shape[1] == 3, (
            'mesh normals must have shape Nx3')
        # make read only
        normals.setflags(write=False)
        validated['normals'] = normals
        
        # uvs
        if 'uvs' in mesh_data:
            # convert to float32
            uvs = np.array(mesh_data['uvs'], dtype=np.float32)
            # assert Nx2 shape
            assert len(uvs.shape) == 2 and uvs.shape[1] == 2, (
                'mesh uvs must have shape Nx2')
            # make read only
            uvs.setflags(write=False)
            validated['uvs'] = uvs
        
        # vertex color
        if 'vertex_color' in mesh_data:
            # convert to float32
            vertex_color = np.array(mesh_data['vertex_color'], dtype=np.float32)
            # assert Nx3 shape
            assert (len(vertex_color.shape) == 2 and
                vertex_color.shape[1] == 3), (
                'mesh vertex_color must have shape Nx3')
            # make read only
            vertex_color.setflags(write=False)
            validated['vertex_color'] = vertex_color
        
        return validated
    '''
    
    def _update_gl_data(self):
        # determine how many floats each vertex has
        if self.albedo == 'FLAT':
            combined_floats = np.concatenate(
                (self.mesh_data['vertices'], self.mesh_data['normals']),
                axis=1
            )
        elif self.albedo == 'VERTEX_COLOR':
            combined_floats = np.concatenate(
                (self.mesh_data['vertices'],
                 self.mesh_data['normals'],
                 self.mesh_data['vertex_color']),
                axis=1,
            )
        elif self.albedo == 'TEXTURE':
            combined_floats = np.concatenate(
                (self.mesh_data['vertices'],
                 self.mesh_data['normals'],
                 self.mesh_data['uvs']),
                axis=1,
            )
        
        # send the vertex floats to opengl
        self.vertex_buffer.set_array(combined_floats)
        
        # send the face ints to opengl
        self.face_buffer.set_array(self.mesh_data['faces'])
    
    def _cleanup_gl_data(self):
        self.vertex_buffer.delete()
        self.face_buffer.delete()
    
    @property
    def vertex_stride(self):
        if self.albedo == 'FLAT':
            return (3+3) * 4
        elif self.albedo == 'VERTEX_COLOR':
            return (3+3+3) * 4
        elif self.albedo == 'TEXTURE':
            return (3+3+2) * 4
    
    def activate(self, shader_locations):
        if not self._active_mesh is not self:
            
            if self._active_mesh is not None:
                self._active_mesh.deactivate()
            
            self.face_buffer.bind()
            self.vertex_buffer.bind()
            
            GL.glEnableVertexAttribArray(shader_locations['vertex_position'])
            if 'vertex_normal' in shader_locations:
                GL.glEnableVertexAttribArray(shader_locations['vertex_normal'])
            if 'vertex_position' in shader_locations:
                GL.glVertexAttribPointer(
                    shader_locations['vertex_position'],
                    3,
                    GL.GL_FLOAT,
                    False,
                    self.vertex_stride,
                    self.vertex_buffer,
                )
            if 'vertex_normal' in shader_locations:
                GL.glVertexAttribPointer(
                    shader_locations['vertex_normal'],
                    3,
                    GL.GL_FLOAT,
                    False,
                    self.vertex_stride,
                    self.vertex_buffer + ((3)*4),
                )
            self._active_mesh = self
    
    def deactivate(self, shader_locations):
        self.face_buffer.unbind()
        self.vertex_buffer.unbind()
        self._active_mesh = None
