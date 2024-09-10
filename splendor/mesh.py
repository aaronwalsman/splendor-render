import numpy as np

import OpenGL.GL as GL
from OpenGL.arrays import vbo

from splendor.named_asset import NamedAsset

class Mesh(NamedAsset):
    
    #SUPPORTED_ALBEDOS = ('FLAT', 'VERTEX_COLOR', 'TEXTURE')
    
    _loaded_assets = {}
    
    _active_mesh = None
    
    def __init__(self,
        name=None,
        asset=None,
        path=None,
        vertices=None,
        normals=None,
        uvs=None,
        faces=None,
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
            np.zeros(0, dtype=np.uint32),
            target=GL.GL_ELEMENT_ARRAY_BUFFER
        )
        #self.vertex_vao = GL.glGenVertexArrays(1)
        #self.vertex_vbo = GL.glGenBuffers(1)
        #self.face_ebo = GL.glGenBuffers(1)
        
        
        # load and validate the mesh_data
        self.load_mesh(
            asset=asset,
            path=path,
            vertices=vertices,
            normals=normals,
            uvs=uvs,
            faces=faces,
        )
    
    def load_mesh(self,
        asset=None,
        path=None,
        vertices=None,
        normals=None,
        uvs=None,
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
                faces is None
            ), 'both "asset" and raw mesh data were specified'
            self.source_type = 'asset'
        elif path is not None:
            assert (
                vertices is None and
                normals is None and
                uvs is None and
                faces is None
            ), 'both "path" and raw mesh data were specified'
            self.source_type = 'path'
        else:
            self.source_type = 'raw'
        
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
        faces = np.array(faces, dtype=np.uint32)
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
        
        self._update_gl_data()
        
    @property
    def vertices(self):
        return self._vertices
    
    @property
    def normals(self):
        return self._normals
    
    @property
    def uvs(self):
        return self._uvs
    
    @property
    def faces(self):
        return self._faces
    
    def _update_gl_data(self):
        # determine how many floats each vertex has
        '''
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
        '''
        
        combined_floats = np.concatenate(
            (self.vertices, self.normals, self.uvs),
            axis=1,
        )
        
        if False:
            # send the vertex floats to opengl
            vertex_bytes = combined_floats.size * combined_floats.itemsize
            self.vertex_buffer.set_array(combined_floats, vertex_bytes)
            self.vertex_buffer.create_buffers()
            self.vertex_buffer.copy_data()
            
            # send the face ints to opengl
            face_bytes = self.faces.size * self.faces.itemsize
            self.face_buffer.set_array(self.faces.reshape(-1), face_bytes)
            self.face_buffer.create_buffers()
            self.face_buffer.copy_data()
        
        self.vertex_buffer = vbo.VBO(combined_floats)
        self.face_buffer = vbo.VBO(
            self.faces,
            target=GL.GL_ELEMENT_ARRAY_BUFFER
        )
        
    def _cleanup_gl_data(self):
        self.vertex_buffer.delete()
        self.face_buffer.delete()
    
    @property
    def vertex_stride(self):
        '''
        if self.albedo == 'FLAT':
            return (3+3) * 4
        elif self.albedo == 'VERTEX_COLOR':
            return (3+3+3) * 4
        elif self.albedo == 'TEXTURE':
            return (3+3+2) * 4
        '''
        return (3+3+2) * 4
    
    def activate(self, shader_locations):
        if not self._active_mesh is not self:
            
            if self._active_mesh is not None:
                self._active_mesh.deactivate()
            
            self.face_buffer.bind()
            self.vertex_buffer.bind()
            
            # TODO: This probably only needs to be run once?
            GL.glEnableVertexAttribArray(location_data['vertex_position'])
            GL.glEnableVertexAttribArray(location_data['vertex_normal'])
            GL.glEnableVertexAttribArray(location_data['vertex_uv'])
            GL.glVertexAttribPointer(
                shader_locations['vertex_position'],
                3,
                GL.GL_FLOAT,
                False,
                self.vertex_stride,
                self.vertex_buffer,
            )
            GL.glVertexAttribPointer(
                shader_locations['vertex_normal'],
                3,
                GL.GL_FLOAT,
                False,
                self.vertex_stride,
                self.vertex_buffer + ((3)*4),
            )
            GL.glVertexAttribPointer(
                shader_locations['vertex_uv'],
                2,
                GL.GL_FLOAT,
                False,
                self.vertex_stride,
                self.vertex_buffer + ((3+3)*4),
            )
            
            '''
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
            '''
            self._active_mesh = self
    
    def deactivate(self, shader_locations):
        self.face_buffer.unbind()
        self.vertex_buffer.unbind()
        self._active_mesh = None
