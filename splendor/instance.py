import numpy as np

import OpenGL.GL as GL

from splendor.named_asset import NamedAsset
from splendor.mesh import Mesh
from splendor.material import SurfaceMaterial

class Instance(NamedAsset):
    
    _loaded_assets = {}
    
    def __init__(self,
        name=None,
        transform=None,
        scene=None,
    ):
        super().__init__(name=name)
        
        self.transform = transform
    
    @property
    def transform(self):
        return self._transform
    
    @transform.setter
    def transform(self, transform):
        if transform is None:
            transform = np.eye(4, dtype=np.float32)
        assert np.shape(transform) == (4,4)
        self._transform = np.array(transform, dtype=np.float32)
    
    def activate(self, shader_locations):
        GL.glUniformMatrix4fv(
            shader_locations['model_pose'],
            1,
            GL.GL_TRUE,
            self.transform,
        )
    
    #def unload_shader_values(self, shader_locations):
    #    self.material.unload_shader_values(shader_locations)
    #    self.mesh.unload_shader_values(shader_locations)

class InstanceTypeList:
    def __init__(self, instances):
        self._sorted_instances = {}
        for instance in instances:
            class_name = instance.__class__.__name__
            try:
                self._sorted_instances[class_name].append(instance)
            except KeyError:
                self._sorted_instances[class_name] = [instance]
        
        for class_name, instance_list in self._sorted_instances.items():
            instance_list[0].sort_instances(instance_list)
    
    def __iter__(self):
        return iter(self._sorted_instances.values())

class MeshInstance(Instance):
    def __init__(self,
        name=None,
        transform=None,
        scene=None,
        mesh=None,
        material=None,
    ):
        super().__init__(name=name, transform=transform, scene=scene)
        
        self.mesh = mesh
        self.material = material
    
    @property
    def mesh(self):
        return self._mesh
    
    @mesh.setter
    def mesh(self, mesh):
        assert isinstance(mesh, Mesh)
        self._mesh = mesh
    
    @property
    def material(self):
        return self._material
    
    @material.setter
    def material(self, material):
        assert isinstance(material, SurfaceMaterial)
        self._material = material
    
    @property
    def shader_name(self):
        '''
        albedo = self.mesh.albedo
        if albedo == 'TEXTURE':
            if self.material.material_properties_texture is None:
                return 'textured_shader'
            else:
                return 'textured_material_properties_shader'
        elif albedo == 'VERTEX_COLOR':
            return 'vertex_color_shader'
        elif albedo == 'FLAT':
            return 'flat_color_shader'
        '''
        return 'surface_shader'
    
    def activate(self, shader_locations):
        super().activate(shader_locations)
        self.material.activate(shader_locations)
        self.mesh.activate(shader_locations)
    
    def render(self, shader_locations):
        self.activate(shader_locations)
        num_triangles = len(self.mesh.faces)
        #print('pre')
        #GL.glDrawElements(
        #    GL.GL_TRIANGLES,
        #    num_triangles*3,
        #    GL.GL_UNSIGNED_INT,
        #    None,
        #)
        #print('rendered triangles')
    
    @staticmethod
    def sort_instances(instances):
        material_mesh_sort = {}
        for instance in instances:
            material_name = instance.material.name
            mesh_name = instance.mesh.name
            if material_name not in material_mesh_sort:
                material_mesh_sort[material_name] = {}
            if mesh_name not in material_mesh_sort[material_name]:
                material_mesh_sort[material_name][mesh_name] = []
            material_mesh_sort[material_name][mesh_name].append(instance)
        
        instances = []
        for material_name in material_mesh_sort:
            for mesh_name in material_mesh_sort[material_name]:
                instances.extend(material_mesh_sort[material_name][mesh_name])
        
        return instances
