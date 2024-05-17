import math

from OpenGL import GL

import numpy as np

import splendor.contexts.egl as egl
from splendor.core import SplendorRender
from splendor.frame_buffer import FrameBufferWrapper
from splendor.image import save_image
from splendor.camera import orthographic_matrix

def test_render():
    egl.initialize_plugin()
    egl.initialize_device()
    
    render_frame_buffer = FrameBufferWrapper(512, 512, anti_alias=True)
    
    renderer = SplendorRender()
    
    renderer.load_mesh(
        'ground_plane_mesh',
        mesh_primitive={
            'shape' : 'mesh_grid',
            'axes' : (0,2),
            'x_divisions' : 3,
            'y_divisions' : 3,
            'x_extents' : (-2.5,2.5),
            'y_extents' : (-2.5,2.5),
            'flip_normals' : True,
        },
        color_mode='flat_color',
    )
    
    renderer.load_material(
        'ground_plane_material',
        flat_color=(0,1,0),
        ambient=1.,
        metal=0.,
        rough=1.0,
        base_reflect=0.,
    )
    
    renderer.load_mesh(
        'sphere_mesh',
        mesh_primitive={
            'shape' : 'sphere',
        },
        #mesh_primitive={
        #    'shape' : 'mesh_grid',
        #    'axes' : (0,2),
        #    'x_divisions' : 3,
        #    'y_divisions' : 3,
        #    'x_extents' : (-0.5,0.5),
        #    'y_extents' : (-0.5,0.5),
        #    'flip_normals' : True,
        #},
        color_mode='flat_color',
    )
    renderer.load_material(
        'sphere_material',
        flat_color=(1,0,0),
        ambient=1.,
        metal=0.,
        rough=1.0,
        base_reflect=0.,
    )
    
    d = math.radians(-15)
    c = math.cos(d)
    s = math.sin(d)
    ground_plane_transform = np.array([
        [1,0,0,  0],
        [0,c,s,  0],
        [0,-s,c,-5],
        [0,0,0,  1],
    ])
    renderer.add_instance(
        'ground_plane',
        'ground_plane_mesh',
        'ground_plane_material',
        transform=ground_plane_transform,
    )
    
    sphere_transform = np.array([
        [1,0,0,  0],
        [0,1,0,  1],
        [0,0,1, -5],
        [0,0,0,  1],
    ])
    
    renderer.add_instance(
        'sphere',
        'sphere_mesh',
        'sphere_material',
        transform=sphere_transform,
    )
    
    renderer.set_ambient_color((0.4, 0.4, 0.4))
    renderer.add_direction_light(
        'top_light',
        #(0,-0.707,-0.707),
        (0,-1,0),
        (1,1,1),
    )
    
    # SHADOW
    
    shadow_fb = GL.glGenFramebuffers(1)
    depth_texture = GL.glGenTextures(1)
    
    GL.glBindTexture(GL.GL_TEXTURE_2D, depth_texture)
    initial_data = np.zeros((512,512), dtype=np.float32)
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_DEPTH_COMPONENT,
        512, 512, 0, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT, initial_data)
    GL.glTexParameteri(
        GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
    GL.glTexParameteri(
        GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT);
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT);
    
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, shadow_fb)
    GL.glFramebufferTexture2D(
        GL.GL_FRAMEBUFFER,
        GL.GL_DEPTH_ATTACHMENT,
        GL.GL_TEXTURE_2D,
        depth_texture,
        0,
    )
    GL.glDrawBuffer(GL.GL_NONE)
    GL.glReadBuffer(GL.GL_NONE)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
    
    # render shadow
    GL.glViewport(0, 0, 512, 512)
    GL.glScissor(0, 0, 512, 512)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, shadow_fb)
    GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
    shadow_projection = (orthographic_matrix(-10,10,-10,10) @ np.array([
        [ 1, 0, 0, 0],
        [ 0,-1, 0, 0],
        [ 0, 0, 1, 0],
        [ 0, 0, 0, 1],
    ]))
    shadow_view_matrix = np.linalg.inv(np.array([
        [ 1, 0, 0, 0],
        [ 0, 0, 1,10],
        [ 0,-1, 0,-5],
        [ 0, 0, 0, 1],
    ]))
    
    renderer.shader_library.use_program('depthmap_shadow_shader')
    try:
        location_data = renderer.shader_library.get_shader_locations(
            'depthmap_shadow_shader')
        GL.glUniformMatrix4fv(
            location_data['view_matrix'],
            1, GL.GL_TRUE,
            shadow_view_matrix.astype(np.float32),
        )
        GL.glUniformMatrix4fv(
            location_data['projection_matrix'],
            1, GL.GL_TRUE,
            shadow_projection.astype(np.float32),
        )
        GL.glUniform1f(location_data['radial_k1'], 0)
        GL.glUniform1f(location_data['radial_k2'], 0)
        
        for mesh_name in ['ground_plane_mesh', 'sphere_mesh']:
            renderer.load_mesh_color_shader_data(
                mesh_name, 'depthmap_shadow_shader')
        
            instance_name = mesh_name.replace('_mesh', '')
            mesh_data = renderer.loaded_data['meshes'][mesh_name]
            num_triangles = len(mesh_data['faces'])
            
            instance_data = (
                renderer.scene_description['instances'][instance_name])
            GL.glUniformMatrix4fv(
                location_data['model_pose'],
                1, GL.GL_TRUE,
                instance_data['transform'].astype(np.float32),
            )
            
            GL.glDrawElements(
                GL.GL_TRIANGLES,
                num_triangles*3,
                GL.GL_UNSIGNED_INT,
                None,
            )
        
        renderer.finish_frame()
        
        texture_data = GL.glReadPixels(
            0, 0, 512, 512,
            GL.GL_DEPTH_COMPONENT,
            GL.GL_FLOAT,
        )
        print(texture_data.min(), texture_data.max())
        texture_image = (texture_data * 255).astype(np.uint8)
        save_image(texture_image, 'tmp_depth.png')
    
    finally:
        GL.glUseProgram(0)
    
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
    
    # END SHADOW
    
    # TEST RENDER FROM SHADOW PERSPECTIVE
    render_frame_buffer.enable()
    renderer.viewport_scissor(0,0,512,512)
    
    # load shadow data
    renderer.shader_library.use_program('flat_color_shader')
    location_data = renderer.shader_library.get_shader_locations(
        'flat_color_shader')
    GL.glUniformMatrix4fv(
        location_data['shadow_view_matrix'],
        1, GL.GL_TRUE,
        shadow_view_matrix.astype(np.float32),
    )
    GL.glUniformMatrix4fv(
        location_data['shadow_projection_matrix'],
        1, GL.GL_TRUE,
        shadow_projection.astype(np.float32),
    )
    GL.glActiveTexture(GL.GL_TEXTURE4)
    GL.glBindTexture(GL.GL_TEXTURE_2D, depth_texture)
    
    #renderer.set_projection(shadow_projection)
    #renderer.set_view_matrix(shadow_view_matrix)
    #renderer.color_render()
    #frame = render_frame_buffer.read_pixels()
    #save_image(frame, './tmp_color.png')
    
    render_frame_buffer.enable()
    renderer.viewport_scissor(0,0,512,512)
    renderer.color_render()
    frame = render_frame_buffer.read_pixels()
    save_image(frame, './tmp.png')

if __name__ == '__main__':
    test_render()
