"""Real EGL rendering: color science and isolation from sensor data passes."""
import json
import numpy as np
import pytest
from OpenGL import GL
from splendor.contexts.egl import EGLContext
from splendor.core import SplendorRender
from splendor.camera import projection_matrix


@pytest.fixture(scope='module')
def renderer():
    context = EGLContext()
    r = SplendorRender()
    data = dict(vertices=np.array([[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]],dtype=np.float32),
                faces=np.array([[0,1,2],[0,2,3]],dtype=np.int32),
                normals=np.array([[0,0,1]]*4,dtype=np.float32))
    r.load_mesh('flat',mesh_data=data,color_mode='flat_color')
    r.load_mesh('textured',mesh_data=dict(data,uvs=np.array([[0,0],[1,0],[1,1],[0,1]],dtype=np.float32)),color_mode='textured')
    r.load_sensor('rgb',64,64,anti_alias=False)
    view=np.eye(4);view[2,3]=-2
    r.load_camera('main',view_matrix=view,projection=projection_matrix(np.radians(60),1))
    r.set_ambient_color((1,1,1))
    r.load_texture('gray',texture_data=np.full((4,4,3),128,dtype=np.uint8),color_space='srgb')
    r.load_material('texture',texture_name='gray',rough=1,base_reflect=0)
    linear=((128/255+.055)/1.055)**2.4
    r.load_material('constant',flat_color=(linear,)*3,rough=1,base_reflect=0)
    yield r
    r.clear_scene()


def pixel(r, material, mesh='flat', sensor='rgb', **kw):
    r.clear_instances()
    r.add_instance('quad',mesh_name=mesh,material_name=material,mask_color=(17/255,83/255,201/255))
    r.color_render('main',sensor=sensor,**kw)
    return r.read_sensor(sensor)[32,32].astype(float)


def test_srgb_texture_matches_linear_constant(renderer):
    r=renderer
    encoded=128/255
    linear=((encoded+.055)/1.055)**2.4
    r.load_texture('gray',texture_data=np.full((4,4,3),128,dtype=np.uint8),color_space='srgb')
    r.load_material('texture',texture_name='gray',rough=1,base_reflect=0)
    r.load_material('constant',flat_color=(linear,)*3,rough=1,base_reflect=0)
    r.set_color_output(tone_map='none')
    a=pixel(r,'constant');b=pixel(r,'texture','textured')
    np.testing.assert_allclose(a,[128]*3,atol=1)
    np.testing.assert_allclose(a,b,atol=1)
    # Mip storage and sampling use an actual sRGB internal format.
    GL.glBindTexture(GL.GL_TEXTURE_2D,r.gl_data['texture_buffers']['gray']['texture'])
    assert GL.glGetTexLevelParameteriv(GL.GL_TEXTURE_2D,0,GL.GL_TEXTURE_INTERNAL_FORMAT)==GL.GL_SRGB8


def test_raw_data_and_hdr(renderer):
    r=renderer
    r.load_texture('raw',texture_data=np.full((4,4,3),128,dtype=np.uint8),color_space='raw')
    r.load_material('raw_mat',texture_name='raw',rough=1,base_reflect=0)
    r.load_sensor('hdr',64,64,anti_alias=False,color_format=GL.GL_RGBA32F)
    r.set_color_output(tone_map='none',color_space='linear',precision=32)
    np.testing.assert_allclose(pixel(r,'raw_mat','textured','hdr'),[128/255]*3,atol=1e-5)
    r.load_material('bright',flat_color=(4,4,4),rough=1,base_reflect=0)
    np.testing.assert_allclose(pixel(r,'bright',sensor='hdr'),[4]*3,atol=1e-5)
    r.set_color_output(exposure=1,tone_map='reinhard',color_space='linear',precision=32)
    np.testing.assert_allclose(pixel(r,'bright',sensor='hdr'),[8/9]*3,atol=1e-5)
    r.set_color_output(tone_map='none',color_space='linear')
    # Data channels must not silently inherit color-texture decoding.
    with pytest.raises(ValueError,match='raw'):
        r.load_material('bad',material_properties_texture='gray')
    r.load_material('properties',texture_name='gray',material_properties_texture='raw')
    pixel(r,'properties','textured')


def test_masks_depth_and_coordinates_ignore_output_settings(renderer):
    r=renderer
    r.load_material('white',flat_color=(1,1,1),rough=1,base_reflect=0)
    r.set_color_output(tone_map='none')
    pixel(r,'white')
    depth_before=r.read_sensor('rgb',read_depth=True,projection=r.scene_description['cameras']['main']['projection']).copy()
    r.mask_render('main',sensor='rgb')
    mask_before=r.read_sensor('rgb').copy()
    np.testing.assert_array_equal(mask_before[32,32],[17,83,201])
    r.coord_render('main',sensor='rgb')
    coords_before=r.read_sensor('rgb').copy()
    r.set_color_output(exposure=4,tone_map='reinhard',color_space='linear')
    pixel(r,'white')
    depth_after=r.read_sensor('rgb',read_depth=True,projection=r.scene_description['cameras']['main']['projection'])
    np.testing.assert_array_equal(depth_before,depth_after)
    r.mask_render('main',sensor='rgb')
    np.testing.assert_array_equal(mask_before,r.read_sensor('rgb'))
    r.coord_render('main',sensor='rgb')
    np.testing.assert_array_equal(coords_before,r.read_sensor('rgb'))


def test_accumulation_serialization_and_window_target(renderer):
    r=renderer
    r.set_color_output(exposure=1,tone_map='reinhard')
    expected=pixel(r,'constant')
    r.color_render('main',sensor='rgb',clear=False)
    np.testing.assert_allclose(r.read_sensor('rgb')[32,32],expected,atol=1)
    description=json.loads(r.get_json_description())
    r.set_color_output()
    r.load_scene({'color_output':description['color_output']})
    assert r.scene_description['color_output']==description['color_output']
    # sensor=None uses the current framebuffer, as in the GLFW window.
    r._bind_sensor('rgb')
    r.color_render('main')
    np.testing.assert_allclose(r.read_sensor('rgb')[32,32],expected,atol=1)


def test_distortion_and_msaa(renderer):
    r=renderer
    r.load_sensor('lens',64,64,anti_alias=True,anti_alias_samples=4,enable_radial_distortion=True)
    r.scene_description['cameras']['main']['radial_k1']=.1
    r.set_color_output(tone_map='none')
    try:
        np.testing.assert_allclose(pixel(r,'texture','textured','lens'),[128]*3,atol=2)
    finally:
        r.scene_description['cameras']['main']['radial_k1']=0
    r.remove_sensor('lens')
    assert ('sensor','lens') not in r.gl_data['color_buffers']


def test_precision_setting(renderer):
    r=renderer
    with pytest.raises(ValueError,match='precision'):
        r.set_color_output(precision=8)
    # 16 is the default; both precisions agree on 8-bit display output, and
    # switching reallocates the cached linear target and any distortion
    # intermediate at the new format.
    r.load_sensor('prec',64,64,anti_alias=False,enable_radial_distortion=True)
    r.set_color_output(tone_map='none')
    assert r.scene_description['color_output']['precision']==16
    half=pixel(r,'constant',sensor='prec')
    for fbo_key,precision,fmt in ((('sensor','prec'),16,GL.GL_RGBA16F),
                                  (('sensor','prec'),32,GL.GL_RGBA32F)):
        r.set_color_output(tone_map='none',precision=precision)
        result=pixel(r,'constant',sensor='prec')
        np.testing.assert_allclose(result,half,atol=1)
        assert r.gl_data['color_buffers'][fbo_key].color_format==fmt
        assert r.gl_data['sensor_buffers']['prec']['intermediate_fbo'].color_format==fmt
    r.set_color_output(tone_map='none')
    r.remove_sensor('prec')


def test_srgb_cubemap_background(renderer):
    r=renderer
    r.clear_instances()
    r.set_color_output(tone_map='none')
    r.load_cubemap('gray_sky',cubemap_data=np.full((4,24,3),128,dtype=np.uint8),color_space='srgb')
    r.load_image_light('sky',reflect_cubemap='gray_sky',irradiance_sh=np.zeros((9,3)),render_background=True,set_active=True)
    try:
        r.color_render('main',sensor='rgb')
        np.testing.assert_allclose(r.read_sensor('rgb')[32,32],[128]*3,atol=1)
    finally:
        r.set_active_image_light(None)



def test_scene_infers_texture_color_space_without_mutation(renderer):
    r = renderer
    names = ('infer_color', 'infer_data', 'infer_shared', 'infer_explicit', 'infer_unused')
    scene = {
        'textures': {name: {'texture_data': np.full((4, 4, 3), 128, dtype=np.uint8)}
                     for name in names},
        'materials': {
            'infer_separate': {'texture_name': 'infer_color',
                               'material_properties_texture': 'infer_data'},
            'infer_mixed': {'texture_name': 'infer_shared',
                            'material_properties_texture': 'infer_shared'},
            'infer_override': {'texture_name': 'infer_explicit'},
        },
    }
    scene['textures']['infer_explicit']['color_space'] = 'raw'
    with pytest.warns(UserWarning, match='infer_shared.*both color.*raw'):
        r.load_scene(scene)
    expected = ('srgb', 'raw', 'raw', 'raw', 'srgb')
    for name, space in zip(names, expected):
        assert r.scene_description['textures'][name]['color_space'] == space
    for name in names:
        assert ('color_space' in scene['textures'][name]) == (name == 'infer_explicit')
    # An explicit sRGB declaration is preserved, including the existing error
    # when that texture is subsequently used as material data.
    with pytest.raises(ValueError, match='raw'):
        r.load_scene({
            'textures': {'infer_bad': {'texture_data': np.zeros((4, 4, 3), dtype=np.uint8),
                                      'color_space': 'srgb'}},
            'materials': {'infer_bad': {'material_properties_texture': 'infer_bad'}},
        })
