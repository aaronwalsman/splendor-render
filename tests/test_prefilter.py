"""Real EGL rendering: GGX environment prefilter, DFG table, and cache."""
import numpy as np
import pytest
from OpenGL import GL
from splendor.contexts.egl import EGLContext
from splendor.core import SplendorRender
from splendor.camera import projection_matrix
from splendor.image_light import prefilter, dfg


def srgb_to_linear(x):
    x = x / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


@pytest.fixture(scope='module')
def renderer():
    context = EGLContext()
    r = SplendorRender()
    yield r
    r.clear_scene()


def bake(r, strip, name, samples=64):
    r.load_cubemap(name, cubemap_data=strip, color_space='srgb')
    return prefilter.prefilter_cubemap(
        r.gl_data['cubemap_buffers'][name]['cubemap'],
        strip.shape[0], samples=samples)


def test_level0_identity(renderer):
    # Roughness 0 must reproduce the source exactly (linear-decoded): this
    # pins the bake's cubemap orientation to the upload convention.
    rng = np.random.default_rng(0)
    strip = rng.integers(0, 256, (16, 96, 3), dtype=np.uint8)
    levels = bake(renderer, strip, 'identity')
    assert len(levels) == 5  # log2(16)+1
    np.testing.assert_allclose(
        levels[0], srgb_to_linear(strip.astype(np.float32)), atol=5e-3)


def test_uniform_environment_invariance(renderer):
    # A constant environment must prefilter to the same constant at every
    # roughness level (normalization / energy preservation).
    strip = np.full((16, 96, 3), 128, dtype=np.uint8)
    expected = srgb_to_linear(128.0)
    levels = bake(renderer, strip, 'uniform', samples=256)
    for level in levels:
        np.testing.assert_allclose(level, expected, atol=2e-3)


def test_rough_levels_are_broad(renderer):
    # A single bright face must spread with roughness: the roughness-1 level
    # is visibly soft, not identifiable downsampled mirror detail.
    strip = np.zeros((32, 192, 3), dtype=np.uint8)
    strip[:, :32] = 255  # px face only
    levels = bake(renderer, strip, 'point', samples=512)
    sharp, soft = levels[0], levels[-1]
    h = soft.shape[0]
    faces = [soft[:, i*h:(i+1)*h].mean() for i in range(6)]
    assert sharp.max() - sharp.min() > 0.9
    # peak flattened well below the source's 1.0
    assert soft.max() < 0.65
    # substantial energy spread into the four perpendicular faces
    # (the opposite face stays dark: GGX is hemisphere-limited)
    assert all(f > 0.05 for f in faces[2:])
    assert faces[0] == max(faces)


def test_dfg_table():
    table = dfg.dfg_lookup_table(size=32, samples=256)
    assert table.shape == (32, 32, 2)
    assert np.all(table >= 0.0) and np.all(table <= 1.0 + 1e-3)
    # scale + bias is the reflectance of a white-F0 mirror: bounded by 1
    assert np.all(table.sum(axis=-1) <= 1.0 + 1e-3)
    # near-mirror, head-on: response approaches (1, 0)
    assert table[0, -1, 0] > 0.9
    assert table[0, -1, 1] < 0.1


def test_prefilter_cache(renderer, monkeypatch, tmp_path):
    monkeypatch.setenv('SPLENDOR_HOME', str(tmp_path))
    strip = np.full((8, 48, 3), 64, dtype=np.uint8)
    renderer.load_cubemap('cached', cubemap_data=strip, color_space='srgb')
    texture = renderer.gl_data['cubemap_buffers']['cached']['cubemap']
    first = prefilter.cached_prefilter_cubemap(strip, 'srgb', texture, samples=64)
    calls = []
    def fake_bake(*args, **kwargs):
        calls.append(1)
        return [np.zeros((8, 48, 3), dtype=np.float32)]
    monkeypatch.setattr(prefilter, 'prefilter_cubemap', fake_bake)
    second = prefilter.cached_prefilter_cubemap(strip, 'srgb', texture, samples=64)
    assert not calls
    for a, b in zip(first, second):
        np.testing.assert_allclose(a, b, atol=1e-3)  # float16 round trip
    # different parameters miss the cache
    prefilter.cached_prefilter_cubemap(strip, 'raw', texture, samples=64)
    assert calls


def test_image_light_render(renderer):
    r = renderer
    data = dict(
        vertices=np.array([[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]],dtype=np.float32),
        faces=np.array([[0,1,2],[0,2,3]],dtype=np.int32),
        normals=np.array([[0,0,1]]*4,dtype=np.float32))
    r.load_mesh('flat', mesh_data=data, color_mode='flat_color')
    r.load_sensor('rgb', 64, 64, anti_alias=False)
    view = np.eye(4); view[2,3] = -2
    r.load_camera('main', view_matrix=view,
        projection=projection_matrix(np.radians(60), 1))
    r.set_color_output(tone_map='none')
    strip = np.full((16, 96, 3), 128, dtype=np.uint8)
    r.load_cubemap('sky', cubemap_data=strip, color_space='srgb')
    r.load_image_light('light', reflect_cubemap='sky',
        irradiance_sh=np.zeros((9, 3)), render_background=False,
        set_active=True)
    r.load_material('mirror', flat_color=(1,1,1), rough=0, metal=1,
        base_reflect=0.5)
    r.add_instance('quad', mesh_name='flat', material_name='mirror',
        mask_color=(1/255, 2/255, 3/255))
    try:
        r.color_render('main', sensor='rgb')
        center = r.read_sensor('rgb')[32, 32].astype(float)
        # a head-on metallic mirror in a uniform environment reflects
        # approximately the environment color
        np.testing.assert_allclose(center, [128]*3, atol=8)
        # roughness must not change the color of a uniform environment
        r.load_material('mirror', flat_color=(1,1,1), rough=1, metal=1,
            base_reflect=0.5)
        r.color_render('main', sensor='rgb')
        rough_center = r.read_sensor('rgb')[32, 32].astype(float)
        np.testing.assert_allclose(rough_center, center, atol=20)
    finally:
        r.set_active_image_light(None)
        r.clear_instances()
