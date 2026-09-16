# Linear color rendering

`color_render` shades into a floating-point RGBA buffer, resolves MSAA and any
lens distortion, then applies one display transform. Constant material colors,
vertex colors, background/ambient colors, light intensities, and spherical
harmonic irradiance coefficients are linear RGB. No PBR lobe, Fresnel, shadow,
or light-distance equations changed in this update.

## Inputs

```python
renderer.load_texture('albedo', texture_path='base_color.png', color_space='srgb')
renderer.load_texture('properties', texture_path='material.png', color_space='raw')
renderer.load_cubemap('environment', cubemap_path='sky.png', color_space='srgb')
```

The default for 8-bit RGB(A) textures/cubemaps is `srgb`. OpenGL's sRGB internal
format decodes RGB for sampling; alpha remains linear. `raw` preserves numeric
channels and is required for packed material-property maps (metal, roughness,
base reflectance, ambient in RGBA order). Normal maps, when used by a custom
shader, likewise need `raw`. This change does not add a normal-map shader.

The source files and arrays are unchanged. A gray sRGB byte value of 128 represents
approximately 0.216 linear intensity; use that linear value for an equivalent
`flat_color`. See the [OpenGL sRGB texture specification](https://registry.khronos.org/OpenGL/specs/gl/glspec33.core.pdf).

## Output

```python
renderer.set_color_output(
    exposure=0.0, tone_map='reinhard', color_space='srgb', precision=16)
```

These are the defaults. Exposure is measured in stops: +1 multiplies linear
intensity by two. Reinhard maps each exposed RGB channel `x` to `x / (1 + x)`.
`precision` selects the bit depth of the intermediate linear buffers:
16 (`GL_RGBA16F`) or 32 (`GL_RGBA32F`). Half float is more than enough for
display output at half the memory and bandwidth; use 32 when reading back
radiometric linear HDR. Changing precision reallocates the cached linear
targets on the next `color_render`.
The final sRGB encoding uses the piecewise sRGB transfer function. Alpha and depth
are copied without the color transform. Output settings serialize with the scene.

To evaluate colors without highlight compression, choose `tone_map='none'`.
For unmodified linear HDR output, use `tone_map='none', color_space='linear'`,
zero exposure, and a `GL_RGBA32F` sensor. Display output and raw HDR output use
the same lighting calculation.

The display pass applies to both named sensors and the currently bound window
framebuffer/viewport. It includes scene backgrounds and color/debug geometry.
Calling `color_render(clear=False)` accumulates onto that target's previous
**linear color render**, then applies the display transform once. It does not
import unrelated drawing or mask renders from the output framebuffer into the
HDR buffer. A new or resized color target starts cleared.

## Sensor isolation and cost

Mask and coordinate rendering bypass the floating-point color path and display
transform. Their encoded IDs/data remain exact; exposure does not affect them.
Depth from color rendering is copied through the display pass. Lens-distortion
interpolation and MSAA happen before the display transform.

Color rendering adds one cached floating-point target per sensor/window viewport
and one fullscreen pass. Targets are reused and released when sensors are removed
or resized. With MSAA, the HDR target also needs multisampled storage; this increases
GPU memory use. Mask-only sensors do not allocate this extra color target.

## Existing assets

Old scenes tuned in encoded RGB will change appearance. Previously generated SH
lighting may also contain legacy gamma/normalization choices: coefficients are
treated as linear, and this update does not regenerate them. Recalibrate those
lighting assets separately, after comparing under the new color pipeline.

Run actual EGL rendering checks with:

```sh
python -m pytest tests/test_color_pipeline.py -q
```


### Legacy scene texture defaults

`load_scene` examines material references before uploading textures. When
`color_space` is omitted, material-property textures default to `raw`; other
textures retain the `srgb` default. A texture used for both color and material
properties defaults to `raw` and emits a warning. Explicit declarations are
preserved, and the input scene dictionary is not modified. Direct `load_texture`
calls still require `color_space='raw'` for material-property maps.
