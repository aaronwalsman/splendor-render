# GGX prefiltered environment reflections (as built)

Environment reflection roughness is a perceptual 0-1 material parameter mapped
to a baked angular spread, replacing the historical screen-footprint mip bias
(`texture(cube, dir, roughness*4)`).

## Bake

`splendor/image_light/prefilter.py` importance-samples GGX (Karis split-sum,
N=V=R, alpha = perceptual roughness squared, pdf-matched source mips against
fireflies) on the GPU into a fixed ladder of 6 roughness levels
(0, .2, .4, .6, .8, 1), stored as an explicit RGB16F linear cubemap mip chain:
level 0 sharp at the source face size, each level half size. The ladder length
is constant so the roughness-to-LOD mapping does not depend on environment
resolution (clamped only for tiny sources). Face cameras/orientation come from
`image_light.panorama`; a roughness-0 bake reproduces the source exactly
(tested), which pins orientation. Replaces the retired Gaussian
`blurry_mipmaps` approach.

Bakes run at `load_image_light` time and are cached in
`~/.cache/splendor/prefilter/` (XDG / `SPLENDOR_HOME` aware), keyed by content
hash of the source pixels + bake parameters + `PREFILTER_VERSION`. Stored
float16, ~1 MB per 512 environment; a cold bake plus render measured ~1 s
total, cache hits are a file load. Reloading a cubemap's pixels invalidates
and rebakes.

## Shading

`lighting_model.py` samples with explicit LOD `rough * prefilter_max_lod`
(trilinear between levels) and applies the split-sum environment BRDF from a
shared DFG lookup table (`image_light/dfg.py`, 64x64 RG16F, numpy-integrated
once and cached; k = alpha/2 matching pbr.py's IBL geometry term), plus
Filament-style multi-scattering energy compensation so a uniform environment
reflects the same color at every roughness (tested). The `reflect_gamma`,
`reflect_correction`, and `reflect_bias` knobs are retired;
`image_light_properties` became `image_light_diffuse` (scale, bias). The
background pass still samples the raw cubemap with its own blur semantics; SH
diffuse is untouched.

## Reflection anti-aliasing

Explicit LOD alone made near-mirror reflections sparkle wherever they were
compressed on screen (sphere rims, distant mirrors — see
`DM/roughness_comparison/aliasing_demo.py`). Two complementary mechanisms fix
it:

1. Geometric specular AA (Tokuyoshi-Kaplanyan) computed on the *reflected
   direction*, so both curvature- and perspective-induced minification widen
   the lobe: screen-space direction variance adds to GGX alpha^2
   (`SPECULAR_AA_SIGMA2 = 0.0625` on reflected dirs = the standard 0.25 on
   normals; `SPECULAR_AA_KAPPA = 0.18` caps widening at perceptual ~0.65).
   The widened roughness drives the ladder LOD, DFG lookup and energy
   compensation together.
2. A footprint-filtered mirror end: below ladder level 1 the raw auto-mipped
   cubemap (`reflect_footprint_sampler`, the same texture the background
   uses) is sampled with hardware LOD selection and blended toward ladder
   level 1 by roughness. A rough-0 GGX lobe is a delta, so the correctly
   filtered mirror IS the footprint-filtered environment; this covers the
   mild-minification regime (a few environment texels per pixel) where
   roughness-sized ladder steps are far too coarse.

A naive `max(rough_lod, resolution_footprint_lod)` does NOT work: ladder
levels are roughness units, so a resolution-mip floor over-blurs every curved
mirror.

## Known limits
- The BRDF integration is single-scattering plus scalar energy compensation;
  grazing-angle behavior also interacts with the historical 0.2 cos_theta
  clamp in `fresnel_schlick_rough` (still present, used by diffuse/ambient).
- 8-bit sRGB sources are decoded to linear before convolution, but a true
  HDR (float) cubemap input path still does not exist, so bright suns are
  clipped at 1.0 before the bake.

Validation: `tests/test_prefilter.py` (EGL: level-0 identity, uniform-
environment invariance, broadness at roughness 1, DFG sanity, cache behavior,
rendered uniform-environment color constancy across roughness). Direct-light
renders match the pre-change baseline within 1/255. Visual roughness sweeps
vs the accepted baseline:
`/home/awalsman/Development/DM/roughness_comparison/prefilter_comparison.png`.
