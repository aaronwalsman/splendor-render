# Plan: predictable rough environment reflections

Status: IMPLEMENTED 2026-09-15 (authorized by Aaron in a later session); see
environment_prefilter.md for the as-built description. This file is kept as
the design/validation record. Notable deviations: the roughness ladder is a
fixed 6 levels (not 9) so the roughness-to-LOD mapping is independent of
environment resolution; a screen-footprint LOD floor was tried and REMOVED
(ladder levels are roughness units, so a resolution-mip floor over-blurs
curved mirrors) leaving near-mirror aliasing as the open concern from step 3;
multi-scattering energy compensation (Filament-style) was added so rough
metals do not darken.

## Objective and accepted baseline

Replace screen-footprint-dependent environment reflection blur with a GGX
prefiltered environment implementation that remains visibly soft at roughness 1.
Preserve the restored historical appearance as the comparison baseline. Do not
trade its broad rough reflections for sharper ones merely to simplify sampling.
Roughness is a perceptual 0–1 material parameter, not a mip-level index.

The live checkout now uses GL_LINEAR_MIPMAP_LINEAR for cubemaps and retains
`texture(sampler, direction, roughness * 4)` (a bias, not explicit LOD).
This restoration is authorized and completed. No GGX prefiltering is implemented.

History: cfdacf6 (2024-12-14, macOS work) disabled cubemap mip filtering; merge
9358620 brought it into this branch on 2026-04-08. Local master 28b8746 retains
the original filtering and bias. The historical combination can be much blurrier
than `textureLod(..., roughness * 4)` because it adds to automatic mip selection.
The earlier explicit-LOD experiment was NOT an adequate replacement baseline.

## Existing experiment and evidence

Standalone generator and instructions:
`/home/awalsman/Development/DM/roughness_comparison/compare.py` and `README.md`.
Artifacts in `/tmp/splendor-roughness/` include environment.json, direct.json,
comparison.png, logs, and isolated renderer snapshots current/restored_bias/
mip_sampling. These temporary artifacts may not survive; preserve useful images
in the repo when taking up this work.

The snapshots compare the previously unfiltered checkout, restored filtering
with historical bias, and filtering with explicit LOD, all with the same other
code. They are not full historical-checkout renders. Metallic spheres isolate
specular response; roughness values are .05, .25, .5, .75, 1. Direct-light images
were pixel-identical for unfiltered vs explicit-LOD variants. After restoration,
the live renderer matched environment-restored_bias.png pixel-for-pixel.

The generator predates restoration and assumes a disabled-filter source line
when patching copies. Update that assumption and label the baseline explicitly
before rerunning. Do not overwrite useful snapshots before preserving them.

## Proposed implementation

1. Bake a fixed set of GGX-prefiltered roughness levels per environment. Every
   material shares this set; do not bake per object or per material texture.
   Store levels in a cubemap mip chain, with sharp level 0 and progressively
   broader filtering at higher levels. Nine levels for a 256-pixel base face is
   a reasonable experiment, not a settled quality requirement.
2. Define and document the same perceptual-roughness-to-level mapping in baker
   and shader. For a linear mapping over N baked roughness samples, use
   `lod = roughness * (N - 1)` and trilinear interpolation. GGX microfacet alpha
   is typically perceptual roughness squared; do not accidentally square twice
   or bake alpha values while sampling them as perceptual roughness.
3. Use explicit LOD for the prefiltered reflection lookup. The baked angular
   spread, rather than a footprint bias, determines material roughness.
   Separately evaluate aliasing of near-mirror reflections at different distances.
4. Add the shared 2D BRDF integration lookup (roughness, N dot V) for a consistent
   split-sum specular IBL approximation. It is independent of environment and
   can be generated once. Integrate with the existing Fresnel/specular energy
   terms carefully; do not multiply equivalent factors twice. Document the
   remaining approximation, particularly grazing angles and rough surfaces.
5. Preserve linear HDR through decoding, convolution and GPU storage. Current
   cubemap uploads accept 8-bit images; add a deliberate floating-point path
   while preserving explicit sRGB/raw handling for legacy inputs. Do not clip
   a bright sun before convolution or apply display tone mapping during baking.
6. Cache results by source identity/content, face size, sampling settings, bake
   algorithm version and roughness mapping. Bake expensive integrations offline
   or on cache miss, never each frame. Report bake time, cache size and runtime
   cost. Keep the implementation small and directly callable.
7. Keep the background panorama and diffuse lighting separate from the specular
   bake. The existing SH diffuse path does not become a specular cubemap level.
   Preserve cubemap face order/orientation, including the current X reflection
   in skybox sampling. Avoid changing skybox blur semantics unintentionally.

## Scope boundaries

Do not change shadows, robot materials, scene exposure, direct-light equations,
USD import defaults, or general lighting calibration to make this test pass.
Do not infer diffuse-only materials from names containing 'lambert'.
There are concurrent uncommitted color-pipeline, camera, drawing, examples and
viewer changes. Inspect status and preserve them; do not reset the checkout.
A future assignment authorizes the work; this document alone is not approval to
start modifying the renderer.

## Validation and review gates

- First preserve/render the restored bias baseline through both basic
  splendor_viewer and splendor_render, including the legacy cereal scene.
- Compare roughness sweeps with identical geometry, view, exposure, material
  reflectance and source environment. Include dielectric and metallic spheres.
- Use a detailed environment and a synthetic bright HDR light patch. At roughness
  1 reflections must be broad, comparable in character to the accepted baseline,
  not identifiable mirror detail with a little downsampling. If physically
  motivated filtering differs, present it for visual review rather than silently
  weakening the requested softness or increasing unrelated lighting.
- Repeat at multiple resolutions/distances: angular roughness response should
  remain consistent, unlike historical bias. Quantify aliasing separately.
- Check uniform-environment invariance for the normalized prefilter itself,
  HDR energy preservation, cube seams/orientation and smooth interpolation
  between levels. Compare selected outputs with a high-sample numerical reference.
- Verify direct-light rendering stays unchanged; run existing color-pipeline,
  material-property texture, mask/depth, distortion and MSAA checks as applicable.
  The tested EGL device previously rejected default 8x MSAA: distinguish hardware
  limits from regressions and use supported sample counts for comparisons.
- Deliver before/after images, commands, bake/cache/runtime measurements and a
  clear account of approximation limits before replacing the accepted default.

## Relevant code and references

- splendor/core.py: load_cubemap, load_image_light, material loading
- splendor/shaders/skybox.py: implicit sampling and bias overloads
- splendor/shaders/lighting_model.py: roughness*MAX_MIPMAP (currently 4), IBL terms
- splendor/shaders/pbr.py: existing GGX/Fresnel implementation
- splendor/image_light/: environment processing and SH integration
- tests/test_color_pipeline.py and docs/color_management.md
- https://google.github.io/filament/main/filament.html (IBL prefilter + DFG LUT)
- https://google.github.io/filament/dup/cmgen.html (offline baking reference)
