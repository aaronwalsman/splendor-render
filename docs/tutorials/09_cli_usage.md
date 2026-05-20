# Command-Line Tools

Splendor provides two main CLI tools: `splendor_render` for headless
batch rendering and `splendor_viewer` for interactive scene viewing.

## splendor_render

Render a scene to an image file:

```bash
# Color render (default)
splendor_render examples/02_primitives color.png --resolution 768x512

# Mask render
splendor_render examples/02_primitives mask.png --render-mode mask

# Depth render
splendor_render examples/03_shadows depth.npy --render-mode depth
```

Options:
- Second positional argument is the output file (PNG for color/mask, NPY for depth)
- `--resolution` — render resolution as `WxH` (default: 512x512)
- `--render-mode` — `color`, `mask`, or `depth`
- `--anti-alias-samples` — MSAA sample count (0 to disable, default 8)

## splendor_viewer

Open a scene in an interactive window:

```bash
splendor_viewer examples/03_shadows
```

The viewer watches the scene file and reloads automatically when it
changes on disk — edit the JSON in your text editor and see the result
immediately.

### Viewer controls

| Input | Action |
|-------|--------|
| Left-drag | Orbit around the point under cursor |
| Right-drag / Shift+left-drag | Pan |
| Scroll wheel | Zoom |
| **S** | Save screenshot (timestamped, current directory) |
| **M** | Toggle color / mask render mode |

## Using scenes by name

Both tools can load scenes by asset name (from the asset library) or by
file path:

```bash
# By asset name (searches asset library directories)
splendor_viewer cereal

# By file path
splendor_viewer examples/02_primitives.json

# Without the .json extension (also works)
splendor_viewer examples/02_primitives
```

Source: [examples/09_cli_usage.sh](../../examples/09_cli_usage.sh)
