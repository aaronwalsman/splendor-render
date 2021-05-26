# Splendor Render
Shiny!!

## Purpose
This package is designed to be a lightweight but nice looking 3D renderer in python.  It has primarily been designed to generate online training data for computer vision, reinforcement learning and robotics applications.

There are a few other packages that do similar things that you may also want to check out:
- [PyRender](https://github.com/mmatl/pyrender) is a really solid renderer and does a lot of what Splendor Render can do.  It is also probably much more stable and compliant with various specifications and supports more formats of things, and definitely has waaaaay better documentation than we do.  You should really go check out PyRender and see if that works for you.  I *think* the one thing we support that PyRender does not is image-based lighting, which is really nice and makes for shiny pictures that I'm *very* proud of.

- [PyBullet](https://pybullet.org/wordpress/) is focused more on physics simulation, but some people use it for generating images as well.  I haven't looked into it, but I would be surprised if they have all the nice shader stuff and image-based lighting that we do.

### Features
- **Splendor Render is shiny!**  Splendor Render uses various approximations of physically based rendering (PBR) for image-based lighting (IBL).  These were big buzzwords in video game development ten years ago, but they basically mean we use a lighting/material model that uses environment maps for lighting and provides relatively simple material controls (base reflectance, metal, roughness) that are designed to approximate physics.  

- **Splendor Render is fast!** For simple scenes you can easily generate images using image-based lighting at faster than 1000hz (including copying images back from the graphics card) with a good GPU.

- **Splendor Render is easy to use!**  I think.  If not, let me know and I'll try to make it better.

- **Splendor Render does not support shadows!**  This is not a feature.  This is an anti-feature.  I really need to get shadows in here at some point.

- **Splendor Render does not support transparent objects!**  This is also an anti-feature.  I'm just hoping that by publicly shaming myself in the README like this, I will some day take the time to implement this.

## Getting Started
Install this package:
```
pip install -e .
```

Run the interactive viewer:
```
splendor_viewer cereal
```

Render a single image to a file:
```
splendor_render cereal ./my_render.jpg
```

## Asset Paths
Splendor Render supports asset paths for easy asset loading.  To make an asset library, simply make a .cfg structured like this:
```
[paths]
assets = /path/to/assets/folder
image_lights = %(assets)s/image_lights
meshes = %(assets)s/meshes
materials = %(assets)s/materials
textures = %(assets)s/textures
panoramas = %(assets)s/panoramas
scenes = %(assets)s/scenes
```
Paths do not necessarily need to be relative to a single assets directory but it does make it more convenient.  Also, any path can use `{HERE}` to specify a path relative to the location of the .cfg file.  Using the asset library in this way lets you refer to scene components without specifying their full path.  The scene entry for a mesh can then look something like:
```
"meshes": {
  "my_mesh":{
    "mesh_asset":"mesh_name",
    "scale":1.0,
    "color_mode":"flat_color"
}
```
As long as a file named `mesh_name.obj` exists in the mesh path specified by the .cfg file, then everything will be loaded correctly.  In order to specify what asset file to use when loading a scene, you can pass it into SplendorRender:
```
renderer = SplendorRender(assets='/path/to/assets.cfg')
```
If nothing is passed in, the default asset file `default_assets.cfg` is used.  The command line utilities should all have `--assets` flags that let you specify an assets file when loading a scene for viewing/rendering/etc.

Finally, multiple asset files can be used by specifying a comma-separated list.  When looking for assets, the paths are searched in order until something is found.  The keyword `DEFAULT` can be used to refer to the default assets path found in `default_assets.cfg`.
