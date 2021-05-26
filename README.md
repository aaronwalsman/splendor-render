## Welcome to splendor-render
Shiny!!

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
