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

- **Splendor Render can run on servers!** Splendor Render can use EGL to run on headless servers that have no physical display.

- **Splendor Render is easy to use!**  I think.  If not, let me know and I'll try to make it better.

- **Splendor Render does not support shadows!**  This is not a feature.  This is an anti-feature.  I really need to get shadows in here at some point.

- **Splendor Render does not support transparent objects!**  This is also an anti-feature.  I'm just hoping that by publicly shaming myself in the README like this, I will some day take the time to implement this.

## Getting Started
Install this package:
```
pip install splendor-render
```

Run a script to install the assets:
```
splendor_asset_installer
```
This downloads about 15MB worth of images and textures that are used in various examples.  The install location defaults to `~/.cache/splendor` but can be changed by setting the `SPLENDOR_HOME` environment variable.

Run the interactive viewer:
```
splendor_viewer cereal
```

Render a single image to a file:
```
splendor_render cereal ./my_render.jpg
```

### Example
In the examples above `cereal` refers to `~/.cache/splendor/default_assets/scenes/cereal.json` using the asset library structure found in `~/.cache/splendor/default_assets.cfg`.  Let's take a look to get a sense of how scenes are stored in splendor-render:

```
"cubemaps":{
    "diffuse": {
        "cubemap_asset": "dresden_dif",
        "mipmaps": null
    },
    "reflect": {
        "cubemap_asset": "dresden_ref",
        "mipmaps": null
    }
},
```
The first section `"cubemaps"` defines two cube maps that will be used for the image-based lighting in the scene.  The first is named `"diffuse"` and will be used later to represent the non-shiny component of the lighting.  The second is named `"reflect"` and will be used later for the reflections (the shiny component of the lighting) and the background.  The `"cubemap_asset"` tells splendor-render to load the `"dresden_dif"` and `"dresden_ref"` assets accordingly, which live in `assets/cubemaps`.  These cubemaps don't do anything until we actually use them for something though, which leads us to...

```
"image_lights":{
    "background_1": {
        "diffuse_cubemap": "diffuse",
        "reflect_cubemap": "reflect",
        "blur":0,
        "render_background":1
    }
},
"active_image_light" : "background_1",
```

The next section defines the image-light that will illuminate the scene.  You can see it references the `"diffuse"` and `"reflect"` cubemaps that we defined earlier.  The `"render_background"=1` attribute tells the system to render the reflection map as the scene background, and the `"blur"` attribute defines how blurry the background will be when rendered.

```
"meshes": {
    "cereal_box": {
        "mesh_primitive":{
            "shape":"cube",
            "x_extents":[-3.5,3.5],
            "y_extents":[-5.5,5.5],
            "z_extents":[-1,1],
            "bezel":0.05
        },
        "scale":1.0,
        "color_mode":"textured"
    }
},
```

Here we define a single mesh named `"cereal_box"`.  You can see it is constructed using a primitive cube with some preset extents and a small bezel.  Arbitrary meshes are supported using .obj files.

```
"textures": {
    "splendor" : {
        "texture_asset":"splendor_texture"
    },
    "splendor_matprop" : {
        "texture_asset":"splendor_matprop"
    }
},
```

Next we load two textures.  This is similar to the cubemaps we saw at the beginning where both are referred to using their asset name.  These also don't do anything util we use them for something.\

```
"materials": {
    "varying": {
        "texture_name":"splendor",
        "material_properties_texture":"splendor_matprop"
    },
    "fixed": {
        "texture_name":"splendor",
        "ambient":1.0,
        "metal":0.75,
        "rough":0.75,
        "base_reflect":0.04
    }
},
```

Here we build two materials that can be used in the scene.  The first is `"varying"` which uses the second texture (`"splendor_matprop"`) to define the material properties of the surface.  The second sets each of these values explicitly for the entire surface.  The `"ambient"` parameter affects how much the surface is affected by the ambient light (uniform lighting bias) in the scene.  The other three (`"metal"`, `"rough"` and `"base_reflect"`) control the interaction between the lighting and the surface.  Briefly, `"base_reflect"` makes the surface more shiny.  When `"base_reflect=1"` the surface will become a pure mirror and will ignore the albedo (surface color/texture) entirely.  The `"rough"` parameter makes these reflections more blurry, while `"metal"` is another kind of shiny that incorporates the albedo and makes the surface look like colored refelctive foil instead of a pure mirror.

```
"instances": {
    "cube_1": {
        "mesh_name": "cereal_box",
        "material_name": "varying",
        "transform": [
            [-1,0,0,0],
            [0,1,0,0],
            [0,0,-1,0],
            [0,0,0,1]],
        "mask_color": [1,1,1]
    }
},
```

Finally we add an instance to the scene.  In splendor-render no meshes are displayed until we create an instance of them.  This allows us to have many copies of a mesh in the scene while only loading it once.  Each instance is essentially a combination of a mesh, a material and a 3D transform that places it somewhere in the scene.  Here you can see we refer to the `"cereal_box"` mesh and `"varying"` material that we created earlier.

```
"ambient_color": [0,0,0],
"point_lights": {},
"direction_lights": {},
"camera":{
    "view_matrix":[0.0, 0.0, 0, 12.0, 0, 0],
    "projection":[
        [ 1.73205081,  0.0       ,  0.0       ,  0.0       ],
        [ 0.0       ,  1.73205081,  0.0       ,  0.0       ],
        [ 0.0       ,  0.0       , -1.002002  , -0.1001001 ],
        [ 0.0       ,  0.0       , -1.0       ,  0.0       ]]
}
```
This last section defines the ambient color (in this case `[0,0,0]` which means no contribution), an empty list of `"point_lights"` and `"direction_lights"` (we are using the image light above instead of these simpler lights), as well as camera information.

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
