# system
import math

# numpy
import numpy

# local
import renderpy.core as core
import renderpy.camera as camera

def second_test():
    return {
  "background":{},
  "meshes": {
    "test_cube": {
      "primitive": "cube"
    },
    "test_sphere": {
      "primitive": "sphere"
    }
  },
  "materials": {
    "spinner": {
      "example_texture": "spinner_tex",
      "ka": 1.0,
      "kd": 1.0,
      "ks": 0.5,
      "shine": 4.0
    },
    "candy_color": {
      "example_texture": "candy_color2",
      "ka": 1.0,
      "kd": 1.0,
      "ks": 0.5,
      "shine": 4.0
    }
  },
  "instances": {
    "cube1": {
      "mesh_name": "test_cube",
      "material_name": "spinner",
      "transform": [
        [
          0.707,
          0.0,
          -0.707,
          0.0
        ],
        [
          0.0,
          1.0,
          0.0,
          0.0
        ],
        [
          0.707,
          0.0,
          0.707,
          0.0
        ],
        [
          0.0,
          0.0,
          0.0,
          1.0
        ]
      ],
      "mask_color": [
        0,
        0,
        0
      ]
    },
    "sphere1": {
      "mesh_name": "test_sphere",
      "material_name": "candy_color",
      "transform": [
        [
          0.707,
          0.0,
          -0.707,
          0.0
        ],
        [
          0.0,
          1.0,
          0.0,
          0.0
        ],
        [
          -0.707,
          0.0,
          -0.707,
          -4.0
        ],
        [
          0.0,
          0.0,
          0.0,
          1.0
        ]
      ],
      "mask_color": [
        1,
        1,
        1
      ]
    }
  },
  "ambient_color": [
    0.2,
    0.2,
    0.2
  ],
  "point_lights": {},
  "direction_lights": {
    "light_main": {
      "direction": [
        0.5,
        0.0,
        -0.866
      ],
      "color": [
        1,
        1,
        1
      ]
    }
  },
  "camera": {
    "pose": [
      [
        1,
        0,
        0,
        0
      ],
      [
        0,
        1,
        0,
        0
      ],
      [
        0,
        0,
        1,
        0
      ],
      [
        0,
        0,
        0,
        1
      ]
    ],
    "projection": [
      [
        1.0000000000000002,
        0.0,
        0.0,
        0.0
      ],
      [
        0.0,
        1.0000000000000002,
        0.0,
        0.0
      ],
      [
        0.0,
        0.0,
        -1.002002002002002,
        -0.20020020020020018
      ],
      [
        0.0,
        0.0,
        -1.0,
        0.0
      ]
    ]
  }
}



def third_test():
    return {
  "background":{
    "background_1": {
      "example_texture": "woods_background",
    }
  },
  "meshes": {
    "test_cube": {
      "primitive": "cube"
    },
    "test_sphere": {
      "primitive": "sphere"
    }
  },
  "materials": {
    "spinner": {
      "example_texture": "spinner_tex",
      "ka": 1.0,
      "kd": 1.0,
      "ks": 0.5,
      "shine": 4.0
    },
    "candy_color": {
      "example_texture": "candy_color2",
      "ka": 1.0,
      "kd": 1.0,
      "ks": 0.5,
      "shine": 4.0
    }
  },
  "instances": {
    "cube1": {
      "mesh_name": "test_cube",
      "material_name": "spinner",
      "transform": [
        [
          0.707,
          0.0,
          -0.707,
          0.0
        ],
        [
          0.0,
          1.0,
          0.0,
          0.0
        ],
        [
          0.707,
          0.0,
          0.707,
          0.0
        ],
        [
          0.0,
          0.0,
          0.0,
          1.0
        ]
      ],
      "mask_color": [
        0,
        0,
        0
      ]
    },
    "sphere1": {
      "mesh_name": "test_sphere",
      "material_name": "candy_color",
      "transform": [
        [
          0.707,
          0.0,
          -0.707,
          0.0
        ],
        [
          0.0,
          1.0,
          0.0,
          0.0
        ],
        [
          -0.707,
          0.0,
          -0.707,
          -4.0
        ],
        [
          0.0,
          0.0,
          0.0,
          1.0
        ]
      ],
      "mask_color": [
        1,
        1,
        1
      ]
    }
  },
  "ambient_color": [
    0.2,
    0.2,
    0.2
  ],
  "point_lights": {},
  "direction_lights": {
    "light_main": {
      "direction": [
        0.5,
        0.0,
        -0.866
      ],
      "color": [
        1,
        1,
        1
      ]
    }
  },
  "camera": {
    "pose": [
      [
        1,
        0,
        0,
        0
      ],
      [
        0,
        1,
        0,
        0
      ],
      [
        0,
        0,
        1,
        0
      ],
      [
        0,
        0,
        0,
        1
      ]
    ],
    "projection": [
      [
        1.73205081,
        0.0,
        0.0,
        0.0
      ],
      [
        0.0,
        1.73205081,
        0.0,
        0.0
      ],
      [
        0.0,
        0.0,
        -1.002002002002002,
        -0.100100
      ],
      [
        0.0,
        0.0,
        -1.0,
        0.0
      ]
    ]
  }
}



fourth_test = {
  "image_lights":{
    "background_1": {
      "diffuse_textures":[
            "/home/awalsman/Development/renderpy/renderpy/example_background9/px_dif.jpg",
            "/home/awalsman/Development/renderpy/renderpy/example_background9/nx_dif.jpg",
            "/home/awalsman/Development/renderpy/renderpy/example_background9/py_dif.jpg",
            "/home/awalsman/Development/renderpy/renderpy/example_background9/ny_dif.jpg",
            "/home/awalsman/Development/renderpy/renderpy/example_background9/pz_dif.jpg",
            "/home/awalsman/Development/renderpy/renderpy/example_background9/nz_dif.jpg"],
      "reflection_textures":[
            "/home/awalsman/Development/renderpy/renderpy/example_background9/px_ref.jpg",
            "/home/awalsman/Development/renderpy/renderpy/example_background9/nx_ref.jpg",
            "/home/awalsman/Development/renderpy/renderpy/example_background9/py_ref.jpg",
            "/home/awalsman/Development/renderpy/renderpy/example_background9/ny_ref.jpg",
            "/home/awalsman/Development/renderpy/renderpy/example_background9/pz_ref.jpg",
            "/home/awalsman/Development/renderpy/renderpy/example_background9/nz_ref.jpg"],
      "reflection_mipmaps": None,
      "blur":2,
      "render_background":1,
      "diffuse_contrast":5,
      "diffuse_tint_lo":(0.0,0.0,0.0),
      "diffuse_tint_hi":(0.0,0.0,0.0),
      "reflect_tint":(0.0,0.0,0.0)
    }
  },
  "active_image_light" : "background_1",
  "meshes": {
    "test_cube": {
      "primitive": "cube"
    },
    "test_sphere": {
      "primitive": "sphere"
    },
    "test_ape": {
      "mesh_path": "/home/awalsman/Development/LINEMOD_Occlusion/OcclusionChallengeICCV2015/models/Ape/001.obj",
      "scale":40.0
    },
    "test_cracker": {
      "mesh_path" : "/home/awalsman/Development/cracker/meshes/003_cracker_box.obj",
      "scale":20.0
    }
  },
  "materials": {
    "cliff": {
      "example_texture": "bokeh",
      "ka": 1.0,
      "kd": 0.5,
      "ks": 0.8,
      "shine": 4.0,
      "image_light_kd": 1.0,
      "image_light_ks": 0.0,
      "image_light_blur_reflection": 4,
    },
    "white": {
      "example_texture": "white",
      "ka": 1.0,
      "kd": 0.5,
      "ks": 0.8,
      "shine": 4.0,
      "image_light_kd": 1.0,
      "image_light_ks": 0.0,
      "image_light_blur_reflection": 0,
    },
    "candy_color": {
      "example_texture": "candy_color2",
      "ka": 1.0,
      "kd": 0.5,
      "ks": 0.5,
      "shine": 4.0,
      "image_light_kd": 1.0,
      "image_light_ks": 0.0,
      "image_light_blur_reflection": 0.3,
    },
    "cracker_mat": {
      "texture": "/home/awalsman/Development/cracker/meshes/003_cracker_box.png",
      "ka":1.0,
      "kd":0.5,
      "ks":0.5,
      "shine":4.0,
      "image_light_kd": 0.9,
      "image_light_ks": 0.1,
      "image_light_blur_reflection": 4.0,
    },
  },
  "instances": {
    "sphere2": {
      "mesh_name": "test_cracker",
      "material_name": "cracker_mat",
      "transform": [
        [ 0.707, 0.000,-0.707, 5.000],
        [ 0.000, 1.000, 0.000, 0.000],
        [ 0.707, 0.000, 0.707,-4.000],
        [ 0.000, 0.000, 0.000, 1.000]],
      "mask_color": [ 0, 0, 0]
    },
    "xuan_0":{
        "mesh_name": "test_ape",
      "material_name": "candy_color",
      "transform": [
        [ 0.707, 0.000,-0.707, 5.000],
        [ 0.000, 1.000, 0.000, 0.000],
        [ 0.707, 0.000, 0.707,-4.000],
        [ 0.000, 0.000, 0.000, 1.000]],
      "mask_color": [ 0, 0, 0]
    },
    "xuan_1":{
        "mesh_name": "test_ape",
      "material_name": "candy_color",
      "transform": [
        [ 0.707, 0.000,-0.707, 0.000],
        [ 0.000, 1.000, 0.000,-5.000],
        [ 0.707, 0.000, 0.707,-4.000],
        [ 0.000, 0.000, 0.000, 1.000]],
      "mask_color": [ 0, 0, 0]
    },
    "xuan_2":{
        "mesh_name": "test_ape",
      "material_name": "candy_color",
      "transform": [
        [ 0.707, 0.000,-0.707, 10.0],
        [ 0.000, 1.000, 0.000, -5.0],
        [ 0.707, 0.000, 0.707, -4.0],
        [ 0.000, 0.000, 0.000,  1.0]],
      "mask_color": [ 0, 0, 0]
    },
    "xuan_3":{
        "mesh_name": "test_ape",
      "material_name": "candy_color",
      "transform": [
        [ 0.707, 0.000,-0.707,-5.000],
        [ 0.000, 1.000, 0.000, 0.000],
        [ 0.707, 0.000, 0.707,-4.000],
        [ 0.000, 0.000, 0.000, 1.000]],
      "mask_color": [ 0, 0, 0]
    },
    "sphere1": {
      "mesh_name": "test_sphere",
      "material_name": "white",
      "transform": [
        [ 0.707, 0.000,-0.707, 4.000],
        [ 0.000, 1.000, 0.000, 0.000],
        [-0.707, 0.000,-0.707, 0.000],
        [ 0.000, 0.000, 0.000, 1.000]],
      "mask_color": [ 1, 1, 1]
    }
  },
  "ambient_color": [ 0.0, 0.0, 0.0],
  "point_lights": {},
  "direction_lights": {},
  "camera": {
    "pose": [
      [ 1, 0, 0, 0],
      [ 0, 1, 0, 0],
      [ 0, 0, 1, 0],
      [ 0, 0, 0, 1]],
    "projection": camera.projection_matrix(math.radians(75), 1, far_clip=1000),
  }
}

