# system
import math

# numpy
import numpy

# local
import renderpy.core as core
import renderpy.camera as camera

def first_test(width, height):
    r = core.Renderpy()
    
    proj = camera.projection_matrix(math.radians(90), width/height)
    r.set_projection(proj)
    
    r.load_mesh(
            'test_cube',
            './example_meshes/cube.obj')

    r.load_mesh(
            'test_sphere',
            './example_meshes/sphere.obj')
    
    r.load_material(
            'spinner',
            './example_meshes/spinner_tex.png')
    
    r.load_material(
            'candy_color',
            './example_meshes/candy_color2.png')
    
    r.add_instance(
            'cube1',
            mesh_name = 'test_cube',
            material_name = 'spinner',
            transform = numpy.array(
                [[0.707,0,-0.707,0],[0,1,0,0],[0.707,0,0.707,0],[0,0,0,1]]))
    
    r.add_instance(
            'sphere1',
            mesh_name = 'test_sphere',
            material_name = 'candy_color',
            transform = numpy.array(
                [[0.707,0,-0.707,0],[0,1,0,0],[-0.707,0,-0.707,-4],[0,0,0,1]]),
            mask_color = numpy.array([1,1,1]))
    
    r.add_direction_light(
            name = 'light_main',
            direction = numpy.array([0.5,0,-0.866]),
            color = numpy.array([1,1,1]))

    r.set_ambient_color(numpy.array([0.2, 0.2, 0.2]))
    
    import json
    print(r.get_json_description(indent=2))
    
    return r


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



def fourth_test():
    return {
  "image_lights":{
    "background_1": {
      "diffuse_textures":[
            #"/home/awalsman/Development/cube_maps/overhead_area_diffuse/px.png",
            #"/home/awalsman/Development/cube_maps/overhead_area_diffuse/nx.png",
            #"/home/awalsman/Development/cube_maps/overhead_area_diffuse/py.png",
            #"/home/awalsman/Development/cube_maps/overhead_area_diffuse/ny.png",
            #"/home/awalsman/Development/cube_maps/overhead_area_diffuse/pz.png",
            #"/home/awalsman/Development/cube_maps/overhead_area_diffuse/nz.png"],
            #"/home/awalsman/Development/matterport_environments/f8f15aaf58354ce1b990df2ab33381bb/px_dif.jpg",
            #"/home/awalsman/Development/matterport_environments/f8f15aaf58354ce1b990df2ab33381bb/nx_dif.jpg",
            #"/home/awalsman/Development/matterport_environments/f8f15aaf58354ce1b990df2ab33381bb/py_dif.jpg",
            #"/home/awalsman/Development/matterport_environments/f8f15aaf58354ce1b990df2ab33381bb/ny_dif.jpg",
            #"/home/awalsman/Development/matterport_environments/f8f15aaf58354ce1b990df2ab33381bb/pz_dif.jpg",
            #"/home/awalsman/Development/matterport_environments/f8f15aaf58354ce1b990df2ab33381bb/nz_dif.jpg"],
            "/home/awalsman/Development/renderpy/renderpy/example_background9/px_dif.jpg",
            "/home/awalsman/Development/renderpy/renderpy/example_background9/nx_dif.jpg",
            "/home/awalsman/Development/renderpy/renderpy/example_background9/py_dif.jpg",
            "/home/awalsman/Development/renderpy/renderpy/example_background9/ny_dif.jpg",
            "/home/awalsman/Development/renderpy/renderpy/example_background9/pz_dif.jpg",
            "/home/awalsman/Development/renderpy/renderpy/example_background9/nz_dif.jpg"],
      "reflection_textures":[
            #"/home/awalsman/Development/cube_maps/overhead_area_diffuse/px.png",
            #"/home/awalsman/Development/cube_maps/overhead_area_diffuse/nx.png",
            #"/home/awalsman/Development/cube_maps/overhead_area_diffuse/py.png",
            #"/home/awalsman/Development/cube_maps/overhead_area_diffuse/ny.png",
            #"/home/awalsman/Development/cube_maps/overhead_area_diffuse/px.png",
            #"/home/awalsman/Development/cube_maps/overhead_area_diffuse/nz.png"],
            #"/home/awalsman/Development/matterport_environments/f8f15aaf58354ce1b990df2ab33381bb/px_ref.jpg",
            #"/home/awalsman/Development/matterport_environments/f8f15aaf58354ce1b990df2ab33381bb/nx_ref.jpg",
            #"/home/awalsman/Development/matterport_environments/f8f15aaf58354ce1b990df2ab33381bb/py_ref.jpg",
            #"/home/awalsman/Development/matterport_environments/f8f15aaf58354ce1b990df2ab33381bb/ny_ref.jpg",
            #"/home/awalsman/Development/matterport_environments/f8f15aaf58354ce1b990df2ab33381bb/pz_ref.jpg",
            #"/home/awalsman/Development/matterport_environments/f8f15aaf58354ce1b990df2ab33381bb/nz_ref.jpg"],
            "/home/awalsman/Development/renderpy/renderpy/example_background9/px_ref.jpg",
            "/home/awalsman/Development/renderpy/renderpy/example_background9/nx_ref.jpg",
            "/home/awalsman/Development/renderpy/renderpy/example_background9/py_ref.jpg",
            "/home/awalsman/Development/renderpy/renderpy/example_background9/ny_ref.jpg",
            "/home/awalsman/Development/renderpy/renderpy/example_background9/pz_ref.jpg",
            "/home/awalsman/Development/renderpy/renderpy/example_background9/nz_ref.jpg"],
      "reflection_mipmaps": None, #[
            #["/home/awalsman/Development/renderpy/renderpy/example_background/px_ref_%i.jpg"%i for i in range(1,8)],
            #["/home/awalsman/Development/renderpy/renderpy/example_background/nx_ref_%i.jpg"%i for i in range(1,8)],
            #["/home/awalsman/Development/renderpy/renderpy/example_background/py_ref_%i.jpg"%i for i in range(1,8)],
            #["/home/awalsman/Development/renderpy/renderpy/example_background/ny_ref_%i.jpg"%i for i in range(1,8)],
            #["/home/awalsman/Development/renderpy/renderpy/example_background/pz_ref_%i.jpg"%i for i in range(1,8)],
            #["/home/awalsman/Development/renderpy/renderpy/example_background/nz_ref_%i.jpg"%i for i in range(1,8)]],
      "blur":2,
      "render_background":1,
      "diffuse_contrast":5,
      #"diffuse_lo_rescale":0.0,
      #"diffuse_hi_rescale":1.0,
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
      #"mesh_path" : "/home/awalsman/Development/renderpy/renderpy/from_xuan/mesh0.obj",
        #"/home/awalsman/Downloads/from_adam/model_normalized.obj",
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
        0,
        0,
        0
      ]
    },
    "xuan_0":{
        "mesh_name": "test_ape",
      "material_name": "candy_color",
      "transform": [
        [
          0.707,
          0.0,
          -0.707,
          5.0
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
        0,
        0,
        0
      ]
    },
    "xuan_1":{
        "mesh_name": "test_ape",
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
          -5.0
        ],
        [
          0.707,
          0.0,
          0.707,
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
        0,
        0,
        0
      ]
    },
    "xuan_2":{
        "mesh_name": "test_ape",
      "material_name": "candy_color",
      "transform": [
        [
          0.707,
          0.0,
          -0.707,
          10.0
        ],
        [
          0.0,
          1.0,
          0.0,
          -5.0
        ],
        [
          0.707,
          0.0,
          0.707,
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
        0,
        0,
        0
      ]
    },
    "xuan_3":{
        "mesh_name": "test_ape",
      "material_name": "candy_color",
      "transform": [
        [
          0.707,
          0.0,
          -0.707,
          -5.0
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
        0,
        0,
        0
      ]
    },
    "sphere1": {
      "mesh_name": "test_sphere",
      "material_name": "white",
      "transform": [
        [
          0.707,
          0.0,
          -0.707,
          4.0
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
        1,
        1,
        1
      ]
    }
  },
  "ambient_color": [
    0.0,
    0.0,
    0.0
  ],
  "point_lights": {},
  "direction_lights": {},
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
    "projection": camera.projection_matrix(math.radians(75), 1, far_clip=1000),
      #  [
      #[
      #  1.33205081,
      #  0.0,
      #  0.0,
      #  0.0
      #],
      #[
      #  0.0,
      #  1.33205081,
      #  0.0,
      #  0.0
      #],
      #[
      #  0.0,
      #  0.0,
      #  -1.002002002002002,
      #  -0.100100
      #],
      #[
      #  0.0,
      #  0.0,
      #  -1.0,
      #  0.0
      #]
    #]
  }
}

