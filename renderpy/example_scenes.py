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

