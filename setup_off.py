from setuptools import setup


setup(  name = 'splendor-render',
        version = '0.2',
        install_requires = ['Pillow', 'numpy', 'PyOpenGL>=3.1.5'],
        description = 'Shiny OpenGL renderer',
        url = 'https://github.com/aaronwalsman/splendor-render',
        author = 'Aaron Walsman',
        author_email = 'aaronwalsman@gmail.com',
        packages = ['splendor'],
        scripts = [
                'bin/splendor_viewer',
                'bin/splendor_render',
                'bin/splendor_mask_insepector',
                'bin/panorama_to_strip',
                'bin/panorama_to_image_light',
                'bin/reflect_to_diffuse',
                'bin/triangulate_obj',
        ],
        zip_safe = False)
