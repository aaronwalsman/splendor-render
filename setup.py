from setuptools import setup

setup(  name = 'renderpy',
        version = '0.2',
        install_requires = ['numpy', 'PyOpenGL>=3.1.5'],
        description = 'Bare-bones python OpenGL renderer for textured meshes',
        url = 'https://gitlab.cs.washington.edu/awalsman/renderpy',
        author = 'Aaron Walsman',
        author_email = 'awalsman@cs.washington.edu',
        packages = ['renderpy'],
        scripts = [
                'bin/drpy_viewer',
                'bin/drpy_mask_viewer',
                'bin/drpy_render',
                'bin/panorama_to_strip',
                'bin/panorama_to_image_light',
                'bin/reflect_to_diffuse'
        ],
        zip_safe = False)
