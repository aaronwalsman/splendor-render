from setuptools import setup

setup(  name = 'renderpy',
        version = '0.2',
        install_requires = ['numpy', 'PyOpenGL'],
        description = 'Bare-bones python OpenGL renderer for textured meshes',
        url = 'https://gitlab.cs.washington.edu/awalsman/renderpy',
        author = 'Aaron Walsman',
        author_email = 'awalsman@cs.washington.edu',
        packages = ['renderpy'],
        scripts = [
                'bin/drpy_viewer',
                'bin/drpy_mask_viewer',
                'bin/drpy_render',
                'bin/panorama_to_cube',
                'bin/reflection_to_diffuse'
        ],
        zip_safe = False)
