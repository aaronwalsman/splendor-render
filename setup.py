from setuptools import setup

setup(  name = 'renderpy',
        version = '0.2',
        description = 'Bare-bones python OpenGL renderer for textured meshes',
        url = 'https://gitlab.cs.washington.edu/awalsman/renderpy',
        author = 'Aaron Walsman',
        author_email = 'awalsman@cs.washington.edu',
        packages = ['renderpy'],
        scripts = [
                'bin/panorama_to_cube'
        ],
        zip_safe = False)
