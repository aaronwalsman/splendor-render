import setuptools

with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='splendor-render',
    version='0.3.3',
    author='Aaron Walsman',
    author_email='aaronwalsman@gmail.com',
    description='A fancy shiny python renderer',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/aaronwalsman/splendor-render',
    install_requires=[
        'Pillow',
        'numpy',
        'gdown',
        'PyOpenGL>=3.1.5',
        'tqdm',
    ],
    packages=setuptools.find_packages(),
    entry_points={
        'console_scripts':[
            'splendor_asset_installer=splendor.scripts.'
                'splendor_asset_installer:main',
            'splendor_viewer=splendor.scripts.splendor_viewer:main',
            'splendor_render=splendor.scripts.splendor_render:main',
            'splendor_mask_inspector=splendor.scripts.splendor_mask_inspector:'
                'main',
            'panorama_to_strip=splendor.scripts.panorama_to_strip:main',
            'panorama_to_image_light=splendor.scripts.panorama_to_image_light:'
                'main',
            'reflect_to_diffuse=splendor.scripts.reflect_to_diffuse:main',
            'triangulate_obj=splendor.scripts.triangulate_obj:main',
        ]
    },
    python_requires='>=3.6',
    classifiers = [
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
