import os
import configparser

root_path = os.path.join(os.path.dirname(__file__), '..')
setup_cfg_path = os.path.join(root_path, 'setup.cfg')

parser = configparser.ConfigParser()
parser.read(setup_cfg_path)

def resolve_path(path):
    if os.path.isabs(path):
        return path
    else:
        return os.path.abspath(os.path.join(root_path, path))

def get_asset_files(path, extensions):
    return {
            name : os.path.join(path, file_name)
            for file_name, (name, extension)
            in [(fp, os.path.splitext(fp)) for fp in os.listdir(path)]
            if extension.lower() in extensions}

image_lights = get_asset_files(
        resolve_path(parser['paths']['image_lights']), ('',))

meshes = get_asset_files(
        resolve_path(parser['paths']['meshes']), ('.obj',))

panoramas = get_asset_files(
        resolve_path(parser['paths']['panoramas']), ('.png', '.jpg'))

scenes = get_asset_files(
        resolve_path(parser['paths']['scenes']), ('.json',))

'''
image_light_path = resolve_path(parser['paths']['image_lights'])
image_lights = {
        file_name : os.path.join(image_light_path, file_name)
        for file_name in os.listdir(image_light_path)}

mesh_path = resolve_path(parser['paths']['meshes'])
meshes = {
        name : os.path.join(mesh_path, file_name)
        for file_name, (name, extension)
        in [(fp, os.path.splitext(fp)) for fp in os.listdir(mesh_path)]
        if extension.lower() == '.obj'}

panorama_path = resolve_path(parser['paths']['meshes'])
panoramas = {
        name : os.path.join(panorama_path, file_name)
        for file_name, (name, extension)
        in [(fp, 
'''
