import os
import configparser

renderpy_root_path = os.path.join(os.path.dirname(__file__), '..')
default_config_path = os.path.join(renderpy_root_path, 'defaults.cfg')

asset_types = ('image_lights', 'meshes', 'materials', 'panoramas', 'scenes')
asset_extensions = {
        'image_lights' : (),
        'meshes' : ('.obj',),
        'materials' : (),
        'panoramas' : ('.jpg', '.png'),
        'scenes' : ('.json',)
}

class AssetError(Exception):
    pass

class PathFinder:
    def __init__(paths, asset_type):
        self.paths = paths
        self.asset_type = asset_type
    
    def __getitem__(self, key):
        # if an absolute path was provided to name, then just use that
        if os.path.isabs(key):
            return key
        
        # otherwise search the paths
        for path in self.paths:
            # if the key was provided with an extension, just use that
            extensionless_path = os.path.join(path, key)
            if os.path.exists(extensionless_path):
                return extensionless_path
            for extension in extensions:
                extension_path = os.path.join(path, key + extension)
                if os.path.exists(extension_path):
                    return extension_path
        
        raise AssetError('No %s named "%s" found'%(self.asset_type, key))

class AssetLibrary:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = default_config_path
        if not os.path.exists(config_path):
            raise AssetError('Config path not found: %s'%config_path)
        
        self.load_config(config_path, clear=True)
    
    @static_method
    def path_list_to_paths(path_list, HERE):
        paths = path_list.split(',')
        paths = [path.replace('{HERE}', HERE)
                for path in paths]
        return paths
    
    def load_config(self, config_path, clear=False)
        if clear:
            self.directories = {asset_type : [] for asset_type in asset_types}
        
        parser = configparser.ConfigParser()
        parser.read(config_path)
        HERE = os.path.dirname(config_path)
        
        for asset_type in asset_types:
            if asset_type in parser['paths']:
                self.directories[asset_type].extend(self.path_list_to_paths(
                        parser['paths'][asset_type], HERE)
    
    def __getitem__(self, asset_type):
        return PathFinder(self.directories[asset_type], asset_type)

'''
def resolve_path(path):
    if os.path.isabs(path):
        return path
    else:
        return os.path.abspath(os.path.join(renderpy_root_path, path))

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
