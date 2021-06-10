import os
import configparser

from splendor.exceptions import SplendorAssetException

splendor_module_path = os.path.join(os.path.dirname(__file__))
default_assets_path = os.path.join(splendor_module_path, 'default_assets.cfg')

asset_types = (
        'image_lights',
        'meshes',
        'materials',
        'textures',
        'cubemaps',
        'panoramas',
        'scenes')
asset_extensions = {
        'image_lights' : ('.jpg', '.png'),
        'meshes' : ('.obj',),
        'textures' : ('.jpg', '.png'),
        'cubemaps' : ('.jpg', '.png'),
        'materials' : (),
        'panoramas' : ('.jpg', '.png'),
        'scenes' : ('.json',)
}

class PathFinder:
    def __init__(self, paths, asset_type):
        self.paths = paths
        self.asset_type = asset_type
    
    def __getitem__(self, key):
        # if the key exists relative to the current directory
        # or is an absolute path, then use that
        if os.path.exists(key):
            return key
        
        # otherwise search the paths
        for path in self.paths:
            # if the key was provided with an extension, just use that
            extensionless_path = os.path.join(path, key)
            if os.path.exists(extensionless_path):
                return extensionless_path
            for extension in asset_extensions[self.asset_type]:
                extension_path = os.path.join(path, key + extension)
                if os.path.exists(extension_path):
                    return extension_path
        
        raise SplendorAssetException(
            'No %s named "%s" found'%(self.asset_type, key))

class AssetLibrary:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = default_assets_path
        
        self.clear()
        config_paths = config_path.split(',')
        for config_path in config_paths:
            if config_path == 'DEFAULT':
                config_path = default_assets_path
            if not os.path.exists(config_path):
                raise SplendorAssetException(
                    'Config path not found: %s'%config_path)
            
            self.load_config(config_path, clear=False)
    
    @staticmethod
    def path_list_to_paths(path_list, HERE):
        paths = path_list.split(',')
        paths = [path.format(HERE=HERE) for path in paths]
        return paths
    
    def clear(self):
        self.directories = {asset_type : [] for asset_type in asset_types}
    
    def load_config(self, config_path, clear=False):
        if clear:
            self.clear()
        
        parser = configparser.ConfigParser()
        parser.read(config_path)
        HERE = os.path.dirname(os.path.abspath(config_path))
        
        for asset_type in asset_types:
            if asset_type in parser['paths']:
                paths = self.path_list_to_paths(
                        parser['paths'][asset_type], HERE)
                self.directories[asset_type].extend(paths)
    
    def __getitem__(self, asset_type):
        return PathFinder(self.directories[asset_type], asset_type)
