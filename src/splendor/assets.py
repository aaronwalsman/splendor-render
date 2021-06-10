import os
import configparser
import zipfile

from splendor.download import download, agree_to_zip_licenses
from splendor.home import get_splendor_home, make_splendor_home
from splendor.exceptions import SplendorAssetException

splendor_home = get_splendor_home()
#splendor_module_path = os.path.join(os.path.dirname(__file__))
#default_assets_path = os.path.join(splendor_home, 'default_assets.cfg')

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

def install_assets(url, destination=splendor_home):
    print('='*80)
    print('Installing: %s'%url)
    make_splendor_home()
    
    print('-'*80)
    asset_path = os.path.join(destination, 'assets.zip')
    downloaded_path = download(url, asset_path, overwrite=True)
    
    print('-'*80)
    print('Checking for Licenses')
    if agree_to_zip_licenses(downloaded_path):
        print('Extracting Contents To: %s'%destination)
        with zipfile.ZipFile(downloaded_path, 'r') as z:
            z.extractall(destination)
    else:
        print('Must Agree to All Licensing.  Aborting.')
    
    os.remove(downloaded_path)

def default_assets_installed():
    return 'default_assets.cfg' in os.path.listdir(splendor_home)

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
    def __init__(self, asset_packages=None):
        self.clear()
        if asset_packages is None:
            if default_assets_installed:
                asset_packages = ['default_assets']
            else:   
                asset_packages = []
        else:
            if isinstance(asset_packages, str):
                asset_packages = asset_packages.split(',')
        for asset_package in asset_packages:
            if os.path.exists(asset_package):
                asset_package_cfg = asset_package
            else:
                home_files = os.listdir(splendor_home)
                if asset_package + '.cfg' in home_files:
                    asset_package_cfg = os.path.join(
                        splendor_home, asset_package + '.cfg')
                elif asset_package in home_files:
                    asset_package_cfg = os.path.join(
                        splendor_home, asset_package)
                else:
                    raise SplendorAssetException(
                        'Config path not found: %s'%config_path)
            self.load_config(asset_package_cfg, clear=False)
    
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
