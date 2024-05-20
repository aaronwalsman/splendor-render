class AssetLibrary:
    def __init__(self, asset_packages=None):
        self.clear()
        
        # use default assets if None was specified
        if asset_packages is None:
            if example_assets_installed:
                asset_packages = ['example_assets']
            else:
                asset_packages = []
        else:
            if isinstance(asset_packages, str):
                asset_packages = asset_packages.split(',')
    
    def clear(self):
        self.directories = {asset_type : [] for asset_type in asset_types}
