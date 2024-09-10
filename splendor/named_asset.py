from splendor.session import RenderSession

class NamedAsset:
    
    @staticmethod
    def list_assets():
        return set(_loaded_assets.keys())
    
    @staticmethod
    def lookup_loaded_name(name):
        return _loaded_assets[name]
    
    def __init__(self, name=None):
        
        # if no name was specified, construct a unique name automatically
        if name is None:
            while True:
                suffix = len(self._loaded_assets)
                name = f'{self.__class__.__name__}_{suffix}'
                if name not in self._loaded_assets:
                    break
                suffix += 1
        
        # make sure the specified name does not already exist
        else:
            assert name not in self._loaded_assets, (
                f'{self.ASSET_CLASS_NAME} named "{name}" already exists')
        
        self.name = name
        self._loaded_assets[name] = self
        
        #active_session = RenderSession.active_session
        #active_session.add_asset(self)
    
    def __del__(self):
        del(self._loaded_assets[self.name])
    
    def __str__(self):
        return self.name

'''
    @classmethod
    def load_asset(cls, asset):
        asset_path = asset_library[self.ASSET_NAME][asset]
        cls.load_path(asset, asset_path)

class TestAsset(NamedAsset):
    def __init__(self, name, asset):
        super().__init__(name)
        self.load_asset(asset)
    
    @classmethod
    def load_path(cls, name, asset):
        print(name, asset)

if __name__ == '__main__':
    TestAsset.load_asset('help', 'me_jon_keto')
'''
