class NamedAsset:
    _loaded_assets = {}
    
    @staticmethod
    def all_loaded_names():
        return set(_loaded_assets.keys())
    
    @staticmethod
    def lookup_loaded_name(name):
        return _loaded_assets[name]
    
    def __init__(self, data, name=None):
        if name is None:
            i = str(len(self._loaded_assets))
        else:
            assert name not in self._loaded_assets, (
                f'asset named "{name}" already exists')
        self.data = data
        self.name = name
        self._loaded_assets[name] = self
    
    def __del__(self):
        del(self._loaded_assets[self.name])
    
    @property
    def data(self):
        return self._data
    
    @data.setter(self, data):
        self._data = self.validate_data(data)
        self.update_gl_data()
    
    @staticmethod
    def validate_data(data):
        return data
    
    def update_gl_data(self):
        pass
    
    def cleanup_gl_data(self):
        pass

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
