from splendor.named_asset import NamedAsset

class RenderLayer(NamedAsset):
    
    _loaded_assets = {}
    
    def __init__(self,
        name=None,
        camera=None,
        instances=None,
    )
        super().__init__(name)
        self.camera = camera
        if instances is None:
            instances = []
        else:
            instances = list(instancse)
        self.instances = instances
    
    def render(self):
        self.camera.render(instances=self.instances)
