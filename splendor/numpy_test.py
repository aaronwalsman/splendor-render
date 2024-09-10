import numpy as np

'''
Want: a numpy array with a dirty flag for whenever the data changes
None of the below works, come back later
'''

class DirtyArray(np.lib.mixins.NDArrayOperatorsMixin):
    #def __new__(cls, input_array):
    #    obj = np.asarray(input_array).view(cls)
    #    obj.dirty=True
    #    return obj
    
    def __init__(self, *args, **kwargs):
        self.dirty = True
        super().__init__(*args, **kwargs)
    
    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.dirty = getattr(obj, 'dirty', True)
    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        self.dirty = True
        return super().__array_ufunc__(ufunc, method, *inputs, **kwargs)
    
    def __array_func__(self, *args, **kwargs):
        self.dirty = True
        return super().__array_func__(*args, **kwargs)
