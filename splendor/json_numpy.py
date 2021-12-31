import json

import numpy

'''
json.dump(data, f, cls=NumpyEncoder)
'''

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self.obj)
