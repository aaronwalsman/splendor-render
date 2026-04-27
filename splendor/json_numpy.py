"""JSON encoder that handles numpy arrays and scalars."""
import json

import numpy

'''
json.dump(data, f, cls=NumpyEncoder)
'''

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that converts numpy arrays to lists and numpy scalars to Python scalars."""
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):
            return obj.item()
        return json.JSONEncoder.default(self, obj)
