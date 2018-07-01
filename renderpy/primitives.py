import os

root = os.path.abspath(os.path.dirname(__file__))

primitive_paths = {
        file_name.replace('.obj', '') :
            os.path.join(root, 'primitive_meshes', file_name)
            for file_name in os.listdir(os.path.join(root, 'primitive_meshes'))}
