import dill
import os
import sys

root = "cache_nbcf"


class Serialized:

    def __init__(self):
        if not os.path.exists(root):
            os.makedirs(root)

    def save_object(self, object, object_name):
        sys.setrecursionlimit(10000)
        if not os.path.exists(object_name):
            os.makedirs(object_name)
        with open(f"./{root}/{object_name}.plk", "wb") as file_pkl:
            dill.dump(object, file_pkl)

    def load_object(self, object_name):
        if not os.path.exists(object_name):
            raise FileNotFoundError(f"File {object_name} not found")
        with open(f"./{root}/{object_name}.plk", "rb") as file_pkl:
            return dill.load(file_pkl)
