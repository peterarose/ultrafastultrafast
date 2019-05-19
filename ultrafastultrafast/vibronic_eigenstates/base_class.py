import yaml
import os
import numpy as np
import itertools

class DataOrganizer:
    def __init__(self,parameter_file_path):
        self.base_path = parameter_file_path
        self.load_params()

    def load_params(self):
        with open(os.path.join(self.base_path,'params.yaml')) as yamlstream:
            self.params = yaml.load(yamlstream,Loader=yaml.SafeLoader)

    def get_closest_index_and_value(self,value,array):
        index = np.argmin(np.abs(array - value))
        value = array[index]
        return index, value
