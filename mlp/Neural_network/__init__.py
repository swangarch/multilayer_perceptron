from .utils.activation_func import relu, sigmoid
from .utils.data_generation import *
from .utils.nnClass import NN
from .utils.nnUtils import *


__all__ = ["NN", relu, sigmoid, generate_data_1d, generate_data_3d, generate_data_rand, split_dataset]