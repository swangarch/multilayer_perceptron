from .utils.activation_func import relu, sigmoid
from .utils.data_generation import generate_data_1d, generate_data_3d
from .utils.nnClass import NN


__all__ = ["NN", relu, sigmoid, generate_data_1d, generate_data_3d]