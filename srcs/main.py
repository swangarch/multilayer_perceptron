import numpy as np
from numpy import ndarray as array
from utils.activation_func import relu, relu_deriv, sigmoid, sigmoid_deriv, activ_deriv
from utils.data_generation import generate_data
from utils.nn import forward_layer, network, mse_loss, create_bias, gradient_descent, mean_gradients
import matplotlib.pyplot as plt
from datetime import datetime
import os
from utils.mlpClass import Mlp


def main():
    net_shape = (1, 64, 32, 1)
    # activation_funcs = (relu, None)
    activation_funcs = (relu, relu, None)
    # net_shape = (3, 64, 32, 2)
    # activation_funcs = (relu, relu, sigmoid)

    mlp = Mlp(net_shape, activation_funcs)

    inputs, truths = generate_data(142, 100)
    mlp.train(inputs, truths, 10000, 0.05)

    test_inputs, test_truths = generate_data(123, 80)
    mlp.test(test_inputs, test_truths)
    mlp.show_loss()

    # mlp.save_weight()


if __name__ == "__main__":
    main()