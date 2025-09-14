from numpy import ndarray as array
import numpy as np


def create_bias(net_shape:tuple, value):
    """Create bias for given network shape."""

    biases = []
    for num in net_shape[1:]:
        biases.append(np.full((num), value, dtype=np.float32))
    return biases


def forward_layer(arr1:array, arr2:array, biases:array, activ_func):
    """Perform layer forwarding."""

    res = arr1 @ arr2 + biases
    if activ_func is None:
        return res
    else: 
        return activ_func(res)


def init_matrix(shapes: tuple, value:float, random=False):
    """Init matrix in given shapes."""

    matrixs = []
    if (len(shapes) < 4):
        raise("Invalid network structure.")
    for i in range(len(shapes) - 1):
        if random == False:
            matrix = np.full((shapes[i], shapes[i + 1]), value, dtype=np.float32)
        else:
            matrix = np.random.randn(shapes[i], shapes[i + 1]) * np.sqrt(2.0 / shapes[i])
        matrixs.append(matrix)
    return matrixs


def network(net: tuple):
    """Init network weights."""

    return init_matrix(net, 0.1, True)


def mean_gradients(net_structure, Wgrads, Bgrads, num_data):
    """Calcualte mean gradient for weights and biases."""

    len_nets = len(net_structure) - 1
    Wgrads_mean = init_matrix(net_structure, 0.0)
    Bgrads_mean = create_bias(net_structure, 0)
    for idx, Wgrad in enumerate(Wgrads):
        for i in range(len_nets):
            Wgrads_mean[i] += np.transpose(Wgrad[len_nets - i - 1])
            Bgrads_mean[i] += Bgrads[idx][len_nets - i - 1]
    for i in range(len_nets):
        Wgrads_mean[i] /= num_data
        Bgrads_mean[i] /= num_data
    return Wgrads_mean, Bgrads_mean


def gradient_descent(nets, biases, Wgrads_mean, Bgrads_mean, learning_rate):
    """Perform gradient descent for weights and biases."""

    len_nets = len(Wgrads_mean)
    for i in range(len_nets):
        nets[i] -= learning_rate * Wgrads_mean[i]
        biases[i] -= learning_rate * Bgrads_mean[i]


def loss(func, truth, predict):
    """Calculate loss."""

    return func(truth, predict)


def mse_loss(truth: array, predict: array):
    """Calculate mse loss."""

    return 0.5 * np.mean((truth - predict) ** 2)
