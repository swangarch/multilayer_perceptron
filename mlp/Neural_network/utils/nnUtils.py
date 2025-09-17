from numpy import ndarray as array
import numpy as np


def create_bias(net_shape:tuple, value):
    """Create bias for given network shape."""

    biases = []
    for num in net_shape[1:]:
        biases.append(np.full(((num), 1), value, dtype=np.float32))
    return biases


def create_actives(net_shape:tuple, value):
    """Create bias for given network shape."""

    actives = []
    for num in net_shape:
        actives.append(np.full(((num), 1), value, dtype=np.float32))
    return actives


def forward_layer(weights:array, activations:array, biases:array, activ_func):
    """Perform layer forwarding."""

    res = weights @ activations + biases
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
            matrix = np.full((shapes[i + 1], shapes[i]), value, dtype=np.float32)
        else:
            matrix = np.random.randn(shapes[i + 1], shapes[i]) * np.sqrt(2.0 / shapes[i])
        matrixs.append(matrix)
    return matrixs


def network(net: tuple):
    """Init network weights."""

    return init_matrix(net, 0.1, True)


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


def accuracy_1d(truth: array, predict: array):
    """"""
    truth_flat = truth.reshape(-1)
    predict_flat = predict.reshape(-1)
    correct = np.sum(truth_flat == predict_flat)
    total = truth_flat.shape[0]
    return correct / total


def shuffle_data(inputs, truths):
    """Random shuffle the inputs and outputs data"""

    indices = np.arange(len(inputs))
    np.random.shuffle(indices)
    inputs_shuffled = inputs[indices]
    truths_shuffled = truths[indices]
    return inputs_shuffled, truths_shuffled


def split_dataset(inputs, truths, ratio=0.8):
    inputs, truths = shuffle_data(inputs, truths) #random shuffle
    num_data = len(inputs)
    inputs_train = inputs[: int(num_data * ratio) - 1]
    truths_train = truths[: int(num_data * ratio) - 1]
    inputs_test = inputs[int(num_data * ratio) -1 :]
    truths_test = truths[int(num_data * ratio) -1 :]

    return inputs_train, truths_train, inputs_test, truths_test


