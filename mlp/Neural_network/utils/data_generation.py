import numpy as np
import math
import random as rd


def generate_data_3d(seed, number):
    """Generate 3d dataset for different types."""

    np.random.seed(seed)
    inputs = []
    truths = []

    for _ in range(number):
        x = np.random.rand(3)
        inputs.append(x.reshape(-1, 1))

        y1 = 0.3*x[0] + 0.5*x[1] + 0.2*x[2]
        y2 = x[0] * x[1] + 0.2 * x[2] ** 2 + 0.1 * np.sin(np.pi * x[0])
        truths.append(np.array([y1, y2]).reshape(-1, 1))

    inputs = np.array(inputs)
    truths = np.array(truths)
    return inputs, truths


def generate_data_1d(seed, number, option=None):
    """Generate 1d data set for different types."""

    np.random.seed(seed)
    num = rd.randint(0, 5)
    inputs = []
    truths = []

    if option is not None:
        num = option

    for _ in range(number):
        x = np.random.rand()
        inputs.append(np.array([x]).reshape(-1, 1))
        
        y = 0
        if num % 6 == 0:
            y = 0.5 * x**2 + 0.3 * np.sin(2 * np.pi * x) + 0.2 * np.exp(-2*x)
        elif num % 6 == 1:
            y = 2*x
        elif num % 6 == 2:
            y = x * math.sin(x)
        elif num % 6 == 3:
            y = x * (1 - math.exp(-x**2))
        elif num % 6 == 4:
            y = math.sin(2*x) * math.exp(-0.1*x) + 0.5 * math.cos(5*x)
        elif num % 6 == 5:
            y = 0.5 * x**2 + 0.3 * np.sin(2 * np.pi * x)
        truths.append(np.array([float(y)]).reshape(-1, 1))

    inputs = np.array(inputs)
    truths = np.array(truths)
    return inputs, truths



    