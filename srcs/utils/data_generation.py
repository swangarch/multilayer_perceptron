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
        inputs.append(x)

        y1 = 0.3*x[0] + 0.5*x[1] + 0.2*x[2]
        y2 = x[0] * x[1] + 0.2 * x[2] ** 2 + 0.1 * np.sin(np.pi * x[0])
        truths.append(np.array([y1, y2]))

    inputs = np.array(inputs)
    truths = np.array(truths)
    return inputs, truths


def generate_data_1d(seed, number):
    """Generate 1d data set for different types."""

    np.random.seed(seed)
    num = rd.randint(0, 5)
    inputs = []
    truths = []

    for _ in range(number):
        x = np.random.rand()
        inputs.append([x])
        
        y = 0
        if num % 5 == 0:
            y = 0.5 * x**2 + 0.3 * np.sin(2 * np.pi * x) + 0.2 * np.exp(-2*x)
        elif num % 5 == 1:
            y = 2*x
        elif num % 5 == 2:
            y = x * math.sin(x)
        elif num % 5 == 3:
            y = x * (1 - math.exp(-x**2))
        elif num % 5 == 4:
            y = math.sin(2*x) * math.exp(-0.1*x) + 0.5 * math.cos(5*x)
        truths.append([float(y)])

    inputs = np.array(inputs)
    truths = np.array(truths)
    return inputs, truths



    