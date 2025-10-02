import numpy as np
import math


def generate_data_rand(seed, number, noise_scale=0.2):
    """Generate 1d data based on case 5 with added random noise."""

    np.random.seed(seed)
    inputs = []
    truths = []
    for _ in range(number):
        x = np.random.rand()
        inputs.append(np.array([x]))
        # y = 0.5 * x**2 + 0.3 * np.sin(2 * np.pi * x) + 0.2 * np.exp(-2*x)
        # y = 2*x
        # y = x * math.sin(x)
        # y = x * (1 - math.exp(-x**2))
        y = math.sin(2*x) * math.exp(-0.1*x) + 0.5 * math.cos(5*x)
        # y = 0.5 * x**2 + 0.3 * np.sin(2 * np.pi * x)
        # y = - 0.5 * (x - 0.5)**2 + 0.3 
        noise = np.random.normal(0, noise_scale)
        y_noisy = y + noise
        truths.append(np.array([float(y_noisy)]))

    inputs = np.array(inputs)
    truths = np.array(truths)
    return inputs, truths


    