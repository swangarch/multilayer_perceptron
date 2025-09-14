import numpy as np
import matplotlib.pyplot as plt
import math


# def generate_data(seed, number):
#     np.random.seed(seed)  # 保证可复现

#     inputs = []
#     truths = []

#     for _ in range(number):
#         x = np.random.rand(3)  # 生成三个 0~1 的随机输入
#         inputs.append(x)

#         y1 = 0.3*x[0] + 0.5*x[1] + 0.2*x[2]
#         y2 = x[0] * x[1] + 0.2 * x[2] ** 2 + 0.1 * np.sin(np.pi * x[0])
#         truths.append(np.array([y1, y2]))

#     inputs = np.array(inputs)
#     truths = np.array(truths)

#     return inputs, truths, number


def generate_data(seed, number):
    np.random.seed(seed)  # 保证可复现
    
    inputs = []
    truths = []
    
    for _ in range(number):
        x = np.random.rand()  # 生成一个 0~1 的随机输入
        inputs.append([x])
        
        # y = 0.5 * x**2 + 0.3 * np.sin(2 * np.pi * x) + 0.2 * np.exp(-2*x)
        # y = 2*x
        # y = x * math.sin(x)
        # y = x * (1 - math.exp(-x**2))
        y = math.sin(2*x) * math.exp(-0.1*x) + 0.5 * math.cos(5*x)
        truths.append([float(y)])

    inputs = np.array(inputs)
    truths = np.array(truths)
    return inputs, truths



    