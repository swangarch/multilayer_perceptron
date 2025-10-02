from numpy import ndarray as array
import math
import numpy as np
import sys


def relu(value: array):
	return np.maximum(0, value)


def relu_deriv(value: array):
	return (value > 0).astype(float)


def sigmoid(value: array):
	return 1.0 / (1.0 + math.e ** -value)


def softmax(value: array):
	# print(value.shape)
	# sys.exit(1)
	exp_ = np.exp(value - np.max(value, axis=0, keepdims=True))
	sum_ = np.sum(exp_, axis=0, keepdims=True)
	soft_ = exp_ / sum_
	return soft_
 

def sigmoid_deriv(value: array):
	return value * (1 - value)


def activ_deriv(active_func: callable, value:array, deriv_map: dict):
	if active_func is None:
		return 1
	
	if active_func == softmax:
		raise ValueError("Softmax derivative not implemented")
	
	deriv_func = deriv_map[active_func]
	if deriv_func is None:
		return 1
	return deriv_func(value)


def get_activation_funcs_by_name(activs):
    activmap = {
        "relu":relu,
        "sigmoid": sigmoid,
		"softmax": softmax,
        "none": None
    }
    return [activmap[a] for a in activs]