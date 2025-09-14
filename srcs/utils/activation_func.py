from numpy import ndarray as array
import math
import numpy as np


def no_activ(value: array):
	return value


def relu(value: array):
	return np.maximum(0, value)


def relu_deriv(value: array):
	return (value > 0).astype(float)


def sigmoid(value: array):
	return 1.0 / (1.0 + math.e ** -value)


def sigmoid_deriv(value: array):
	return value * (1 - value)


def activ_deriv(active_func: callable, value:array):
	if active_func is None:
		return 1

	deriv_func_map = dict()
	deriv_func_map[relu] = relu_deriv
	deriv_func_map[sigmoid] = sigmoid_deriv
	
	deriv_func = deriv_func_map[active_func]
	if deriv_func is None:
		return 1
	return deriv_func(value)