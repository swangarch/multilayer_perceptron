from numpy import ndarray as array
import math
import numpy as np


def relu(value: array):
	return np.maximum(0, value)


def relu_deriv(value: array):
	return (value > 0).astype(float)


def sigmoid(value: array):
	return 1.0 / (1.0 + math.e ** -value)


def sigmoid_deriv(value: array):
	return value * (1 - value)


def activ_deriv(active_func: callable, value:array, deriv_map: dict):
	if active_func is None:
		return 1
	
	deriv_func = deriv_map[active_func]
	if deriv_func is None:
		return 1
	return deriv_func(value)