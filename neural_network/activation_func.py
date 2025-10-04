#!/usr/bin/python3

from numpy import ndarray as array
import math
import numpy as np


def relu(value: array):
	return np.maximum(0, value)


def relu_deriv(value: array):
	return (value > 0).astype(float)


def leaky_relu(value: array):
	return np.maximum(0.02 * value, value)


def leaky_relu_deriv(value: array):
	return np.where(value > 0, 1.0, 0.02)


def sigmoid(value: array):
	return 1.0 / (1.0 + math.e ** -value)


def softmax(value: array):
	exp = np.exp(value - np.max(value, axis=0, keepdims=True))
	sum = np.sum(exp, axis=0, keepdims=True)
	soft = exp / sum
	return soft


def sigmoid_deriv(value: array):
	return value * (1 - value)


def activ_deriv(active_func: callable, value:array, deriv_map: dict):
	"""Get the derivative of an activation function."""

	if active_func is None:
		return 1
	if active_func == softmax:
		raise ValueError("Softmax derivative not implemented")
	deriv_func = deriv_map[active_func]
	if deriv_func is None:
		return 1
	return deriv_func(value)


def get_activation_funcs_by_name(activs):
	"""Get activation funcs by name, convert string to actual function."""

	activmap = {
		"relu":relu,
		"sigmoid": sigmoid,
		"softmax": softmax,
		"leaky_relu": leaky_relu,
		"none": None
	}
	return [activmap[a] for a in activs]