import math
import numpy as np
from numpy import ndarray as array


def relu(value: float):
	return value if value >= 0 else 0


def sigmoid(value: array):
	return 1.0 / (1.0 + math.e ** -value)


def weighted_sum(li1, li2, bias):
	return sum([li1[i] * li2[i] for i in range(len(li1))]) + bias


def forward(arr1:array, arr2:array, bias:float):

	print("[ARR1]\n", arr1)
	print("[ARR2]\n", arr2)

	res = arr1 @ arr2 + bias
	activ = sigmoid(res)
	return activ


def network(net: tuple):
	netWeight = []
	if (len(net) < 4):
		raise("Invalid network structure.")
	for i in range(len(net) - 1):
		matrix = np.zeros(shape=(net[i], net[i + 1]), dtype=np.float32)
		netWeight.append(matrix)

	return netWeight


def main():
	input = np.array([0.4, 0.3, 0.8])
	activ = input
	nets = network((3, 5, 5, 5, 2))
	for i in range(len(nets) - 1):
		activ = forward(activ, nets[i], 0.5)
		print("\033[33m[RES]", activ, "\033[0m")


if __name__ == "__main__":
	main()