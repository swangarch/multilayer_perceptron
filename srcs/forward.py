import math
import numpy as np
from numpy import ndarray as array


def relu(value: float):
	return value if value >= 0 else 0


def sigmoid(value: array):
	return 1.0 / (1.0 + math.e ** -value)


def forward(arr1:array, arr2:array, bias:float):

	print("[ARR1]\n", arr1)
	print("[ARR2]\n", arr2)

	res = arr1 @ arr2 + bias
	activ = sigmoid(res)
	return activ


def back_probagation(nets:array, truth:array, res:list):
	length = len(nets)

	for i in range(length):
		# result = res[length - i - 1]
		# print(result)
		# diff = result - truth #[2d vector]
		print(f"\033[32mWEIGHT layer[{length - i - 1}]\n", nets[length - i - 1])
		# print("[DIFF]", diff)
	print("\033[0m", end="")


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
	truth = np.array([0.28, 0.72])
	activ = input
	nets = network((3, 5, 5, 5, 2))
	actives = [activ]
	for i in range(len(nets)):
		activ = forward(activ, nets[i], 0.5)
		actives.append(activ)
		print("\033[33m[RES]", activ, "\033[0m")

	print(actives)
	back_probagation(nets, truth, actives)


if __name__ == "__main__":
	main()