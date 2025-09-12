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


# def back_probagation(nets:array, truth:array, layer_values:list):
# 	length = len(nets)

# 	for i in range(length):
# 		curr_idx = len - i - 1
# 		prev_idx = curr_idx - 1
# 		if prev_idx < 0:
# 			break
# 		print(f"\033[32mWEIGHT layer[{length - i - 1}]\n", nets[length - i - 1])
# 	print("\033[0m", end="")


def network(net: tuple):
	netWeight = []
	if (len(net) < 4):
		raise("Invalid network structure.")
	for i in range(len(net) - 1):
		matrix = np.full((net[i], net[i + 1]), 0.33, dtype=np.float32)
		netWeight.append(matrix)

	return netWeight


def main():
	#-----------------------------init-----------------------------
	inputs = [np.array([0.4, 0.3, 0.8]), np.array([0.45, 0.21, 0.66]), np.array([0.95, 0.19, 0.16])]
	truths = [np.array([0.28, 0.72]), np.array([0.58, 0.62]), np.array([0.18, 0.42])]
	# nets = network((3, 5, 5, 5, 2))
	nets = network((3, 4, 4, 2))

	#-----------------------------forward-----------------------------
	active_neurons = []
	for input in inputs:
		activ = input
		actives_in_one = [activ]
		for i in range(len(nets)):
			print(activ)
			print(nets[i])
			activ = forward(activ, nets[i], 0.5)
			actives_in_one.append(activ)
			print("\033[33m[RES]", activ, "\033[0m")

		active_neurons.append(actives_in_one)

		print("----------------------------------------------")
	
	#-----------------------------backward-----------------------------
	print("-------------------------------[FORWARD DONE]-----------------------------------------")
	for i in range(len(active_neurons)): #for number of training data
		print("[INPUT DATA]--------------------------------------------------", i)
		diff_arr = truths[i] - active_neurons[i][-1]
		# print(diff_arr)
		for j in range(len(nets)): #for layer of net
			idx = len(nets) - j - 1
			# print(nets[idx])
			if j == len(nets) - 1:
				print(diff_arr)
				# print(nets[j])
				# err_weights = np.copy(diff_arr).reshape(-1, 1) * nets[j]
				err_weights = np.copy(diff_arr) * nets[j] 
				print(err_weights)


if __name__ == "__main__":
	main()