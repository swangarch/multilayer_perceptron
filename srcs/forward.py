import math
import numpy as np
from numpy import ndarray as array


def relu(value: float):
	return value if value >= 0 else 0


def sigmoid(value: array):
	return 1.0 / (1.0 + math.e ** -value)


def forward(arr1:array, arr2:array, bias:float):

	# print("[ARR1]\n", arr1)
	# print("[ARR2]\n", arr2)

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


# def main():
# 	#-----------------------------init-----------------------------
# 	inputs = [np.array([0.4, 0.3, 0.8]), np.array([0.45, 0.21, 0.66]), np.array([0.95, 0.19, 0.16])]
# 	truths = [np.array([0.28, 0.72]), np.array([0.58, 0.62]), np.array([0.18, 0.42])]
# 	# nets = network((3, 5, 5, 5, 2))
# 	nets = network((3, 4, 4, 2))

# 	#-----------------------------forward-----------------------------
# 	active_neurons = []
# 	for input in inputs:
# 		activ = input
# 		actives_in_one = [activ]
# 		for i in range(len(nets)):
# 			print(activ)
# 			print(nets[i])
# 			activ = forward(activ, nets[i], 0.5)
# 			actives_in_one.append(activ)
# 			print("\033[33m[RES]", activ, "\033[0m")

# 		active_neurons.append(actives_in_one)

# 		print("----------------------------------------------")
	
# 	#-----------------------------backward-----------------------------
# 	print("-------------------------------[FORWARD DONE]-----------------------------------------")

# 	i_data = 0
# 	len_weight = len(nets)
# 	for i in len_weight:
# 		idx = len_weight - i - 1
# 		diff_arr = truths[i_data] - active_neurons[i_data][idx + 1]
# 		trans_net = nets[idx].copy()
# 		np.transpose(trans_net)
# 		loss_prev_layer = diff_arr @ trans_net
# 		sigmoid_derivative = 


def main():
	#-----------------------------init-----------------------------
	inputs = np.array([0.4, 0.3, 0.8])
	truths = np.array([0.28, 0.72])
	nets = network((3, 4, 4, 2))


	for epoch in range(1000):
		activ = inputs
		actives = [activ]
		for i in range(len(nets)):
			activ = forward(activ, nets[i], 0.5)
			actives.append(activ)
		
		# -----------------------------forward end-----------------------------
		
		grads = []
		len_weight = len(nets)
		len_out = len(truths)
		diff_arr = (truths - actives[-1])
		print("[DIFF ARR]:", diff_arr, "[LAST ACTIVATION]", actives[-1])
		Tdiff_arr = diff_arr.copy().reshape((len_out, 1))
		Tactive = actives[-2].copy()
		Tactive = np.transpose(Tactive).reshape((1, len(Tactive)))
		grad = Tdiff_arr @ Tactive
		grads.append(grad)
		# print(grad)
		for i in range(len_weight):
			idx = len_weight - i - 1
			if idx < 1:
				break
			trans_net = nets[idx].copy()
			trans_net = np.transpose(trans_net)
			loss_prev_layer = diff_arr @ trans_net
			activ_func_derivative = actives[idx] * (1 - actives[idx])
			
			diff_arr = loss_prev_layer * activ_func_derivative
			Tdiff_arr = diff_arr.copy().reshape((len(diff_arr), 1))
			Tactive = actives[idx - 1].copy()
			Tactive = np.transpose(Tactive).reshape((1, len(Tactive)))
			grad = Tdiff_arr @ Tactive
			grads.append(grad)
		# -----------------------------back probab end-----------------------------

		for i, net in enumerate(nets):
			nets[i] = nets[i] - np.transpose(grads[len_weight - i - 1])
		# -----------------------------gradient descent end-----------------------------

		




if __name__ == "__main__":
	main()