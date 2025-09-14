import numpy as np
from utils.activation_func import relu, relu_deriv, sigmoid, sigmoid_deriv, activ_deriv
from utils.nnUtils import forward_layer, network, loss, mse_loss, create_bias, gradient_descent, mean_gradients
import matplotlib.pyplot as plt
from datetime import datetime
import os


class NN:
    """Neural network class, which can perform training and prediction"""

    def __init__(self, shape, activation_functions):
        """Init a multi layer perceptron."""
        
        if len(shape) < 4:
            raise ValueError("Net shape to short.")
        if len(shape) != len(activation_functions) + 1:
            raise ValueError("Mismatched net shape and activation functions.")
        
        self.net_shape = shape  ##
        self.activ_funcs = activation_functions  ##

        self.nets = network(self.net_shape)
        self.len_nets = len(self.nets)
        self.len_out = self.net_shape[-1]
        self.biases = create_bias(self.net_shape, 0.1) 

        self.graph_loss = []
        self.graph_epoch = []
        self.grads_weights = []
        self.grads_biases = []
        self.plt = plt

        self.deriv_func_map = dict()
        self.deriv_func_map[relu] = relu_deriv
        self.deriv_func_map[sigmoid] = sigmoid_deriv


    def check_train_params(self, inputs, truths):
        """Check training parameters."""

        if len(inputs) != len(truths):
            raise ValueError("Mismatched training dataset")


    def train(self, inputs, truths, max_iter=10000, learning_rate=0.01, visualize=True):
        """Train a dataset."""

        self.check_train_params(inputs, truths)
        if visualize == True:
            os.makedirs("visualize", exist_ok=True)
            self.plt.scatter(inputs[:, 0], truths[:, 0], c="blue", label="Truth", s=0.5)

        num_data = len(inputs)
        startTime = datetime.now()

        for epoch in range(max_iter):
            self.grads_weights = []
            self.grads_biases = []
            for i_data, input in enumerate(inputs):
                # -----------------------------forward --------------------------------
                activ = input
                actives = [input]
                for i in range(self.len_nets):
                    activ = forward_layer(activ, self.nets[i], self.biases[i], self.activ_funcs[i])
                    actives.append(activ)
                # -----------------------------forward end-----------------------------

                # -----------------------------back probab --------------------------------
                Wgrads = []  #collect the output gradient matrixs
                Bgrads = []
                diff = (actives[-1] - truths[i_data] ) * activ_deriv(self.activ_funcs[-1], actives[-1], self.deriv_func_map) # last layer difference
                Bgrads.append(diff) #gradient for bias
                Wgrads.append(diff.reshape(-1, 1) @ actives[-2].reshape(1, -1))  # add gradient for weights to grads

                for i in range(self.len_nets - 1, 0, -1): #exclude index == 0
                    loss_prev_layer = diff @ self.nets[i].T  #cal the loss of prev layer
                    diff = loss_prev_layer * activ_deriv(self.activ_funcs[i - 1], actives[i], self.deriv_func_map)
                    Bgrads.append(diff) #gradient for bias
                    Wgrads.append(diff.reshape(-1, 1) @ actives[i - 1].reshape(1, -1))  # add gradient for weights to grads 
                # -----------------------------back probab end-----------------------------
                self.grads_weights.append(Wgrads)
                self.grads_biases.append(Bgrads)

            # -----------------------------show -----------------------------
            if epoch % 10 == 0 and loss(mse_loss, truths[i_data], actives[-1]) < 1:
                self.graph_loss.append(loss(mse_loss, truths[i_data], actives[-1]))
                self.graph_epoch.append(epoch)
            if epoch % 100 == 0 and loss(mse_loss, truths[i_data], actives[-1]) < 1:
                print(f"[EPOCH] {epoch}  [MSE] {loss(mse_loss, truths[i_data], actives[-1])}  [TIME] {datetime.now() - startTime}", end="\r")
            # -----------------------------show end--------------------------
            Wgrads_mean, Bgrads_mean = mean_gradients(self.net_shape, self.grads_weights, self.grads_biases, num_data)
            gradient_descent(self.nets, self.biases, Wgrads_mean, Bgrads_mean, learning_rate)

        print()
        print("[TRAINING DONE]")


    def test(self, test_inputs, test_truths):
        """Test for a new dataset."""
        test_result =[]
        for test_input in test_inputs:
            res = test_input
            for i in range(len(self.nets)):
                res = forward_layer(res, self.nets[i], self.biases[i], self.activ_funcs[i])
                if i == len(self.nets) - 1:
                    test_result.append(list(res))

        self.plt.scatter(test_inputs[:, 0], np.array(test_result)[:, 0], c="red", label="Prediction", s=0.5)
        self.plt.legend(loc="lower left")
        self.plt.savefig("visualize/prediction.png", dpi=300, bbox_inches='tight')
        self.plt.close()
        # -----------------------------test end--------------------------


    def show_loss(self):
        """Show loss func."""

        plt.plot(self.graph_epoch, self.graph_loss, c="red")
        plt.savefig("visualize/loss.png", dpi=300, bbox_inches='tight')
        plt.close()
