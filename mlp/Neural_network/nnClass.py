import numpy as np
# import cupy as np
from .activation_func import *
from .nnUtils import *
import matplotlib.pyplot as plt
from datetime import datetime
import os


class NN:
    """Neural network class, which can perform training and prediction"""

    def __init__(self, shape, activation_functions, classification=False):
        """Init a multi layer perceptron."""
        
        if len(shape) < 4:
            raise ValueError("Net shape to short.")
        if len(shape) != len(activation_functions) + 1:
            raise ValueError("Mismatched net shape and activation functions.")
        
        self.net_shape = shape
        self.activ_funcs = activation_functions

        self.nets = network(self.net_shape)
        self.len_nets = len(self.nets)
        self.len_out = self.net_shape[-1]
        self.biases = create_bias(self.net_shape, 0.1)

        self.Wgrads = [np.zeros_like(w) for w in self.nets] # only for easy batch train
        self.Bgrads = [np.zeros_like(b) for b in self.biases] # only for easy batch train
        self.actives = create_actives(shape, 0.0) # only for easy batch train

        self.graph_loss_train = []
        self.graph_loss_test = []
        self.graph_acc_train = []
        self.graph_acc_test = []
        self.graph_epoch = []

        self.loss_threshold = 0.000000001
        self.loss_test = None
        self.loss_train = None
        self.plt = plt

        self.deriv_map = dict()
        self.deriv_map[relu] = relu_deriv
        self.deriv_map[sigmoid] = sigmoid_deriv

        self.classification = classification


    def check_train_params(self, inputs, truths):
        """Check training parameters."""

        if len(inputs) != len(truths):
            raise ValueError("Mismatched training dataset")


    def train_batch(self, inputs, truths, learning_rate=0.01):
        """Train a batch."""
    
        inputs_batch = np.hstack(inputs)
        truths_batch = np.hstack(truths)
        # -----------------------------forward --------------------------------
        actives = [inputs_batch]
        Bgrads = []
        Wgrads = []

        for i in range(self.len_nets):
            actives.append(forward_layer(self.nets[i], actives[i], self.biases[i], self.activ_funcs[i]))
        # -----------------------------forward end-----------------------------
        # -----------------------------back probab --------------------------------
        local_grad = (actives[-1] - truths_batch) * activ_deriv(self.activ_funcs[-1], actives[-1], self.deriv_map) # last layer difference
        Bgrads.append(np.mean(local_grad, axis=1, keepdims=True))
        Wgrads.append(local_grad @ actives[-2].T / len(inputs))
    
        for i in range(self.len_nets - 1, 0, -1):
            loss_prev_layer = self.nets[i].T @ local_grad  #cal the loss of prev layer
            local_grad = loss_prev_layer * activ_deriv(self.activ_funcs[i - 1], actives[i], self.deriv_map) 
            Bgrads.append(np.mean(local_grad, axis=1, keepdims=True))
            Wgrads.append(local_grad @ actives[i - 1].T / len(inputs))
        # -----------------------------back probab end-----------------------------
        gradient_descent(self.nets, self.biases, Wgrads[::-1], Bgrads[::-1], learning_rate)


    # def train_batch_easy(self, inputs, truths, learning_rate=0.01):  # Only used to explain
    #     """Train a batch, version easier to understand but run slower."""

    #     for i in range(len(self.Wgrads)):
    #         self.Wgrads[i].fill(0)
    #         self.Bgrads[i].fill(0)
    #     for i_data in range(len(inputs)):
    #         # -----------------------------forward --------------------------------
    #         self.actives[0] = inputs[i_data]
    #         for i in range(self.len_nets):
    #             self.actives[i + 1] = forward_layer(self.nets[i], self.actives[i], self.biases[i], self.activ_funcs[i])
    #         # -----------------------------forward end-----------------------------
    #         # -----------------------------back probab --------------------------------
    #         local_grad = (self.actives[-1] - truths[i_data] ) * activ_deriv(self.activ_funcs[-1], self.actives[-1], self.deriv_map) # last layer local_graderence
    #         self.Bgrads[-1] += local_grad
    #         self.Wgrads[-1] += local_grad @ self.actives[-2].T
        
    #         for i in range(self.len_nets - 1, 0, -1): #exclude index == 0
    #             loss_prev_layer = self.nets[i].T @ local_grad  #cal the loss of prev layer
    #             local_grad = loss_prev_layer * activ_deriv(self.activ_funcs[i - 1], self.actives[i], self.deriv_map) 
    #             self.Bgrads[i - 1] += local_grad
    #             self.Wgrads[i - 1] += local_grad @ self.actives[i - 1].T
    #         # -----------------------------back probab end-----------------------------
    #     num_data = len(inputs)
    #     for i in range(self.len_nets):
    #         self.Wgrads[i] /= num_data
    #         self.Bgrads[i] /= num_data
    #     gradient_descent(self.nets, self.biases, self.Wgrads, self.Bgrads, learning_rate)
    

    def inference(self, inputs):
        """After training, use weights to do inference."""

        result = []
        for input in inputs:
            activ = input
            for i in range(self.len_nets):
                activ = forward_layer(self.nets[i], activ, self.biases[i], self.activ_funcs[i])
            result.append(activ)
        return result


    def train(self, inputs, truths, max_iter=10000, learning_rate=0.01, batch_size=20, visualize=True, test_ratio = 0.8, threshold=None, animation=None):
        """Train a dataset."""

        self.check_train_params(inputs, truths)
        self.prepare(visualize, threshold)

        inputs_train, truths_train, inputs_test, truths_test = split_dataset(inputs, truths, test_ratio)
        startTime = datetime.now()
        for epoch in range(max_iter):
            inputs_train, truths_train = shuffle_data(inputs_train, truths_train)
            # mini_batch_training
            count = 0
            while count < len(inputs_train):
                inputs_batch = inputs_train[count: count + batch_size]
                truths_batch = truths_train[count: count + batch_size]
                self.train_batch(inputs_batch, truths_batch, learning_rate)
                count += batch_size
            # mini_batch_training
            stop = self.show_record(epoch, inputs_train, inputs_test, truths_train, truths_test, startTime, animation)
            if stop == True:
                break
        print()
        print("[TRAINING DONE]")
        self.plt.ioff()
        self.plt.show()
        self.plt.close()


    def test(self, inputs, truths, test_inputs, test_truths):
        """Test for a new dataset."""
        test_result = self.inference(test_inputs)
        plt.scatter(inputs[:, 0], np.array(truths)[:, 0], c="blue", label="Prediction", s=0.5)
        plt.scatter(test_inputs[:, 0], np.array(test_result)[:, 0], c="red", label="Prediction", s=0.5)
        plt.legend(loc="lower left")
        plt.savefig("visualize/prediction.png", dpi=300, bbox_inches='tight')
        plt.close()


    def test_animation(self, test_inputs, test_truths, animation):
        """Test for a new dataset."""

        test_result = self.inference(test_inputs)
        self.plt.clf()
        inputs = test_inputs[:, 0].flatten()
        truths = np.array(test_truths)[:, 0].flatten()
        outputs = np.array(test_result)[:, 0].flatten()
        
        sorted_index = np.argsort(inputs)
        inputs_sorted = inputs[sorted_index]
        truths_sorted = truths[sorted_index]
        outputs_sorted = outputs[sorted_index]

        self.plt.scatter(inputs_sorted, truths_sorted, c="blue", label="Truth", s=10)
        if animation == "plot":
            self.plt.plot(inputs_sorted, outputs_sorted, c="red", label="Prediction", lw=1)
        elif animation == "scatter":
            self.plt.scatter(inputs_sorted, outputs_sorted, c="red", label="Prediction", s=10)
        elif animation is None:
            pass
        else:
            raise TypeError("Wrong animation type")
        self.plt.legend(loc="lower left")
        self.plt.pause(0.1)
        

    def show_record(self, epoch, inputs_train, inputs_test, truths_train, truths_test, startTime, animation): #return a boolean to determine if training continue
        """Show and record the loss"""

        if epoch % 50 == 0:
            predicts_train = self.inference(inputs_train)
            predicts_test = self.inference(inputs_test)
            loss_train = loss(mse_loss, truths_train, predicts_train)
            loss_test = loss(mse_loss, truths_test, predicts_test)

            # if classification == True:
            predicts_train = np.array(predicts_train, dtype=np.float32)
            predicts_test = np.array(predicts_test, dtype=np.float32)
            predict_train_class = (predicts_train >= 0.5).astype(np.int32)
            predict_test_class = (predicts_test >= 0.5).astype(np.int32)
            acc_train = accuracy_1d(truths_train, predict_train_class)
            acc_test = accuracy_1d(truths_test, predict_test_class)


            if loss_train < 1 and loss_test < 1:
                self.graph_loss_train.append(loss_train)
                self.graph_loss_test.append(loss_test)
            self.graph_epoch.append(epoch)

            if self.classification == True:
                self.graph_acc_train.append(acc_train)
                self.graph_acc_test.append(acc_test)

            if animation is not None and epoch % 50 == 0:
                self.test_animation(inputs_test[:50], truths_test[:50], animation)
            if epoch % 100 == 0:
                time = str(datetime.now() - startTime).split(".")[0]
                if self.classification == False:
                    print(f"\033[?25l[EPOCH] {epoch}  [LOSS_TRAIN] {loss_train:8f} [LOSS_TEST] {loss_test:8f}  [TIME] {time}\033[?25h", end="\r")
                else:
                    print(f"\033[?25l[EPOCH] {epoch}  [LOSS_TRAIN] {loss_train:8f} [LOSS_TEST] {loss_test:8f} [ACC_TRAIN] {acc_train:5f} [ACC_TEST] {acc_test:5f} [TIME] {time}\033[?25h", end="\r")

            if self.loss_train  is not None and self.loss_test is not None:
                if abs(self.loss_train - loss_train) < self.loss_threshold and abs(self.loss_test - loss_test) < self.loss_threshold:
                    return True
                    
            self.loss_train = loss_train
            self.loss_test = loss_test

        return False


    def save_plots(self):
        """Show loss func."""

        plt.plot(self.graph_epoch, self.graph_loss_train, c="cyan", lw=1, label="Training loss")
        plt.plot(self.graph_epoch, self.graph_loss_test, c="orange", linestyle="--", lw=1, label="Test loss")
        plt.grid(True, linestyle="--", linewidth=0.7, alpha=0.7)
        plt.title("Loss Curves")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")
        plt.savefig("visualize/loss.png", dpi=300, bbox_inches='tight')
        plt.close()

        if self.classification == True:
            plt.plot(self.graph_epoch, self.graph_acc_train, c="cyan", lw=1, label="Training accuracy")
            plt.plot(self.graph_epoch, self.graph_acc_test, c="orange", linestyle="--", lw=1, label="Test accuracy")
            plt.grid(True, linestyle="--", linewidth=0.7, alpha=0.7)
            plt.title("Learning Curves")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.legend(loc="lower right")
            plt.savefig("visualize/accuracy.png", dpi=300, bbox_inches='tight')
            plt.close()
            print("Saved")


    def prepare(self, visualize, threshold):
        """Create plt scatter."""

        if visualize == True:
            os.makedirs("visualize", exist_ok=True)
            plt.ion()
        if threshold is not None:
            self.loss_threshold = threshold