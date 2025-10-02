import numpy as np
from .activation_func import *
from .nnUtils import *
import matplotlib.pyplot as plt
from datetime import datetime
import os
import json
import sys


class NN:
    """Neural network class, which can perform training and prediction"""

    def __init__(self, shape, activation_functions, classification=False, loss="MeanSquareError"):
        """Init a multi layer perceptron."""
        
        if len(shape) < 4:
            raise ValueError("Net shape too short.")
        if len(shape) != len(activation_functions) + 1:
            raise ValueError("Mismatched net shape and activation functions.")
        
        self.net_shape = shape
        print("[NET] ", self.net_shape)
        self.activ_funcs = activation_functions

        self.nets = network(self.net_shape)
        self.len_nets = len(self.nets)
        self.len_out = self.net_shape[-1]
        # self.biases = create_bias(self.net_shape, 0.1)
        self.biases = create_bias(self.net_shape, 0)
        self.loss_func = loss
        # print(self.loss_func)

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

        inputs_batch = inputs.T   # (batch_size, features) -> (features, batch_size)
        truths_batch = truths.T

        # -----------------------------forward --------------------------------
        actives = [inputs_batch]
        Bgrads = []
        Wgrads = []

        for i in range(self.len_nets):
            actives.append(forward_layer(self.nets[i], actives[i], self.biases[i], self.activ_funcs[i]))
        # -----------------------------forward end-----------------------------
        # -----------------------------back probab --------------------------------
        # Last layer
        if self.loss_func == "CrossEntropy" and self.activ_funcs[-1] in [sigmoid, softmax]:
            local_grad = actives[-1] - truths_batch
        else:
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


    def inference(self, inputs):
        """After training, use weights to do inference."""

        activ = inputs.T
        for i in range(self.len_nets):
            activ = forward_layer(self.nets[i], activ, self.biases[i], self.activ_funcs[i])
        return activ.T
    

    def convert_to_onehot(self, array):  # array => (batchsize, category)
        # print("a before", array.shape)
        batch_size = array.shape[0]
        num_classes = self.net_shape[-1]
        onehot = np.zeros((batch_size, num_classes))
        onehotindex = array[:,0].astype(int)
        onehot[np.arange(batch_size), onehotindex] = 1
        array = onehot
        # print("a after", array.shape)
        return array


    def train(self, inputs, truths, max_iter=10000, learning_rate=0.01, batch_size=50, visualize=True, test_ratio = 0.8, threshold=None, animation=None):
        """Train a dataset."""

        self.check_train_params(inputs, truths)
        self.prepare(visualize, threshold)

        inputs_train, truths_train, inputs_test, truths_test = split_dataset(inputs, truths, test_ratio)

        if self.loss_func == "CrossEntropy" and self.activ_funcs[-1] == softmax:
            truths_test = self.convert_to_onehot(truths_test)
            truths_train = self.convert_to_onehot(truths_train)

        startTime = datetime.now()
        try:
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
            print("[TRAINING DONE]")
            self.plt.ioff()
            self.plt.show()
            self.plt.close()

        except KeyboardInterrupt:
            print("Stopped by user.\033[?25h")
        self.save_weights()


    def load_weights(self, file):
        if file is None:
            return
    
        with open(file, mode="r") as f:
            params = json.load(f)    
        ws = [np.array(w) for w in params["weights"]]
        bs = [np.array(b) for b in params["biases"]]
        for i in range(len(self.nets)):
            if self.nets[i].shape != ws[i].shape or self.biases[i].shape != bs[i].shape:
                print("[Load params from file failed, mismatched]")
                return
        self.nets = ws
        self.biases = bs


    def save_weights(self):
        weights_li = [ arr.tolist() for arr in self.nets ]
        biases_li = [ arr.tolist() for arr in self.biases ]
        model_params = {
            "weights": weights_li,
            "biases": biases_li,
        }
        with open("params.json", "w", encoding="utf-8") as f:
            json.dump(model_params, f, indent=4)
        print("[Params saved => (params.json)]\033[?25h")


    def test(self, test_inputs, test_truths):
        """Test for a new dataset."""
        test_result = self.inference(test_inputs)
        plt.scatter(test_inputs[:, 0], np.array(test_truths)[:, 0], c="blue", label="Test truth", s=0.5)
        plt.scatter(test_inputs[:, 0], np.array(test_result)[:, 0], c="red", label="Test prediction", s=0.5)
        plt.legend(loc="lower left")
        plt.savefig("visualize/prediction.png", dpi=300, bbox_inches='tight')
        plt.close()
        if self.classification == True:
            if self.loss_func == "CrossEntropy" and self.activ_funcs[-1] == softmax:
                test_result = np.argmax(test_result, axis=1, keepdims=True)
            elif self.loss_func == "CrossEntropy" and self.activ_funcs[-1] == sigmoid:
                test_result = [ (arr > 0.5).astype(int) for arr in test_result ]
            count = 0
            l = len(test_result)
            for i in range(len(test_result)):
                if test_result[i] == test_truths[i]:
                    count += 1
            print(f"[Test Acc] {(count / l) * 100:.2f}%")
        with open("predictions.json", "w", encoding="utf-8") as f:
            json.dump({"prediction": [r.tolist() for r in test_result]}, f, indent=4)
        print("[Predictions saved => (predictions.json)]\033[?25h")


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


    def cal_loss(self, truths_train, predicts_train, truths_test, predicts_test):
        "Use raw value to calculate loss, no need to convert to category"

        if self.loss_func == "CrossEntropy":
            loss_train = loss(ce_loss, truths_train, predicts_train)
            loss_test = loss(ce_loss, truths_test, predicts_test)
        elif self.loss_func == "MeanSquareError":    
            loss_train = loss(mse_loss, truths_train, predicts_train)
            loss_test = loss(mse_loss, truths_test, predicts_test)
        else:
            raise ValueError("Not supported loss function")
        return loss_train, loss_test

    
    def get_category_by_predict(self, predicts_train, predicts_test, truths_train, truths_test):
        
        # print(predicts_train.shape)
        if self.loss_func == "CrossEntropy" and self.activ_funcs[-1] == softmax:
            predict_train_cat = predicts_train.argmax(axis=1, keepdims=True)
            predict_test_cat = predicts_test.argmax(axis=1, keepdims=True)
            onehot = True
        elif self.loss_func == "CrossEntropy" and self.activ_funcs[-1] == sigmoid:
            predict_train_cat = (predicts_train >= 0.5).astype(np.int32)
            predict_test_cat = (predicts_test >= 0.5).astype(np.int32)
            onehot = False
        else:
            return None, None
        
        # print("------------------------")
        # print(predict_train_cat.shape)
        # print(predict_test_cat.shape)

        truths_train_original = np.argmax(truths_train, axis=1, keepdims=True)
        truths_test_original = np.argmax(truths_test, axis=1, keepdims=True)

        # print(truths_train.shape)
        # print(truths_test.shape)
        # print("------------------------")

        acc_train = accuracy_1d(truths_train_original, predict_train_cat, onehot)
        acc_test = accuracy_1d(truths_test_original, predict_test_cat, onehot)
        return acc_train, acc_test


    def collect_train_record(self, epoch, loss_train, loss_test, acc_train, acc_test):
        if loss_train < 1 and loss_test < 1:
            self.graph_loss_train.append(loss_train)
            self.graph_loss_test.append(loss_test)
        self.graph_epoch.append(epoch)
        if self.classification == True and acc_test is not None and acc_train is not None:
            self.graph_acc_train.append(acc_train)
            self.graph_acc_test.append(acc_test)


    def show_train_info(self, epoch, startTime, loss_train, loss_test, acc_train, acc_test):
        if epoch % 100 == 0:
            time = str(datetime.now() - startTime).split(".")[0]
            if self.classification == False:
                print(f"\033[?25l[EPOCH] {epoch}  [Loss_Train] {loss_train:.4f} [Loss_Val] {loss_test:.4f}  [TIME] {time}\033[?25h")
            else:
                print(f"\033[?25l[EPOCH] {epoch}  [Loss_Train] {loss_train:.4f} [Loss_Val] {loss_test:.4f} [Acc_Train] {(acc_train * 100):.1f}% [Acc_Val] {(acc_test * 100):.1f}% [TIME] {time}\033[?25h")


    def show_record(self, epoch, inputs_train, inputs_test, truths_train, truths_test, startTime, animation): #return a boolean to determine if training continue
        """Show and record the loss"""

        if epoch % 50 == 0:
            predicts_train = self.inference(inputs_train)
            predicts_test = self.inference(inputs_test)
            

            loss_train, loss_test = self.cal_loss(truths_train, predicts_train, truths_test, predicts_test)
            acc_train, acc_test = None, None

            if self.classification == True:
                acc_train, acc_test = self.get_category_by_predict(predicts_train, predicts_test, truths_train, truths_test)

            self.collect_train_record(epoch, loss_train, loss_test, acc_train, acc_test)

            if animation != "none" and epoch % 50 == 0:
                self.test_animation(inputs_test[:50], truths_test[:50], animation)
            
            self.show_train_info(epoch, startTime, loss_train, loss_test, acc_train, acc_test)

            # Early stop --------------------------------------
            if self.loss_train  is not None and self.loss_test is not None:
                if abs(self.loss_train - loss_train) < self.loss_threshold and abs(self.loss_test - loss_test) < self.loss_threshold:
                    return True
            # Early stop --------------------------------------
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


    def prepare(self, visualize, threshold):
        """Create plt scatter."""

        if visualize == True:
            os.makedirs("visualize", exist_ok=True)
            plt.ion()
        if threshold is not None:
            self.loss_threshold = threshold