from Neural_network import NN, relu, sigmoid, generate_data_1d, generate_data_3d
from load_csv import load
import sys
import numpy as np


def test0():
    net_shape = (1, 64, 64, 32, 1)
    activation_funcs = (relu, relu, relu, sigmoid)

    nn = NN(net_shape, activation_funcs)

    inputs, truths = generate_data_1d(142, 200, 5)
    test_inputs, test_truths = generate_data_1d(123, 50)

    nn.train(inputs, truths, 40000, 0.01, batch_size=20, animation="plot")
    nn.test(inputs, truths, test_inputs, test_truths)
    nn.show_loss()


def test1(inputs, truths):
    net_shape = (30, 64, 64, 32, 1)
    activation_funcs = (relu, relu, relu, sigmoid)

    nn = NN(net_shape, activation_funcs)

    nn.train(inputs, truths, 40000, 0.01, batch_size=20, animation="scatter")
    nn.show_loss()


def convert_to_float(type_name:str):
    if type_name == "M":
        return 1.0
    elif type_name == "B":
        return 0.0


def preprocess_data(df):
    df.iloc[:, 0] = df.iloc[:, 0].apply(convert_to_float)

    data = np.array(df).astype(np.float32)

    # truths 和 inputs
    truths = data[:, 0]           # (N,)
    inputs = data[:, 1:]          # (N,M)

    # 每列归一化
    X_min = inputs.min(axis=0)
    X_max = inputs.max(axis=0)
    inputs_norm = (inputs - X_min) / (X_max - X_min + 1e-8)

    # 三维化
    truths = truths[:, np.newaxis, np.newaxis]       # (N,1,1)
    inputs = inputs_norm[:, :, np.newaxis]          # (N,M,1)

    print(truths.shape)   # (N,1,1)
    print(inputs.shape)   # (N,M,1)
    return inputs, truths


def main():
    try:
        argv = sys.argv
        if len(argv) != 2:
            raise ValueError("Wrong argument number")

        df = load(argv[1])

        inputs, truths = preprocess_data(df)
        test1(inputs, truths)

    except KeyboardInterrupt as e:
        print()
        print("Stopped by user.\033[?25h")

    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()