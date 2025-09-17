from Neural_network import NN, relu, sigmoid, generate_data_1d, generate_data_rand, split_dataset
from Data_process import load, preprocess_data
import sys


def test_regression_noise():
    net_shape = (1, 64, 32, 1)
    activation_funcs = (relu, relu, None)

    nn = NN(net_shape, activation_funcs)

    inputs, truths = generate_data_rand(142, 500, 0.02)
    test_inputs, test_truths = generate_data_rand(123, 50, 0.02)

    # nn.train(inputs, truths, 10000, 0.005, batch_size=20, animation="plot")
    nn.train(inputs, truths, 10000, 0.005, batch_size=20)
    nn.test(inputs, truths, test_inputs, test_truths)
    nn.save_plots()


def test_regression():
    net_shape = (1, 64, 32, 1)
    activation_funcs = (relu, relu, None)

    nn = NN(net_shape, activation_funcs)

    inputs, truths = generate_data_1d(142, 1000, 5)
    test_inputs, test_truths = generate_data_1d(123, 50, 5)

    # nn.train(inputs, truths, 20000, 0.005, batch_size=20, animation="plot")
    nn.train(inputs, truths, 20000, 0.005, batch_size=20)
    nn.test(inputs, truths, test_inputs, test_truths)
    nn.save_plots()


def classification(inputs, truths):
    net_shape = (30, 64, 32, 1)
    activation_funcs = (relu, relu, sigmoid)

    nn = NN(net_shape, activation_funcs, classification=True)

    inputs_train, truths_train, inputs_test, truths_test = split_dataset(inputs, truths)

    # nn.train(inputs_train, truths_train, 10000, 0.001, batch_size=20, animation="scatter")
    nn.train(inputs_train, truths_train, 20000, 0.001, batch_size=20)
    nn.test(inputs, truths, inputs_test, truths_test)
    nn.save_plots()


def main():
    try:
        argv = sys.argv
        if len(argv) != 2:
            raise ValueError("Wrong argument number")

        df = load(argv[1])

        inputs, truths = preprocess_data(df)

        classification(inputs, truths)
        # test_regression_noise()

    except KeyboardInterrupt as e:
        print()
        print("Stopped by user.\033[?25h")

    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()