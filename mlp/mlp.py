from Neural_network import NN, relu, sigmoid, generate_data_1d, generate_data_rand, split_dataset
from Data_process import load, preprocess_data
import sys


def print_help():
    print("-------------------------------------------------------------")
    print("Multilayer perceptron()")
    print("Usage:")
    print("   python  mlp.py  <--options>  <data(optional)>")
    print("Options:")
    print("  --classification-data:  csv_data need to be provided.")
    print("  --regression-test:  no csv_data needed, a random generated data will beused.")
    print("  --help:  Show help messages.")
    print("  --More features to come.")
    print("-------------------------------------------------------------")


def test_regression_noise():
    net_shape = (1, 64, 32, 1)
    activation_funcs = (relu, relu, None)

    nn = NN(net_shape, activation_funcs)

    inputs, truths = generate_data_rand(142, 500, 0.02)
    test_inputs, test_truths = generate_data_rand(123, 50, 0.02)

    nn.train(inputs, truths, 10000, 0.005, batch_size=20, animation="plot")
    nn.test(inputs, truths, test_inputs, test_truths)
    nn.save_plots()


def classification(inputs, truths):
    net_shape = (30, 64, 32, 1)
    activation_funcs = (relu, relu, sigmoid)

    nn = NN(net_shape, activation_funcs, classification=True)

    inputs_train, truths_train, inputs_test, truths_test = split_dataset(inputs, truths)

    nn.train(inputs_train, truths_train, 10000, 0.001, batch_size=20, animation="scatter")
    nn.test(inputs, truths, inputs_test, truths_test)
    nn.save_plots()


def main():
    try:
        argv = sys.argv

        if (len(argv) == 2 and argv[1] == "--help"):
            print_help()
        elif (len(argv) == 2 and argv[1] == "--regression-test"):
            test_regression_noise()
        elif (len(argv) == 3 and argv[1] == "--classification-data"):    
            df = load(argv[2])
            inputs, truths = preprocess_data(df)
            classification(inputs, truths)
        else:
            raise ValueError("Wrong arguments. Try: python mlp.py --help")

    except KeyboardInterrupt as e:
        print()
        print("Stopped by user.\033[?25h")

    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()