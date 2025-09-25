from neural_network import NN, relu, sigmoid, generate_data_1d, generate_data_rand, split_dataset
from data_process import load, preprocess_data
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


def classification1(inputs, truths):
    # print(inputs.shape[1])
    net_shape = (inputs.shape[1], 64, 32, 1)
    activation_funcs = (relu, relu, sigmoid)

    nn = NN(net_shape, activation_funcs, classification=True)

    inputs_train, truths_train, inputs_test, truths_test = split_dataset(inputs, truths)

    nn.train(inputs_train, truths_train, 10000, 0.005, batch_size=20, animation="scatter")
    nn.test(inputs, truths, inputs_test, truths_test)
    nn.save_plots()


def classification2(inputs, truths):
    # print(inputs.shape[1])
    net_shape = (inputs.shape[1], 10, 2, 1)
    activation_funcs = (relu, relu, sigmoid)

    nn = NN(net_shape, activation_funcs, classification=True)

    inputs_train, truths_train, inputs_test, truths_test = split_dataset(inputs, truths)

    nn.train(inputs_train, truths_train, 2500, 0.005, batch_size=len(inputs_train), animation="scatter")
    nn.test(inputs, truths, inputs_test, truths_test)
    nn.save_plots()


def main():
    try:
        argv = sys.argv

        if (len(argv) == 2 and argv[1] == "--help"):
            print_help()
        elif (len(argv) == 2 and argv[1] == "--regre-test"):
            test_regression_noise()
        elif (len(argv) == 3 and argv[1] == "--class-data"):    
            df = load(argv[2])
            inputs, truths = preprocess_data(df)
            classification1(inputs, truths)
        elif (len(argv) == 3 and argv[1] == "--class-img"):    
            df = load(argv[2])
            inputs, truths = preprocess_data(df)
            classification2(inputs, truths)
        else:
            raise ValueError("Wrong arguments. Try: python mlp.py --help")

    except KeyboardInterrupt as e:
        print()
        print("Stopped by user.\033[?25h")

    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()