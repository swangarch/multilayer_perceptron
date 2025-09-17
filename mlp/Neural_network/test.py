from utils.activation_func import relu, sigmoid
from utils.data_generation import generate_data_1d, generate_data_3d
from utils.nnClass import NN


def data():
    inputs, truths = generate_data_3d(142, 50)
    print(inputs.shape)
    print(truths.shape)


def test0():
    net_shape = (1, 64, 32, 1)
    activation_funcs = (relu, relu, None)

    nn = NN(net_shape, activation_funcs)

    inputs, truths = generate_data_1d(142, 200, 5)
    test_inputs, test_truths = generate_data_1d(123, 50)

    # nn.train(inputs, truths, 40000, 0.01, batch_size=20, animation="plot")
    nn.train(inputs, truths, 40000, 0.01, batch_size=20)
    nn.test(inputs, truths, test_inputs, test_truths)
    nn.save_plots()


def test1():
    net_shape = (1, 64, 64, 1)
    activation_funcs = (relu, relu, None)

    nn = NN(net_shape, activation_funcs)

    inputs, truths = generate_data_1d(142, 500, 5)
    test_inputs, test_truths = generate_data_1d(123, 50)

    nn.train(inputs, truths, 20000, 0.005, batch_size=50)
    nn.test(inputs, truths, test_inputs, test_truths)
    nn.save_plots()


def test2():
    net_shape = (3, 64, 64, 2)
    activation_funcs = (relu, relu, None)

    nn = NN(net_shape, activation_funcs)

    inputs, truths = generate_data_3d(142, 500)
    test_inputs, test_truths = generate_data_3d(123, 80)

    # nn.train(inputs, truths, 20000, 0.005, animation="scatter")
    nn.train(inputs, truths, 20000, 0.005)
    nn.test(inputs, truths, test_inputs, test_truths)
    nn.save_plots()


def main():
    try:
        # data()
        test0()

    except KeyboardInterrupt as e:
        print()
        print("Stopped by user.\033[?25h")

    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()