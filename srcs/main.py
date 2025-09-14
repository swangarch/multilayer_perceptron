from utils.activation_func import relu, sigmoid
from utils.data_generation import generate_data_1d
from utils.nnClass import NN


def main():
    try:
        net_shape = (1, 64, 32, 1)
        activation_funcs = (relu, relu, None)

        nn = NN(net_shape, activation_funcs)

        inputs, truths = generate_data_1d(142, 100)
        nn.train(inputs, truths, 20000, 0.05)

        test_inputs, test_truths = generate_data_1d(123, 80)
        nn.test(test_inputs, test_truths)
        nn.show_loss()

    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()