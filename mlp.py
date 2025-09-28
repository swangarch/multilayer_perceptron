from neural_network import NN, generate_data_rand, split_dataset, get_activation_funcs_by_name
from data_process import load, preprocess_data, conf_parser
import sys


def print_help():
    print("-------------------------------------------------------------")
    print("Multilayer perceptron()")
    print("Usage:")
    print("   python  mlp.py  <--options>  <data(optional)>")
    print("Options:")
    print("  --class-data:  csv_data need to be provided.")
    print("  --class-img:  csv_data need to be provided, classification of image.")
    print("  --regre-test:  no csv_data needed, a random generated data will beused.")
    print("  --help:  Show help messages.")
    print("  --More features to come.")
    print("-------------------------------------------------------------")


def training(conf, inputs, truths):

    net_shape = conf["shape"]
    activation_funcs = get_activation_funcs_by_name(conf["activation_funcs"])
    nn = NN(net_shape, activation_funcs, classification=conf["classification"])
    inputs_train, truths_train, inputs_test, truths_test = split_dataset(inputs, truths)
    nn.train(inputs_train, truths_train, 
             conf["max_epoch"], 
             conf["learning_rate"], 
             batch_size=conf["batch_size"], 
             test_ratio=conf["train_ratio"],
             threshold=conf["threshold"],
             animation=conf["animation"])

    nn.test(inputs, truths, inputs_test, truths_test)
    nn.save_plots()


def main():
    try:
        argv = sys.argv

        if len(argv) == 1 or ((len(argv) == 2 and argv[1] == "--help")):
            print_help()
        elif (len(argv) == 3 and argv[2] == "--gen-data1d"):
            conf = conf_parser(argv[1])
            if conf is None:
                sys.exit(1)
            if argv[2] == "--gen-data1d":
                inputs, truths = generate_data_rand(142, 500, 0.02)
                training(conf, inputs, truths)
            else:
                df = load(argv[2], conf["index"])
                inputs, truths = preprocess_data(df)
                training(conf, inputs, truths)
        else:
            raise ValueError("Wrong arguments. Try: python mlp.py --help")

    except KeyboardInterrupt as e:
        print()
        print("Stopped by user.\033[?25h")

    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()