#!/usr/bin/python3

import json


def wrong_type(conf:dict, field: str, field_type: type, mandatory:bool) -> bool:
    """Check if a filed in configuration file has right type"""
    
    if mandatory and not field in conf:
        raise ValueError("Missing mandatory field.")
    if field in conf and isinstance(conf[field], field_type):
        return False
    print(f"{field} {type(conf[field])}")
    return True


def check_arg_type(conf: dict) -> bool:
    """Check argument type in json configuration file"""

    if wrong_type(conf, "shape", (list,tuple), True):
        return False
    elif wrong_type(conf, "activation_funcs", (list,tuple), True):
        return False
    elif wrong_type(conf, "weights_init", (list,tuple), True):
        return False
    elif wrong_type(conf, "seed", int, True):
        return False
    elif wrong_type(conf, "loss", str, True):
        return False
    elif wrong_type(conf, "max_epoch", int, True):
        return False
    elif wrong_type(conf, "learning_rate", float, True):
        return False
    elif wrong_type(conf, "batch_size", int, True):
        return False
    elif wrong_type(conf, "classification", bool, True):
        return False
    elif wrong_type(conf, "animation", str, True):
        return False
    elif wrong_type(conf, "train_ratio", float, True):
        return False
    elif wrong_type(conf, "threshold", (bool,float), True):
        return False
    elif wrong_type(conf, "index", bool, True):
        return False
    return True


def valid_net_struct(conf: dict) -> bool:
    """Check if the neural network has valid structure."""

    shape = conf["shape"]
    activ_funcs = conf["activation_funcs"]
    loss = conf["loss"]
    init = conf["weights_init"]

    if len(shape) < 4:
        raise ValueError("Minimum layer is 4.")
    if len(shape) != len(activ_funcs) + 1:
        raise ValueError("Mismatched shape and activation funcions.")
    for num in shape:
        if not isinstance(num, int) or num < 1:
            raise ValueError("Wrong layer neuron numbers")
    for act in activ_funcs:
        if act not in ["relu", "sigmoid", "leaky_relu", "softmax", "none"]:
            raise ValueError("Wrong activation funcion")
    if loss not in ["CrossEntropy", "MeanSquareError"]:
        raise ValueError("Wrong loss function")
    if loss == "CrossEntropy" and activ_funcs[-1] not in  ["sigmoid", "softmax"]:
        raise ValueError("CrossEntropy loss only accept sigmoid or softmax as last layer activation function")
    
    for init_method in init:
        if not init_method in ["he", "xavier", "zero"]:
            raise ValueError("Not supported initialization method")
    if len(init) != len(activ_funcs):
        raise ValueError("Wrong initialization method numbers")
    return True


def check_arg_value(conf: dict) -> bool:
    """Check configuration value."""

    if conf["max_epoch"] <= 0:
        raise ValueError("Max epoch must be positive")
    if conf["learning_rate"] <= 0:
        raise ValueError("Learning rate must be positive")
    if conf["batch_size"] <= 0:
        raise ValueError("Batch size must be positive")
    if conf["train_ratio"] <= 0 or conf["train_ratio"] >= 1:
        raise ValueError("Invalid train ratio")
    if conf["seed"] <= 0:
        raise ValueError("Invalid seed.")


def is_valid_config(conf: dict) -> bool:
    """Check if configuration file is correct."""

    if check_arg_type(conf) == False:
        return False
    if valid_net_struct(conf) == False:
        return False
    check_arg_value(conf)
    if conf["animation"] != "none" and conf["animation"] != "scatter" and conf["animation"] != "plot":
        raise ValueError("Wrong animation type")
    return True


def conf_parser(config: str):
    """Parse the configuration file and check errors."""

    try:
        with open(config, mode='r') as conf_data:
            conf = json.load(conf_data)
        
        if not is_valid_config(conf):
            raise ValueError("Wrong config file.")

        print(f"\033[33m-------------------------------------------------------------")
        print(f"|Load config from file {config}")
        for (key, value) in conf.items():
            print("|", key, ":", value)
        print(f"-------------------------------------------------------------\033[0m")
        return conf

    except Exception as e:
        print("Error in configuration:", e)
        return None


# def main():
#     conf = conf_parser(sys.argv[1])
#     if conf is None:
#         sys.exit(1)


# if __name__ == "__main__":
#     main()