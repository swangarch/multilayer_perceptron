import json
import sys


def wrong_type(conf, field, field_type, mandatory:bool):
    if mandatory and not field in conf:
        raise ValueError("Missing mandatory field.")
    if field in conf and isinstance(conf[field], field_type):
        return False
    print(f"{field} {type(conf[field])}")
    return True


def check_arg_type(conf: dict) -> bool:
    if wrong_type(conf, "shape", (list,tuple), True):
        return False
    elif wrong_type(conf, "activation_funcs", (list,tuple), True):
        return False
    if wrong_type(conf, "max_epoch", int, True):
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
    shape = conf["shape"]
    activ_funcs = conf["activation_funcs"]

    if len(shape) < 4:
        raise ValueError("Minimum layer is 4.")
    if len(shape) != len(activ_funcs) + 1:
        raise ValueError("Mismatched shape and activation funcions.")
    for num in shape:
        if not isinstance(num, int) or num < 1:
            raise ValueError("Wrong layer neuron numbers")
    for act in activ_funcs:
        if act not in ["relu", "sigmoid", "gelu", "softmax", "none"]:
            raise ValueError("Wrong activation funcion")
    return True


def is_valid_config(conf: dict) -> bool:
    if check_arg_type(conf) == False:
        return False
    if valid_net_struct(conf) == False:
        return False
    if conf["animation"] != "none" and conf["animation"] != "scatter" and conf["animation"] != "plot":
        raise ValueError("Wrong animation type")
    return True


def conf_parser(config: str):
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


def main():
    conf = conf_parser(sys.argv[1])


if __name__ == "__main__":
    main()