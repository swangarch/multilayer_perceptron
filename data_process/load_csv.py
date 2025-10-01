#!/usr/bin/python3

import pandas as pd
import numpy as np
import os


def convert_to_float(type_name:str):
    if type_name == "M" or type_name=="1" or type_name == "1.0":
        return 1.0
    elif type_name == "B" or type_name=="0" or type_name == "0.0":
        return 0.0
    elif isinstance(type_name, int):
        return type_name



def preprocess_data(df):
    df.iloc[:, 0] = df.iloc[:, 0].apply(convert_to_float)
    data = np.array(df).astype(np.float32)

    # truths and inputs
    truths = data[:, 0]           # (N,)
    inputs = data[:, 1:]          # (N,M)

    # nomalize col
    X_min = inputs.min(axis=0)
    X_max = inputs.max(axis=0)
    inputs_norm = (inputs - X_min) / (X_max - X_min + 1e-8)

    # make sigle data always in a shape of column
    truths = truths[:, np.newaxis]     # (N,V)
    inputs = inputs_norm[:, :]          # (N,M)

    print("[INPUTS]", inputs.shape)   # (N,M)
    print("[TRUTHS]", truths.shape)   # (N,V)
    
    return inputs, truths


def load(path: str, index: bool) -> pd.DataFrame:
    """Load a csv file and return it's pandas.DataFrame object."""

    try:
        assert isinstance(path, str), "Wrong file path type."
        assert os.path.exists(path), "Data file not exists."
        assert path.endswith(".csv"), "Not csv data."

        if index:
            df = pd.read_csv(path, index_col=0)
        else:
            df = pd.read_csv(path)
        return df

    except AssertionError as e:
        print("AssertionError:", e)
        return None

    except Exception as e:
        print("Error:", e)
        return None
