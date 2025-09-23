#!/usr/bin/python3

import pandas as pd
import numpy as np
import os


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

    # nomalize col
    X_min = inputs.min(axis=0)
    X_max = inputs.max(axis=0)
    inputs_norm = (inputs - X_min) / (X_max - X_min + 1e-8)

    # make sigle data always in a shape of column
    truths = truths[:, np.newaxis, np.newaxis]       # (N,1,1)
    inputs = inputs_norm[:, :, np.newaxis]          # (N,M,1)

    print("[INPUTS]", inputs.shape)   # (N,M,1)
    print("[TRUTHS]", truths.shape)   # (N,1,1)
    
    return inputs, truths


def load(path: str) -> pd.DataFrame:
    """Load a csv file and return it's pandas.DataFrame object."""

    try:
        assert isinstance(path, str), "Wrong file path type."
        assert os.path.exists(path), "Data file not exists."
        assert path.endswith(".csv"), "Not csv data."

        df = pd.read_csv(path, index_col=0)
        return df

    except AssertionError as e:
        print("AssertionError:", e)
        return None

    except Exception as e:
        print("Error:", e)
        return None
