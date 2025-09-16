#!/usr/bin/python3

import pandas as pd
import os


def load(path: str) -> pd.DataFrame:
    """Load a csv file and return it's pandas.DataFrame object."""

    try:
        assert isinstance(path, str), "Wrong file path type."
        assert os.path.exists(path), "Data file not exists."
        assert path.endswith(".csv"), "Not csv data."

        df = pd.read_csv(path, index_col=0)
        print("Loading dataset of dimensions", df.shape)
        return df

    except AssertionError as e:
        print("AssertionError:", e)
        return None

    except Exception as e:
        print("Error:", e)
        return None


# def clean_data(df: dataframe, method:str="dropnan") -> dataframe:
# 	"""Clean data, drop nan value line"""

# 	if method == "dropnan":
# 		num_cols = df.select_dtypes(include=["number"]).columns
# 		df.dropna(subset=num_cols)
# 	elif method == "raw":
# 		return df
# 	elif method == "mean":
# 		return df.fillna(df.mean(numeric_only=True))
# 	elif method == "median":
# 		return df.fillna(df.median(numeric_only=True))
# 	return df
