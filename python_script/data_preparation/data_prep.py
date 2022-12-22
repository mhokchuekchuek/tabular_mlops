from script.data_preparation.clean import clean
import pandas as pd
import numpy as np
from typing import TypeVar
pandas_df = TypeVar("pandas")

def preparation(df: pandas_df, target_col:str):
    engineering(clean(df, target_col)).to_csv("/ml_data/before_preprocessing.csv")
    return clean(df, target_col)