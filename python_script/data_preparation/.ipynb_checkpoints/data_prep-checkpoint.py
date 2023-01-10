from script.data_preparation.clean import clean
from script.data_preparation.convertion import *
import pandas as pd
import numpy as np
from typing import TypeVar
pandas_df = TypeVar("pandas")

def preparation(df: pandas_df, target_col:str):
    convertion(clean(df, target_col)).to_csv("/ml_data/after_preprocessing.csv")
    return convertion(clean(df, target_col))