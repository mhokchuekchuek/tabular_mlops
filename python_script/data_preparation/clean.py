import pandas as pd
import numpy as np
from typing import TypeVar
pandas_df = TypeVar("pandas")

def remove_duplicates(df: pandas_df, target_col:str)-> pandas_df:
    df_1 = df[~(df[target_col].isna())]
    df_2 = df_1.drop_duplicates()
    return df_2

def clean(df: pandas_df, target_col:str)-> pandas_df:
    df_ja = remove_duplicates(df, target_col)
    return df_ja