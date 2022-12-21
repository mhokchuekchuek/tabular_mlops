import pandas as pd
import numpy as np
from typing import TypeVar
pandas_df = TypeVar("pandas")

def remove_duplicates(df: pandas_df, target_col:str)-> pandas_df:
    df_1 = df[~(df[target_col].isna())]
    df_2 = df_1.drop_duplicates()
    return df_2

def check_outlier(df_series):
    Q1,Q3 = np.percentile(df_series , [25,75])
    IQR = Q3 - Q1
    ul = Q3+1.5*IQR
    ll = Q1-1.5*IQR
    return ul,ll

def impute_outlier(df:pandas_df, target_col:str)-> pandas_df: 
    col = []
    for values in df.columns:
        col.append(values)
        if values != target_col:
            if df[values].dtypes == int or df[values].dtypes == float:
                upper, lower = check_outlier(df[values].values)
                median = float(df[values].median())
                # df[values] = np.where((df[values] < lower) | (df[values] > upper), median, df[values])
                df.loc[(df[values] < lower) | (df[values] > upper), values] = np.nan
    return df.reset_index()[col] # target should not be NaN

def clean(df: pandas_df, target_col:str)-> pandas_df:
    df_ja = remove_duplicates(df, target_col)
    df_ja = impute_outlier(df_ja, target_col)
    return df_ja