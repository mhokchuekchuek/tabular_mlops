import pandas as pd
import numpy as np
from typing import TypeVar
import re
pandas_df = TypeVar("pandas")

'''
cleaninq
1) change column type [str => lsit] : feature
2) change column type [str => int] : ["BRT_distance", "BTS_distance", "MRT_distance", "APL_distance"]
'''

def str_to_list(x):
    list_x = []
    if x.split(","):
        for i in x.split(","):
            list_x.append(re.sub("[\[\'\]]", "", i).strip())
    else:
        list_x.append(re.sub("[\[\'\]]", "", x).strip())
    return list_x

def str_to_int(x):
    if x == "no":
        return 10000
    return int(x)

def remove_duplicates(df: pandas_df, target_col:str)-> pandas_df:
    df_1 = df[~(df[target_col].isna())]
    df_2 = df_1.drop_duplicates()
    return df_2

def clean(df: pandas_df, target_col:str)-> pandas_df:
    df_ja = remove_duplicates(df, target_col)
    df_ja["feature"] = df_ja["feature"].apply(str_to_list)
    for i in ["BRT_distance", "BTS_distance", "MRT_distance", "APL_distance"]:
        df_ja[i] = df_ja[i].apply(str_to_int)
    return df_ja