import pandas as pd
import numpy as np
from typing import TypeVar
import re
pandas_df = TypeVar("pandas")

'''
conertion
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


def convertion(df_ja):
    df_ja["feature"] = df_ja["feature"].apply(str_to_list)
    # df_ja["built_year"] = df_ja["built_year"].astype('category')
    for i in ["BRT_distance", "BTS_distance", "MRT_distance", "APL_distance"]:
        df_ja[i] = df_ja[i].apply(str_to_int)
    a = pd.get_dummies(df_ja['feature'].explode()).sum(level=0)
    a = a.astype({col : "int64" for col in a.columns})
    return pd.concat([df_ja, a], axis = 1).drop(columns = ["feature"])