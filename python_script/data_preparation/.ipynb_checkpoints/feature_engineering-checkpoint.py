import re
import pandas as pd 

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

def check_type(df)-> list[str]:
    df_columns = list(df.columns)
    numeric = []
    catagories = []
    for values in df.columns:
        if df[values].dtypes == int or df[values].dtypes == float:
            numeric.append(values)
        else:
            catagories.append(values)
    return catagories

def engineering(df_ja):
    df_ja["feature"] = df_ja["feature"].apply(str_to_list)
    for i in ["BRT_distance", "BTS_distance", "MRT_distance", "APL_distance"]:
        df_ja[i] = df_ja[i].apply(str_to_int)
    a = pd.get_dummies(df_ja['feature'].explode()).sum(level=0)
    a = a.astype({col : "int64" for col in a.columns})
    return pd.concat([df_ja, a], axis = 1).drop(columns = ["feature"])