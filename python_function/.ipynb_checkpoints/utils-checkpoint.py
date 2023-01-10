import pandas as pd
import numpy as np 
import joblib

def check_cols(preprocessor):
    all_columns = []
    for i in range(len(preprocessor.transformers_)):
        if preprocessor.transformers_[i][0] == "num":
            all_columns.extend(preprocessor.transformers[i][2])
        else:
            for j in range(len(preprocessor.transformers_[1][1].steps)):
                for array_ja in preprocessor.transformers_[1][1].steps[j][1].categories_:
                    all_columns.extend(list(array_ja))
    return all_columns

def insert_cols(prepro_1, prepro_2):
    num = []
    j = -1
    for i in check_cols(prepro_1):
        j += 1
        if i not in check_cols(prepro_2):
            num.append(j)
    return num

def check_type_numeric(df)-> list[str]:
    df_columns = list(df.columns)
    numeric = []
    catagories = []
    for values in df.columns:
        if df[values].dtypes == int or df[values].dtypes == float or df[values].dtypes == 'uint8':
            numeric.append(values)
        else:
            catagories.append(values)
    return numeric

def check_type(df)-> list[str]:
    df_columns = list(df.columns)
    numeric = []
    catagories = []
    for values in df.columns:
        if df[values].dtypes == int or df[values].dtypes == float or df[values].dtypes == 'uint8':
            numeric.append(values)
        else:
            catagories.append(values)
    return numeric, catagories

def eval_to_csv(X, y, preprocessor, target, task, model):
    model_evaluate = pd.concat([X, y], axis = 1).rename(columns = {target: "y_true"})
    model_evaluate["y_pred"] = model.predict(preprocessor.fit_transform(X))
    if task == "regression":
        model_evaluate["diff"] = model_evaluate["y_true"] - model_evaluate["y_pred"]
    else:
        _check = (model_evaluate["y_true"] == model_evaluate["y_pred"]).values
        model_evaluate["diff"] = _check
    return model_evaluate
