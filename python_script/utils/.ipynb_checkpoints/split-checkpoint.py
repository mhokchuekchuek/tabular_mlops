import sys
from deepchecks.tabular.checks import MultivariateDrift
from deepchecks.tabular import Dataset
from sklearn.model_selection import train_test_split
from typing import TypeVar
from typing import Union
import pandas as pd
from script.data_preparation.data_prep import preparation
sys.path.append('../..')
pandas_df = TypeVar("pandas")

#set validation = false if we can do validation in cross-validation
def check_type(df:pandas_df)-> list[str]:
    df_columns = list(df.columns)
    numeric = []
    catagories = []
    for values in df.columns:
        if df[values].dtypes == int or df[values].dtypes == float or df[values].dtypes == 'uint8':
            numeric.append(values)
        else:
            catagories.append(values)
    return catagories

def _validation(X, y, task:str, _random_state, ratio:float = 0.2, validation_ratio:float = 0.25):
    if task == "classification":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = ratio, random_state = _random_state, stratify = y)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = validation_ratio, _random_state = _random_state, stratify = y_train) # 0.25 x 0.8 = 0.2
        X_test.to_csv("/ml_data/X_test.csv")
        y_test.to_csv("/ml_data/y_test.csv")
        X_train.to_csv("/ml_data/X_train.csv")
        y_train.to_csv("/ml_data/y_train.csv")
        X_val.to_csv("/ml_data/X_val.csv")
        y_val.to_csv("/ml_data/y_val.csv")
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = ratio, random_state = _random_state)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = validation_ratio, random_state = 1) # 0.25 x 0.8 = 0.2
        X_test.to_csv("/ml_data/X_test.csv")
        y_test.to_csv("/ml_data/y_test.csv")
        X_train.to_csv("/ml_data/X_train.csv")
        y_train.to_csv("/ml_data/y_train.csv")
        X_val.to_csv("/ml_data/X_val.csv")
        y_val.to_csv("/ml_data/y_val.csv")
    return X_train, X_test, X_val, y_train, y_test, y_val

def _no_validation(X, y, task:str, _random_state, ratio:float = 0.2):
    if task == "classification":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = ratio, random_state = _random_state, stratify = y)
        X_test.to_csv("/ml_data/X_test.csv")
        y_test.to_csv("/ml_data/y_test.csv")
        X_train.to_csv("/ml_data/X_train.csv")
        y_train.to_csv("/ml_data/y_train.csv")
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = ratio, random_state = _random_state)
        X_test.to_csv("/ml_data/X_test.csv")
        y_test.to_csv("/ml_data/y_test.csv")
        X_train.to_csv("/ml_data/X_train.csv")
        y_train.to_csv("/ml_data/y_train.csv")
    return X_train, X_test, y_train, y_test

def split_data(df:pandas_df, task:str, target_col: str,ratio:float = 0.2, validation:bool = False, validation_ratio:float = 0.25):
    list_task = ["regression", "classification"]
    assert task in list_task , "please sign 'regression' or 'classification' i sus"
    use_df = preparation(df, target_col)
    X = use_df.drop(columns = [target_col])
    y = use_df[target_col]
    rand_state = 1
    if validation:
        X_train, X_test, X_val, y_train, y_test, y_val = _validation(X, y, task, rand_state,ratio, validation_ratio)
        X_test.to_csv("/ml_data/X_test.csv")
        y_test.to_csv("/ml_data/y_test.csv")
        X_train.to_csv("/ml_data/X_train.csv")
        y_train.to_csv("/ml_data/y_train.csv")
        X_val.to_csv("/ml_data/X_val.csv")
        y_val.to_csv("/ml_data/y_val.csv")
        return X_train, X_test, X_val, y_train, y_test, y_val
    else:
        X_train, X_test, y_train, y_test = _no_validation(X, y, task, rand_state, ratio)
        X_test.to_csv("/ml_data/X_test.csv")
        y_test.to_csv("/ml_data/y_test.csv")
        X_train.to_csv("/ml_data/X_train.csv")
        y_train.to_csv("/ml_data/y_train.csv")
        return X_train, X_test, y_train, y_test

def check_drift(X_train, X_test):
    check_with_condition = MultivariateDrift()
    if check_type(X_train):
        dataset_drift_result = check_with_condition.run(Dataset(X_train,cat_features = check_type(X_train)), Dataset(X_test, cat_features = check_type(X_test)))
    else:
        dataset_drift_result = check_with_condition.run(Dataset(X_train), Dataset(X_test))
    return dataset_drift_result.passed_conditions()

def split_data_with_drift_check(df:pandas_df, task:str, target_col: str,ratio:float = 0.2, validation:bool = False, validation_ratio:float = 0.25):
    list_task = ["regression", "classification"]
    assert task in list_task , "please sign 'regression' or 'classification' i sus"
    check_with_condition = MultivariateDrift()
    use_df = preparation(df, target_col)
    X = use_df.drop(columns = [target_col])
    y = use_df[target_col]
    rand_state = 1
    if validation:
        X_train, X_test, X_val, y_train, y_test, y_val = _validation(X, y, task, rand_state, ratio, validation_ratio)
        while check_drift(X_train, X_test) == False:
            if check_drift(X_train, X_test) == True:
                break
                X_test.to_csv("/ml_data/X_test.csv")
                y_test.to_csv("/ml_data/y_test.csv")
                X_train.to_csv("/ml_data/X_train.csv")
                y_train.to_csv("/ml_data/y_train.csv")
                X_val.to_csv("/ml_data/X_val.csv")
                y_val.to_csv("/ml_data/y_val.csv")
            if rand_state > 50000:
                break
                print("cant_split_with_no_drift")
            rand_state += 1
            X_train, X_test, X_val, y_train, y_test, y_val = _validation(X, y, task, rand_state, ratio, validation_ratio)
        return X_train, X_test, X_val, y_train, y_test, y_val
    else: 
        X_train, X_test, y_train, y_test = _no_validation(X, y, task, rand_state, ratio)
        while check_drift(X_train, X_test) == False:
            if check_drift(X_train, X_test) == True:
                break
                X_test.to_csv("/ml_data/X_test.csv")
                y_test.to_csv("/ml_data/y_test.csv")
                X_train.to_csv("/ml_data/X_train.csv")
                y_train.to_csv("/ml_data/y_train.csv")
            if rand_state > 50000:
                break
                print("cant_split_with_no_drift")
            rand_state += 1
            X_train, X_test, y_train, y_test = _no_validation(X, y, task, rand_state, ratio)
        return X_train, X_test, y_train, y_test