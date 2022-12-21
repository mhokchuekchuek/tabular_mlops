import pandas as pd
import pickle
from sklearn.preprocessing import *
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from typing import TypeVar
pandas_df = TypeVar("pandas")
sklearn_function = TypeVar("sklearn_function")
sklearn_pipeline = TypeVar("sklearn_pipeline")

def check_type(df:pandas_df)-> list[str]:
    df_columns = list(df.columns)
    numeric = []
    catagories = []
    for values in df.columns:
        if df[values].dtypes == int or df[values].dtypes == float or df[values].dtypes == 'uint8':
            numeric.append(values)
        else:
            catagories.append(values)
    return numeric, catagories

def check_outlier(df_series):
    Q1,Q3 = np.percentile(df_series , [25,75])
    IQR = Q3 - Q1
    ul = Q3+1.5*IQR
    ll = Q1-1.5*IQR
    return ul,ll

def impute_outlier(df:pandas_df)-> pandas_df: 
    col = []
    for values in df.columns:
        col.append(values)
        if df[values].dtypes == int or df[values].dtypes == float:
            upper, lower = check_outlier(df[values].values)
            median = float(df[values].median())
            # df[values] = np.where((df[values] < lower) | (df[values] > upper), median, df[values])
            df.loc[(df[values] < lower) | (df[values] > upper), values] = np.nan
    return df.reset_index()[col] # target should not be NaN

def transformer(df:pandas_df, remove_outlier:bool = False, strategy:str = 'median', Scaler = StandardScaler())-> sklearn_pipeline:
    if remove_outlier:
        numeric_features, categorical_features = impute_outlier(check_type(df))
    else:
        numeric_features, categorical_features = check_type(df)
    if numeric_features and categorical_features:
        numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy = strategy)),
            ('scaler', Scaler)])
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        preprocessor = ColumnTransformer(
            transformers = [
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)])
        return preprocessor

    if numeric_features and not categorical_features:
        numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy = strategy)),
            ('scaler', Scaler)])
        preprocessor = ColumnTransformer(
            transformers = [
                ('num', numeric_transformer, numeric_features)])
        return preprocessor
        
    if categorical_features and not numeric_features:
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        preprocessor = ColumnTransformer(
            transformers = [
                ('cat', categorical_transformer, categorical_features)])
    
        return preprocessor

