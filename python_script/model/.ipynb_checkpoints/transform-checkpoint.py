import pandas as pd
import pickle
from sklearn.preprocessing import *
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from typing import TypeVar
import numpy as np
import datetime
import joblib
pandas_df = TypeVar("pandas")
sklearn_function = TypeVar("sklearn_function")
sklearn_pipeline = TypeVar("sklearn_pipeline")

f = open('/code/notebook/script/model/myfile.txt', 'r')
list_pipeline = [x for x in f.readlines()]
if list_pipeline:
    last_timestamp = list_pipeline[0]

def save_pipeline_name(pipeline_name:str ):
    if pipeline_name not in list_pipeline:
        file1 = open('/code/notebook/script/model/myfile.txt', 'a')
        file1.writelines([pipeline_name + " \n"])

def check_save_pipeline(name_ja:str):
    path = Path(f"/save_pipeline/{name_ja}")
    return path.is_file()

def check_save_file():
    path = Path(f"/ml_data/transform.csv")
    return path.is_file()

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

def columns_ja(preprocessor):
    all_columns = []
    for i in range(len(preprocessor.transformers_)):
        if preprocessor.transformers_[i][0] == "num":
            all_columns.extend(preprocessor.transformers[i][2])
        else:
            for j in range(len(preprocessor.transformers_[1][1].steps)):
                for array_ja in preprocessor.transformers_[1][1].steps[j][1].categories_:
                    all_columns.extend(list(array_ja))
    return all_columns

def outlier_removal(X):
    X = pd.DataFrame(X).copy()
    for i in range(X.shape[1]):
        x = pd.Series(X.iloc[:,i]).copy()
        q1 = x.quantile(0.25)
        q3 = x.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        X.iloc[((X.iloc[:,i] < lower_bound) | (X.iloc[:,i] > upper_bound)),i] = np.nan 
    return X

def _num_pipeline(pipeline, remove_outlier):
    if pipeline:
        if remove_outlier:
            num_pipe = [('outlier', FunctionTransformer(outlier_removal))]
            num_pipe.extend(pipeline)
            return Pipeline(steps = num_pipe)
        else:
            return Pipeline(steps = pipeline)
    else:
        return None

def _cat_pipeline(pipeline, remove_outlier):
    if pipeline:
        return Pipeline(steps = pipeline)
    else:
        return None

def _transformer(df, num_pipeline, cat_pipeline, remove_outlier):
    numeric_features, categorical_features = check_type(df)
    if _cat_pipeline(cat_pipeline, remove_outlier) != None and _num_pipeline(num_pipeline, remove_outlier) != None:
        preprocessor = ColumnTransformer(
            transformers = [
                ('num', _num_pipeline(num_pipeline, remove_outlier), numeric_features),
                ('cat', _cat_pipeline(cat_pipeline, remove_outlier), categorical_features)], n_jobs = -1)
        return preprocessor
    else:
        if _cat_pipeline(cat_pipeline, remove_outlier) != None:
            preprocessor = ColumnTransformer(
            transformers = [
                ('cat', _cat_pipeline(cat_pipeline, remove_outlier), categorical_features)])
            return preprocessor
        if _num_pipeline(num_pipeline, remove_outlier) != None:
            preprocessor = ColumnTransformer(
                transformers = [
                    ('num', _num_pipeline(num_pipeline, remove_outlier), numeric_features)])
            return preprocessor
        
def transformer(df ,num_pipeline = [], cat_pipeline = [], remove_outlier:bool = False):
    preprocessor = _transformer(df, num_pipeline, cat_pipeline, remove_outlier)
    _fit = preprocessor.fit(df)
    all_columns =  columns_ja(_fit)
    joblib.dump(_fit, f'/save_pipeline/pipeline_{datetime.datetime.now().strftime("%d_%m_%y_%X")}.pkl')
    save_pipeline_name(f'/save_pipeline/pipeline_{datetime.datetime.now().strftime("%d_%m_%y_%X")}.pkl')
    pd.DataFrame(preprocessor.fit_transform(df), columns = all_columns).to_csv(f"/ml_data/transform.csv")
        
        
        

