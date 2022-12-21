from typing import TypeVar
from sklearn.pipeline import Pipeline
from script.model.transform import transformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from script.utils.model_catalogue import ModelCatalogue
from hyperopt import tpe
from hyperopt import STATUS_OK
from hyperopt import Trials
from hyperopt import hp
from hyperopt import fmin
from sklearn.pipeline import Pipeline
import pandas as pd
import mlflow
from sklearn.preprocessing import *
from functools import partial
import numpy as np
import joblib
from pathlib import Path

model_name = TypeVar("model_name")
params = TypeVar("model_params")
pandas_df = TypeVar("pandas")
model_1 = ModelCatalogue()
# classification metric
classification_metric = ["f1_macro"]
#load_pipeline_name
f = open('/code/notebook/script/model/myfile.txt', 'r')
list_pipeline = [x for x in f.readlines()]

def save_pipeline_name(pipeline_name:str ):
    if pipeline_name not in list_pipeline:
        file1 = open('/code/notebook/script/model/myfile.txt', 'w')
        file1.writelines([pipeline_name])
        
def check_file(name_ja:str):
    path = Path(f"/save_pipeline/{name_ja}")
    return path.is_file()
    
def _str_outlier(str_ja):
    if str_ja:
        return "remove_outlier"
    return 'non_remove_outlier'

def model(df:pandas_df, model_dict, remove_outlier:bool = False, strategy:str = 'median', Scaler = StandardScaler(), transform = True,overwrite = False)-> dict[model_name, params]:
    # for the first_time please use treansform == False
    if transform:
        # set_file_name
        file_name = f"{_str_outlier(remove_outlier)} {strategy} {str(Scaler)[:-2]}"
        # save_file_name
        save_pipeline_name(file_name)
        print(file_name)
        #save_preprocess_pipeline and train_model
        if overwrite == True:
            preprocessor = transformer(df, remove_outlier, strategy, Scaler)
            model = model_1.get_model(model_dict["model"])
            clf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model(**model_dict["params"]))])
            joblib.dump(preprocessor, f'/save_pipeline/{file_name}.pkl')
        else:
            if check_file(file_name):
                preprocessor = joblib.load(f'/save_pipeline/{file_name}.pkl')
                model = model_1.get_model(model_dict["model"])
                clf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('model', model(**model_dict["params"]))])
            else:
                preprocessor = transformer(df, remove_outlier, strategy, Scaler)
                model = model_1.get_model(model_dict["model"])
                clf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('model', model(**model_dict["params"]))])
                joblib.dump(preprocessor, f'/save_pipeline/{file_name}.pkl')
    else:
        model = model_1.get_model(model_dict["model"])
        clf_pipeline = Pipeline(steps=[('model', model(**model_dict["params"]))])
    return clf_pipeline
    
def train(X_train, y_train, model_dict, remove_outlier:bool = False, strategy:str = 'median', Scaler = StandardScaler()):
    training = model(X, model_dict)
    return training.fit(X, y)

def objective(args, X, y, score, remove_outlier:bool = False, strategy:str = 'median', Scaler = StandardScaler(), transform = True,overwrite = False):
    mlflow.sklearn.autolog()
    with mlflow.start_run(run_name = f"{args['model']} {args['params']}", nested=True):
        clf = model(X, args)
        if score in classification_metric:
            skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 1)
            scores = cross_val_score(clf, X, y, cv = skf, scoring = score).mean()
        else:
            scores = -cross_val_score(clf, X, y, cv = 5, scoring = score).mean()
        # Loss must be minimized
        if score in classification_metric:
            loss = 1 - scores
        else:
            loss = scores
        mlflow.log_metric(f"avg_{score}", scores)
        # Dictionary with information for evaluation
        return {'loss': loss, 'params': params, 'status': STATUS_OK}

def mlflow_train(space, X, y, score, remove_outlier:bool = False, strategy:str = 'median', Scaler = StandardScaler(), transform = True,overwrite = False):
    for sp in space:
        mlflow_name = f"{sp['model']}"
        mlflow.set_experiment(mlflow_name)
        tpe_algorithm = tpe.suggest
        bayes_trials = Trials()
        if transform:
            mlflow_name = f"{_str_outlier(remove_outlier)} {strategy} {str(Scaler)[:-2]}"
        else:
            mlflow_name = "no_transform"
        with mlflow.start_run(run_name = mlflow_name):
            best = fmin(fn=partial(objective, X = X, y = y, score = score, remove_outlier = remove_outlier, strategy = strategy, Scaler = Scaler, transform = transform, overwrite = overwrite), space = sp, algo = tpe.suggest, max_evals = 1, trials = bayes_trials)
            
    return best


