from typing import TypeVar
from sklearn.pipeline import Pipeline
from script.model.transform import transformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
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
from sklearn.metrics import *
from functools import partial
import numpy as np
import joblib
from pathlib import Path
import datetime

model_name = TypeVar("model_name")
params = TypeVar("model_params")
pandas_df = TypeVar("pandas")
model_1 = ModelCatalogue()

# classification metric
classification_metric = ["f1_macro"]

#load_pipeline_name
f = open('/code/notebook/script/model/myfile.txt', 'r')
list_pipeline = [x for x in f.readlines()]
if list_pipeline:
    last_timestamp = list_pipeline[-1].replace(" \n","")
    pipeline_ja = joblib.load(last_timestamp)

def model(df:pandas_df, model_dict)-> dict[model_name, params]:
    model = model_1.get_model(model_dict["model"])
    clf_pipeline = Pipeline(steps=[('model', model(**model_dict["params"], n_jobs = -1))])
    return clf_pipeline

def preprocess_tagging(preprocessor):
    a = {}
    for i in preprocessor.transformers:
        for eiei in i[1].steps:
            a[eiei[0]] = str(eiei[1])
    a["preprocess_path"] = list_pipeline[-1]
    return a
    
def train(X_train, y_train, model_dict):
    training = model(X, model_dict)
    return training.fit(X, y)

def objective(args_ja, X, y, score):
    mlflow.sklearn.autolog()   
    # with mlflow.start_run(run_name = f"{args['model']} {args['params']}", nested = True):
    for args in args_ja: 
        with mlflow.start_run(run_name = f"{args['model']} {args['params']}", nested = True) as nested_run:
            mlflow.set_tags(preprocess_tagging(pipeline_ja))
            clf = model(X, args)
            if score in classification_metric:
                skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 1)
                scores = cross_val_score(clf, X, y, cv = skf, scoring = score).mean()
            else:
                kf = KFold(n_splits = 5, shuffle = True, random_state = 1)
                scores = -cross_val_score(clf, X, y, cv = kf, scoring = score).mean()
            # Loss must be minimized
            if score in classification_metric:
                loss = 1 - scores
            else:
                loss = scores
            mlflow.log_metric(f"avg_{score}", scores)
            # Dictionary with information for evaluation
    return {'loss': loss, 'params': params, 'status': STATUS_OK}

# def mlflow_train(space, X, y, score, remove_outlier:bool = False, strategy:str = 'median', Scaler = StandardScaler(), transform = True,overwrite = False):
#     for sp in space:
#         mlflow_name = f"{sp['model']}"
#         mlflow.set_experiment(mlflow_name)
#         tpe_algorithm = tpe.suggest
#         bayes_trials = Trials()
#         if transform:
#             mlflow_name = f"{_str_outlier(remove_outlier)} {strategy} {str(Scaler)[:-2]}"
#         else:
#             mlflow_name = "no_transform"
#         with mlflow.start_run(run_name = mlflow_name):
#             best = fmin(fn=partial(objective, X = X, y = y, score = score, remove_outlier = remove_outlier, strategy = strategy, Scaler = Scaler, transform = transform, overwrite = overwrite), space = sp, algo = tpe.suggest, max_evals = 1, trials = bayes_trials)
            
#     return best

# def trainer(space, X, y, score, experiment_name:str):
#     mlflow.set_experiment("ddproperty")
#     experiment = mlflow.get_experiment_by_name("ddproperty")
#     with mlflow.start_run(run_name = "model", experiment_id=experiment.experiment_id) as run:
#         for sp in space:
#             tpe_algorithm = tpe.suggest
#             bayes_trials = Trials()
#             best = fmin(fn=partial(objective, X = X, y = y, score = score), space = sp, algo = tpe.suggest, max_evals = 1, trials = bayes_trials)

#     return best

def trainer(space, X, y, score, experiment_name:str):
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    tpe_algorithm = tpe.suggest
    bayes_trials = Trials()
    best = fmin(fn=partial(objective, X = X, y = y, score = score), space = space, algo = tpe.suggest, max_evals = 1, trials = bayes_trials)
    return best