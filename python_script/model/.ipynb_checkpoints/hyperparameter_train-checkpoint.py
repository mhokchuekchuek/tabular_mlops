from hyperopt import tpe
from hyperopt import STATUS_OK
from hyperopt import Trials
from hyperopt import hp
from hyperopt import fmin
from sklearn.pipeline import Pipeline
import pandas as pd
import mlflow
from script.model.transform import transform
from functools import partial
import numpy as np

def model(df, args):
    preprocessor = transform(df)
    model_wa = args["model"]
    clf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model_wa(**args["params"]))])
    return clf_pipeline

def objective(args, X, y, score):
    mlflow.sklearn.autolog()
    with mlflow.start_run(nested=True):
        clf = model(X, args)
        scores = cross_val_score(clf, X, y, cv = 5, scoring = score).mean()
        best_score = max(scores)
        loss = 1 - best_score
        return {'loss': 1 - np.median(scores), 'status': STATUS_OK}

def mlflow_train(space, X, y, score):
    for sp in space:
        mlflow_name = sp["model"].__name__
        mlflow.set_experiment(mlflow_name)
        tpe_algorithm = tpe.suggest
        bayes_trials = Trials()
        with mlflow.start_run():
            best = fmin(fn=partial(objective, X = X, y = y, scoring = score), space = sp, algo = tpe.suggest, max_evals = 10, trials = bayes_trials)        
    return best