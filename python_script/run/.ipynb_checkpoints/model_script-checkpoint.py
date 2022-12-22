import sys
sys.path.append('../..')
sys.path.append('/code/notebook/')
from hyperopt import Trials, STATUS_OK, tpe, hp, fmin, STATUS_FAIL, space_eval
from script.data_preparation.data_prep import preparation
from script.model.hyperparams_v2 import *
from script.utils.split import *

# import data
import pandas as pd
data = pd.read_csv("/ml_data/ddproperty_final.csv").drop(columns = "Unnamed: 0")

#split_data
X_train, X_test, y_train, y_test= split_data_with_drift_check(data, "regression", "price")

#train
a = [
    {
    'model':"RandomForestRegressor",
    'params':{}},
    {
    'model': "XGBRegressor",
    'params': {}},
    {
    'model': "LGBMRegressor",
    'params': {}}
]

mlflow_train(a, X_train, y_train, "neg_root_mean_squared_error", transform = True, overwrite = True)
