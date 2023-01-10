import sys
sys.path.append('../..')
sys.path.append('/code/notebook/')
from hyperopt import Trials, STATUS_OK, tpe, hp, fmin, STATUS_FAIL, space_eval
from script.data_preparation.data_prep import preparation
from script.model.hyperparams_v2 import *
from script.utils.split import *
from script.model.transform import *

# import data
import pandas as pd
data = pd.read_csv("/ml_data/ddproperty_final.csv").drop(columns = "Unnamed: 0")

#split_data
X_train, X_test, y_train, y_test= split_data_with_drift_check(data, "regression", "price")

#transform data 
num_pipeline = [('imputer', SimpleImputer(strategy = "median")), ('scaler', StandardScaler())]
cat_pipeline = [("one_hot", OneHotEncoder(handle_unknown='ignore'))]
transformer(X_train, num_pipeline, cat_pipeline)

#train
X_train_1 = pd.read_csv("/ml_data/transform.csv").drop(columns = ["Unnamed: 0"])

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

trainer(a, X_train_1, y_train,"neg_root_mean_squared_error", "ddproperty")
