import numpy as np
import warnings
from tqdm import tqdm
import shap
import pandas as pd
import joblib
import shap
import sys
sys.path.append("/code/notebook")
from script.utils import *

# model 
f = open("/code/notebook/script/model.txt", 'r')
_model = [x for x in f.readlines()]
# pipeline
f = open("/code/notebook/script/pipeline.txt", 'r')
_pipeline = [x for x in f.readlines()]

X_test = pd.read_csv("/ml_data/X_test.csv").drop(columns = "Unnamed: 0")
y_test = pd.read_csv("/ml_data/y_test.csv").drop(columns = "Unnamed: 0")
X_train_1 = pd.read_csv("/ml_data/transform.csv").drop(columns = ["Unnamed: 0"])
model = joblib.load(_model[-1])
preprocessor = joblib.load(_pipeline[-1])

def save_to_csv():
    all_columns = check_cols(preprocessor)
    explainer = shap.TreeExplainer(model[0] ,feature_names = all_columns)
    count_num = len(check_type_numeric(X_test))
    for_pandas = preprocessor.fit_transform(X_test)
    X_test_ja = pd.DataFrame(for_pandas, columns = all_columns)
    X_test_ja.iloc[:,:count_num] = preprocessor.transformers_[0][1].steps[1][1].inverse_transform(X_test_ja.iloc[:,:count_num])
    shap_values = explainer.shap_values(X_test_ja.values)
    global_values = pd.DataFrame(np.reshape(sum(np.abs(shap_values))/len(sum(np.abs(shap_values))), newshape = (1,-1)), columns = all_columns)
    train_dependency_values = pd.DataFrame(np.abs(shap_values), columns = ["importance_values_" + str(i) for i in all_columns])
    pd.concat([train_dependency_values, X_test_ja], axis = 1).to_csv("/model_interpret/local.csv")
    global_values.to_csv("/model_interpret/global.csv")
