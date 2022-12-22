import numpy as np
import warnings
from tqdm import tqdm
import shap
import pandas as pd
import joblib
import shap

X_train = pd.read_csv("/ml_data/X_train.csv").drop(columns = "Unnamed: 0")
y_train = pd.read_csv("/ml_data/y_train.csv").drop(columns = "Unnamed: 0")
X_test = pd.read_csv("/ml_data/X_test.csv").drop(columns = "Unnamed: 0")
y_test = pd.read_csv("/ml_data/y_test.csv").drop(columns = "Unnamed: 0")
feature_names = list(X_train.columns)
model = joblib.load("/artifact/mlruns/813623427044464195/69402f2e004a4032b669f7800a476315/artifacts/model/model.pkl")
# model = joblib.load("/artifact/mlruns/941649382202349625/e311be07056d4a4fa7413ec46830e13b/artifacts/model/model.pkl")
task = "regression"
target = "price"
pipeline_name = "non_remove_outlier median StandardScaler"

all_columns = []
cat_feature = []
for i in range(len(model["preprocessor"].transformers_)):
    if model["preprocessor"].transformers_[i][0] == "num":
        all_columns.extend(model["preprocessor"].transformers[i][2])
    else:
        for j in range(len(model['preprocessor'].transformers_[1][1].categories_)):
            all_columns.extend(list(model['preprocessor'].transformers_[1][1].categories_[j]))
            cat_feature.append(list(model['preprocessor'].transformers_[1][1].categories_[j]))
            
def save_to_csv():
    explainer = shap.TreeExplainer(model[1] ,feature_names = all_columns)
    X_test_ja = pd.DataFrame(model[0].fit_transform(X_test), columns = all_columns)
    ## inverse standard scaler => check_columns_that_numeric_and_inverse_it 
    X_test_ja.iloc[:,:79] = model[0].transformers_[0][1].steps[1][1].inverse_transform(X_test_ja.iloc[:,:79])
    shap_values = explainer.shap_values(model[0].fit_transform(X_test))
    global_values = pd.DataFrame(np.reshape(sum(np.abs(shap_values))/len(sum(np.abs(shap_values))), newshape = (1,-1)), columns = all_columns)
    train_dependency_values = pd.DataFrame(np.abs(shap_values), columns = ["importance_values_" + i for i in all_columns])
    pd.concat([train_dependency_values, X_test_ja], axis = 1).to_csv("/model_interpret/local.csv")
    global_values.to_csv("/model_interpret/global.csv")
