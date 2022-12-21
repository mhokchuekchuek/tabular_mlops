import numpy as np
import warnings
from lime import submodular_pick
from tqdm import tqdm
from lime.lime_tabular import LimeTabularExplainer
import pandas as pd
import joblib

X_train = pd.read_csv("/ml_data/X_train.csv").drop(columns = "Unnamed: 0")
y_train = pd.read_csv("/ml_data/y_train.csv").drop(columns = "Unnamed: 0")
X_test = pd.read_csv("/ml_data/X_test.csv").drop(columns = "Unnamed: 0")
y_test = pd.read_csv("/ml_data/y_test.csv").drop(columns = "Unnamed: 0")
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
            
def load_transform(file_name):
    preprocessor = joblib.load(f'/save_pipeline/{file_name}.pkl')
    return preprocessor
    
if str(model[1]) ==  'KNeighborsClassifier()':
    pred_model = model[-1].predict_proba
else:
    pred_model = model[-1].predict

def check_type(df)-> list[str]:
    df_columns = list(df.columns)
    numeric = []
    catagories = []
    for values in df.columns:
        if df[values].dtypes == int or df[values].dtypes == float:
            numeric.append(values)
        else:
            catagories.append(values)
    return catagories

    
def check_file():
    path = Path("/model_interpret")
    return path.is_file()
    
def _explainer(task, target):
    if task == "classification":
        class_names = y_train[target].unique()
        # Fit the Explainer on the training data set using the LimeTabularExplainer 
        explainer = LimeTabularExplainer(load_transform(pipeline_name).fit_transform(X_train), feature_names = all_columns, class_names = class_names, mode = task, categorical_features = cat_feature)
    else:
        explainer = LimeTabularExplainer(load_transform(pipeline_name).fit_transform(X_train), feature_names = all_columns, mode = "regression", class_names = [target], categorical_features = cat_feature)
        
    return explainer

def return_x(explaination, int_x):
    return explaination.domain_mapper.feature_names[int_x]

def return_x_2(explaination,int_x, i):
    return explaination.explanations[i].domain_mapper.feature_names[int_x]

def _to_csv_local(lime_obj, num):
    cols = ["importance_values_"+i for i in [lime_obj.domain_mapper.feature_names][0]]
    values = lime_obj.domain_mapper.feature_values
    arr = np.array(values).reshape((1, len(cols)))
    df_x = pd.DataFrame(arr , columns = cols)
    return pd.concat([X_test[X_test.index == num], df_x], axis = 1)
    

def _to_csv_global(lime_obj):
    all_df = []
    for i in range(0,5):
        cols = lime_obj.explanations[i].domain_mapper.feature_names
        values = lime_obj.explanations[i].domain_mapper.feature_values
        arr = np.array(values).reshape((1, len(list(lime_obj.explanations[i].local_exp.values())[0])))
        all_df.append(pd.DataFrame(arr , columns = cols))
    return pd.concat(all_df, ignore_index =True)

def save_to_csv(task, target):
    all_df_local = []
    num = -1
    for i in tqdm(model[0].fit_transform(X_test)):
        num+=1
        explainer = _explainer(task, target)
        explaination = explainer.explain_instance(i, pred_model)
        all_df_local.append(_to_csv_local(explaination, num))
        
    explainer = _explainer(task, target)
    sp_obj = submodular_pick.SubmodularPick(explainer, model[0].fit_transform(X_train), pred_model, num_features= len(all_columns), num_exps_desired = 5)
    all_df_global = _to_csv_global(sp_obj)
    pd.concat(all_df_local, ignore_index = True).fillna(0).to_csv("/model_interpret/local.csv")
    all_df_global.fillna(0).to_csv("/model_interpret/global.csv")
