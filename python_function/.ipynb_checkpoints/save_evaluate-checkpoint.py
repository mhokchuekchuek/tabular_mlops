import pandas as pd
import joblib
from sklearn.metrics import *

X_train = pd.read_csv("/ml_data/X_train.csv").drop(columns = "Unnamed: 0")
y_train = pd.read_csv("/ml_data/y_train.csv").drop(columns = "Unnamed: 0")
X_test = pd.read_csv("/ml_data/X_test.csv").drop(columns = "Unnamed: 0")
y_test = pd.read_csv("/ml_data/y_test.csv").drop(columns = "Unnamed: 0")
model = joblib.load("/artifact/mlruns/106265642224330824/382b875fa1544292b580a0674ec371dc/artifacts/model/model.pkl")

# test_set
target = y_train.columns[0]
model_evaluate = pd.concat([X_test, y_test], axis = 1).rename(columns = {target: "y_true"})
model_evaluate["y_pred"] = model.predict(X_test)
model_evaluate["diff"] = model_evaluate["y_true"] - model_evaluate["y_pred"]
model_evaluate.to_csv("/ml_data/evaluate_test.csv")
# train_set 
model_evaluate_train = pd.concat([X_train, y_train], axis = 1).rename(columns = {target: "y_true"})
model_evaluate_train["y_pred"] = model.predict(X_train)
model_evaluate_train["diff"] = model_evaluate_train["y_true"] - model_evaluate_train["y_pred"]
model_evaluate_train.to_csv("/ml_data/evaluate_train.csv")

def evaluate_csv(eval_train_1, eval_test_1):
    index_name = ["Train", "Test"]
    dict_eval = {"neg_mae":[], "neg_rmse":[]}
    for index in index_name:
        if index == "Train":
            dict_eval["neg_mae"].append(mean_absolute_error(eval_train_1["y_true"], eval_train_1["y_pred"]))
            dict_eval["neg_rmse"].append(mean_squared_error(eval_train_1["y_true"], eval_train_1["y_pred"], squared = False))
        else:
            dict_eval["neg_mae"].append(mean_absolute_error(eval_test_1["y_true"], eval_test_1["y_pred"]))
            dict_eval["neg_rmse"].append(mean_squared_error(eval_test_1["y_true"], eval_test_1["y_pred"], squared = False))
    return pd.DataFrame(dict_eval, index = index_name).T

def evaluate_csv_classification(eval_train, eval_test):
    index_name = eval_train["y_true"].unique()
    dict_eval = {"f1_macro_train":[], "f1_macro_test":[]}
    for i in index_name:
        eval_train_ja = eval_train[eval_train["y_true"] == i]
        dict_eval["f1_macro_train"].append(f1_score(eval_train_ja["y_true"], eval_train_ja["y_pred"], average='macro'))
        eval_test_ja = eval_test[eval_test["y_true"] == i]
        dict_eval["f1_macro_test"].append(f1_score(eval_test_ja["y_true"], eval_test_ja["y_pred"], average='macro'))
    return pd.DataFrame(dict_eval, index = index_name)

def to_csv_eval(task):
    if task == "regression":
        return evaluate_csv(model_evaluate_train, model_evaluate).to_csv("/ml_data/visual_eval.csv")
    return evaluate_csv_classification(model_evaluate_train, model_evaluate).to_csv("/ml_data/visual_eval.csv")
        