from typing import TypeVar
from sklearn.pipeline import Pipeline
from script.model.transform import transformer
from sklearn.linear_model import LogisticRegression
from script.utils.model_catalogue import ModelCatalogue

model_name = TypeVar("model_name")
params = TypeVar("model_params")
pandas_df = TypeVar("pandas")
model_1 = ModelCatalogue()

def model(df:pandas_df, model_dict)-> dict[model_name, params]:
    model = model_1.get_model(model_dict["model"])
    clf_pipeline = Pipeline(steps=[('model', model(**model_dict["params"], n_jobs = -1))])
    return clf_pipeline
    
def train(X, y, model_dict):
    training = model(X, model_dict)
    return training.fit(X, y)

    
