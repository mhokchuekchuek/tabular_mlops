from importlib import import_module

class ModelCatalogue:
    """
    A list of models available for experiment
    
    """
    def __init__(self):
        self.model_catalogue = {
            "KNeighborsClassifier": f"sklearn.neighbors",
            "DecisionTreeClassifier": f"sklearn.tree",
            "RandomForestClassifier": f"sklearn.ensemble",
            "LogisticRegression" : f"sklearn.linear_model",
            "XGBClassifier" : f"xgboost",
            "LGBMClassifier" : f"lightgbm",
            "RandomForestRegressor": f"sklearn.ensemble",
            "XGBRegressor" : f"xgboost",
            "LGBMRegressor" : f"lightgbm"
        }
    
    
    def get_model(self, model_name):
        model_path = self.model_catalogue[model_name]
        return getattr(import_module(f"{model_path}"), f"{model_name}")