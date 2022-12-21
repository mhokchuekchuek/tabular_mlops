import pandas as pd
import sys
sys.path.append('/code/notebook')
from script.great_expectation_function.validate import path_to_json

df = pd.read_csv("/ml_data/ddproperty_final.csv").drop(columns = "Unnamed: 0")
input_text = input('input your suit name:')
path_to_json(df, input_text)


