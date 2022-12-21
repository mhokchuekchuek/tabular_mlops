import pandas as pd
import sys
sys.path.append('/code/notebook')
from script.eda.eda_ja import eda

df = pd.read_csv("/ml_data/ddproperty_final.csv").drop(columns = "Unnamed: 0")
input_text = input('input your eda name:')
eda(df, input_text, overwrite = True)
