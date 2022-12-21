#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install dash')


# In[2]:


from interpret.ext.blackbox import *


# In[3]:


import pandas as pd
import joblib
X_train = pd.read_csv("/ml_data/X_train.csv").drop(columns = "Unnamed: 0")
y_train = pd.read_csv("/ml_data/y_train.csv").drop(columns = "Unnamed: 0")
X_test = pd.read_csv("/ml_data/X_test.csv").drop(columns = "Unnamed: 0")
y_test = pd.read_csv("/ml_data/y_test.csv").drop(columns = "Unnamed: 0")
model = joblib.load("/artifact/mlruns/941649382202349625/a8b85d3a1dc54222970863c0c4298e58/artifacts/model/model.pkl")


# In[4]:


explainer = PFIExplainer(model,
                         features = X_train.columns, 
                         classes = y_test["quality"].unique())


# In[5]:


explainer = TabularExplainer(model, 
                             X_train, 
                             features = X_train.columns, 
                             classes = y_test["quality"].unique())

# explain overall model predictions (global explanation)
global_explanation = explainer.explain_global(X_test)


# In[6]:


from raiwidgets import ExplanationDashboard
ExplanationDashboard(global_explanation, model, dataset=X_test, true_y=y_test,port = 5599, public_ip="0.0.0.0")


# In[7]:


import pandas as pd
import joblib
X_train = pd.read_csv("/ml_data/X_train.csv").drop(columns = "Unnamed: 0")
y_train = pd.read_csv("/ml_data/y_train.csv").drop(columns = "Unnamed: 0")
X_test = pd.read_csv("/ml_data/X_test.csv").drop(columns = "Unnamed: 0")
y_test = pd.read_csv("/ml_data/y_test.csv").drop(columns = "Unnamed: 0")
model = joblib.load("/artifact/mlruns/661781806827710659/572147d429ba41aea8b635c3b57042aa/artifacts/model/model.pkl")


# In[8]:


explainer = PFIExplainer(model,
                         features = X_train.columns, 
                         classes = y_test["quality"].unique())


# In[9]:


from raiwidgets import ExplanationDashboard
ExplanationDashboard(global_explanation, model, dataset=X_test, true_y=y_test,port = 5599, public_ip="0.0.0.0")


# In[ ]:




