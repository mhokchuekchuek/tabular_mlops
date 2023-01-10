import streamlit as st
import pandas as pd
import plotly.express as px
import re 

##import_data##
local = pd.read_csv("/model_interpret/local.csv").drop(columns = ["Unnamed: 0"])
global_ex = pd.read_csv("/model_interpret/global.csv").drop(columns = ["Unnamed: 0"])
feature_importance_plot= pd.DataFrame(global_ex.mean().sort_values()).reset_index().rename(columns={"index": "feature", 0: "shap_values"})

st.set_page_config(page_title="", page_icon="", layout="wide")
st.title('Model interpretability')
tab1, tab2, tab3 = st.tabs(["feature_importance", "dependency plot", "Dataset Explorer"])

with tab1:
    st.header("Aggreate feature importance")
    feature_importance_plot_sort = pd.DataFrame(abs(global_ex.mean()).sort_values(ascending = False)).reset_index().rename(columns={"index": "feature", 0: "shap_values"})["feature"].values[:15]
    plot_wa = feature_importance_plot[feature_importance_plot["feature"].isin(list(feature_importance_plot_sort))]
    fig = px.bar(plot_wa, x='shap_values', y='feature',hover_data=['shap_values'], color='shap_values', title = 'Aggregate Feature Importance')
    st.plotly_chart(fig, use_container_width=True)
    
with tab2:
    st.header("Dependency plot")
    cols = tuple(feature_importance_plot.sort_values(by='shap_values', ascending=False)["feature"].values)
    option = st.selectbox('select columns',tuple(cols))
    x = option
    y = "importance_values_" + x
    fig = px.scatter(local, x = x, y= y)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Dataset explorer")
    cols_ja_2 = tuple(feature_importance_plot.sort_values(by='shap_values', ascending=False)["feature"].values)
    option_3 = st.selectbox('select X',tuple(cols_ja_2), key = "3")
    option_4 = st.selectbox('select y',tuple(cols_ja_2), key = "4")
    x = option_3
    y = option_4
    plot_ = st.radio(
        "choose your type of graph",
        ("box-plot", "scatter"), key = "7")
    if plot_ == "box-plot":
        fig = px.box(local, x = x, y= y)
    else:
        fig = px.scatter(local, x = x, y= y)
    st.plotly_chart(fig, use_container_width=True)