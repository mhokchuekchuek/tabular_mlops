import streamlit as st
import pandas as pd
import plotly.express as px
import re 

##import_data##
local = pd.read_csv("/model_interpret/local.csv").drop(columns = ["Unnamed: 0"])
global_ex = pd.read_csv("/model_interpret/global.csv").drop(columns = ["Unnamed: 0"])
feature_importance_plot= pd.DataFrame(global_ex.mean().sort_values()).reset_index().rename(columns={"index": "feature", 0: "shap_values"})

st.set_page_config(page_title="My App 2", page_icon="", layout="wide")
st.title('Model interpretability')
tab1, tab2, tab3 = st.tabs(["feature_importance", "dependency plot", "dataset_explorer"])

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
    _tab1, _tab2= st.tabs(["individual feature", "chcek between feature"])
    with _tab1:
        st.header("individual feature")
        cols_ja = tuple(feature_importance_plot.sort_values(by='shap_values', ascending=False)["feature"].values)
        number = st.number_input('Insert a bins', value = 1)
        option_2 = st.selectbox('select columns',tuple(cols_ja), key = "2")
        cols = option_2
        cols_hist = cols + "_bin"
        num_bins = number
        local[cols_hist] = pd.qcut(local[cols], q = num_bins ).astype('str')
        fig = px.box(local, x = cols_hist, y = cols, labels = {cols_hist:"number of bins"})
        st.plotly_chart(fig, use_container_width=True)
    with _tab2:
        st.header("chcek between feature")
        cols_ja_2 = tuple(feature_importance_plot.sort_values(by='shap_values', ascending=False)["feature"].values)
        option_3 = st.selectbox('select X',tuple(cols_ja_2), key = "3")
        option_4 = st.selectbox('select y',tuple(cols_ja_2), key = "4")
        x = option_3
        y = option_4
        fig = px.scatter(local, x = x, y= y, color = x)
        st.plotly_chart(fig, use_container_width=True)