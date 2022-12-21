import streamlit as st
import streamlit.components.v1 as components
import json
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
import re 

with open('/great_expectation/path.json') as json_file:
    data_html = json.load(json_file)

with open('/eda_html/path.json') as json_file:
    data_html_1 = json.load(json_file)

#import data
f = open("/code/notebook/script/myFile1.txt", "r")
task = f.readlines(0)[0]
data = pd.read_csv("/ml_data/visual_eval.csv", index_col = "Unnamed: 0")
eval_train = pd.read_csv("/ml_data/evaluate_train.csv").drop(columns = "Unnamed: 0")
eval_test = pd.read_csv("/ml_data/evaluate_test.csv").drop(columns = "Unnamed: 0")

##import_data##
local = pd.read_csv("/model_interpret/local.csv").drop(columns = ["Unnamed: 0"])
global_ex = pd.read_csv("/model_interpret/global.csv").drop(columns = ["Unnamed: 0"])
feature_importance_plot= pd.DataFrame(global_ex.mean().sort_values()).reset_index().rename(columns={"index": "feature", 0: "lime_values"}).head(25)
    
st.set_page_config(page_title="My App waaa",layout='wide')
tab1, tab2, tab3, tab4 = st.tabs(["Data Quality", "EDA", "Model evaluate", "Model interpretability"])
with tab1:
    HtmlFile = open(f"{data_html['flask_path']}/{data_html['html_path']}", 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    components.html(source_code, width = 1846, height = 1053, scrolling = True)
with tab2:
    HtmlFile = open(f"{data_html_1['flask_path']}/{data_html_1['html_path']}", 'r', encoding='utf-8')
    source_code_1 = HtmlFile.read() 
    components.html(source_code_1, width = 1846, height = 1053, scrolling = True)
with tab3:
    if task == "regression":
        tab1, tab2= st.tabs(["model performance", "error distribution"])
        with tab1:
            st.header("model performance")
            test_plotly = data
            x = [index for index in test_plotly.index]
            plot = go.Figure(data=[go.Bar(
                name = 'Train',
                x = x,
                y = test_plotly["Train"]
               ), go.Bar(
                name = 'Test',
                x = x,
                y = test_plotly["Test"])]
                )

            plot.update_layout(
                autosize=False,
                width=1000,
                height=500,
                yaxis=dict(
                    title = "score",
                    ticktext=["Very long label", "long label", "3", "label"],
                    tickvals=[1, 2, 3, 4],
                    tickmode="array",
                    titlefont=dict(size=30),
                ),
               title ="train VS test"
            )
            st.plotly_chart(plot, use_container_width=True)

        with tab2:
            with st.container():
                st.header("error distribution in train")
                eval_train["diff"] = eval_train["y_true"] - eval_train["y_pred"]
                fig = px.histogram(eval_train, x = 'diff')
                st.plotly_chart(fig, use_container_width=True)
            st.header("error distribution in test")
            eval_test["diff"] = eval_test["y_true"] - eval_test["y_pred"]
            fig2 = px.histogram(eval_test, x = 'diff')
            st.plotly_chart(fig2, use_container_width=True)

    if task == "classification":
        tab1, tab2= st.tabs(["model performance", "confusion matrix"])
        with tab1:
            st.header("model performance")
            test_plotly = data
            x = [index for index in test_plotly.index]

            plot = go.Figure(data=[go.Bar(
                name = 'Train',
                x = x,
                y = test_plotly["f1_macro_train"]
               ), go.Bar(
                name = 'Test',
                x = x,
                y = test_plotly["f1_macro_test"])]
                )

            plot.update_layout(
                autosize=False,
                width=1000,
                height=500,
                xaxis = dict(
                    title = "target class",
                    tickmode = "array",
                    titlefont = dict(size=30),
                ),
                yaxis=dict(
                    title = "avg f1 score",
                    ticktext =["Very long label", "long label", "3", "label"],
                    tickvals = [1, 2, 3, 4],
                    tickmode = "array",
                    titlefont = dict(size=30),
                ),
               title="train vs test (avg f1 score)"
            )
            st.plotly_chart(plot, use_container_width=True)

        with tab2:
            col1, col2= st.columns(2)
            with col1:
                st.header("confusion_matrix (train set)")
                options = st.multiselect("select target:", eval_train["y_true"].unique(), default = eval_train["y_true"].unique())
                eval_train_act = eval_train[eval_train["y_true"].isin(options)]
                eval_train_pred = eval_train[eval_train["y_pred"].isin(options)]
                y_actu = pd.Series(eval_train_act["y_true"], name='Actual')
                y_pred = pd.Series(eval_train_pred["y_pred"], name='Predicted')
                df_confusion = pd.crosstab(y_actu, y_pred)
                fig = px.imshow(df_confusion, text_auto=True)
                fig.layout.height = 750
                fig.layout.width = 750
                st.plotly_chart(fig)
            with col2:
                st.header("confusion_matrix (test set)")
                options = st.multiselect("select target:", eval_test["y_true"].unique(), default = eval_test["y_true"].unique())
                eval_test_act = eval_test[eval_test["y_true"].isin(options)]
                eval_test_pred = eval_test[eval_test["y_pred"].isin(options)]
                y_actu_test = pd.Series(eval_test_act["y_true"], name='Actual')
                y_pred_test = pd.Series(eval_test_pred["y_pred"], name='Predicted')
                df_confusion_test = pd.crosstab(y_actu_test, y_pred_test)
                fig_ja = px.imshow(df_confusion_test, text_auto=True)
                fig_ja.layout.height = 750
                fig_ja.layout.width = 750
                st.plotly_chart(fig_ja)
with tab4:
    tab1, tab2, tab3 = st.tabs(["feature_importance", "dependency plot", "dataset_explorer"])

    with tab1:
        st.header("Aggreate feature importance")
        fig = px.bar(feature_importance_plot, x='lime_values', y='feature',hover_data=['lime_values'], color='lime_values', title = 'Aggregate Feature Importance')
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("Dependency plot")
        cols = tuple(feature_importance_plot.sort_values(by='lime_values', ascending=False)["feature"].values)
        option = st.selectbox('select columns',tuple(cols))
        x = option
        y = "importance_values_" + x
        fig = px.scatter(local, x = x, y= y )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.header("Dataset explorer")
        _tab1, _tab2= st.tabs(["individual feature", "chcek between feature"])
        with _tab1:
            st.header("individual feature")
            cols_ja = tuple(feature_importance_plot.sort_values(by='lime_values', ascending=False)["feature"].values)
            number = st.number_input('Insert a bins', value = 1)
            option_2 = st.selectbox('select columns',tuple(cols_ja), key = "2")
            cols = option_2
            cols_hist = cols + "_bin"
            num_bins = number
            local[cols_hist] = pd.qcut(local['pH'], q = num_bins ).astype('str')
            fig = px.box(local, x = cols_hist, y = cols, labels = {cols_hist:"number of bins"})
            st.plotly_chart(fig, use_container_width=True)
        with _tab2:
            st.header("chcek between feature")
            cols_ja_2 = tuple(feature_importance_plot.sort_values(by='lime_values', ascending=False)["feature"].values)
            option_3 = st.selectbox('select X',tuple(cols_ja_2), key = "3")
            option_4 = st.selectbox('select y',tuple(cols_ja_2), key = "4")
            x = option_3
            y = option_4
            fig = px.scatter(local, x = x, y= y )
            st.plotly_chart(fig, use_container_width=True)