import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
import streamlit as st
import sys
sys.path.append("/code/notebook")
from script.utils import *
#import data
f = open("/code/notebook/script/task.txt", "r")
task = f.readlines(-1)[0]
data = pd.read_csv("/ml_data/visual_eval.csv", index_col = "Unnamed: 0")
eval_train = pd.read_csv("/ml_data/evaluate_train.csv").drop(columns = "Unnamed: 0")
eval_test = pd.read_csv("/ml_data/evaluate_test.csv").drop(columns = "Unnamed: 0")

# Set the layout of the web app to "wide"
st.set_page_config(page_title="Evaluate", layout="wide")
st.title('Model evaluate')
if task == "regression":
    tab1, tab2, tab3= st.tabs(["model performance", "error distribution", "dataset explorer"])
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
            st.header("error distribution in test (difference from y_pred and y_true)")
            eval_test["diff"] = eval_test["y_true"] - eval_test["y_pred"]
            fig2 = px.histogram(eval_test, x = 'diff')
            st.plotly_chart(fig2, use_container_width=True)
        st.header("Select a range of difference")
        number = st.number_input('Insert upper bond', max_value = max(eval_test["diff"]))
        number_2 = st.number_input('Insert lower bond', min_value = min(eval_test["diff"]))
        df_ja = eval_test[(eval_test["diff"]>= number_2) & (eval_test["diff"] < number)]
        st.dataframe(df_ja)
        
        @st.cache
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_ja)

        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='diff_data.csv',
            mime='text/csv',
        )
        
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

with tab3:
    numeric_cols, cat_cols = check_type(eval_train)
    _tab1, _tab2, = st.tabs(["numerical data", "Catagorical data"])
    with _tab1:
        st.header("numerical data")
        cols_ja_2 = [i for i in numeric_cols]
        cols_select = ["y_true","y_pred", "diff"]
        local = eval_test
        option_3 = st.selectbox('select X',tuple(cols_ja_2), key = "3")
        option_4 = st.selectbox('select y',tuple(cols_select), key = "4")
        x = option_3
        y = option_4
        col1_3, col2_3 = st.columns(2)
        with col1_3:
            plot_ = st.radio(
                "choose your type of graph",
                ("box-plot", "scatter"))
        with col2_3:
            cols_ = st.radio(
                "choose your color",
                (x, y , "diff"))
        if plot_ == "box-plot":
            fig = px.box(local, x = x, y= y , color = cols_)
        else:
            fig = px.scatter(local, x = x, y= y , color = cols_, color_continuous_scale='Inferno_r')
        st.plotly_chart(fig, use_container_width=True)

    with _tab2:
        st.header("Catagorical data")
        cols_ja_2 = [i for i in cat_cols]
        cols_select = ["y_true","y_pred", "diff"]
        local = eval_test
        option_3 = st.selectbox('select X',tuple(cols_ja_2), key = "5")
        option_4 = st.selectbox('select y',tuple(cols_select), key = "6")
        x = option_3
        y = option_4
        col1_3, col2_3 = st.columns(2)
        with col1_3:
            plot_ = st.radio(
                "choose your type of graph",
                ("box-plot", "scatter"), key = "7")
        with col2_3:
            cols_ = st.radio(
                "choose your color",
                (x, y , "diff"), key = "8")
        if plot_ == "box-plot":
            fig = px.box(local, x = x, y= y , color = cols_)
        else:
            fig = px.scatter(local, x = x, y= y , color = cols_, color_continuous_scale='Inferno_r')
        st.plotly_chart(fig, use_container_width=True)

# st.header("individual feature")
# cols_ja = tuple(numeric_cols)
# local = eval_test
# number = st.number_input('Insert a bins', value = 1)
# option_2 = st.selectbox('select columns',tuple(cols_ja), key = "2")
# cols = option_2
# cols_hist = cols + "_bin"
# num_bins = number
# local[cols_hist] = pd.qcut(local[cols], q = num_bins ).astype('str')
# fig = px.box(local, x = cols_hist, y = cols, labels = {cols_hist:"number of bins"})
# st.plotly_chart(fig, use_container_width=True)