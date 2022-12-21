import os
import streamlit as st
option = st.selectbox(
    'what visualization that you want to see?',
    ('model evaluation', 'model interpretability'))

if option == "model evaluation":
    st.write('the visualization is on http://172.19.0.5:1234/')
    os.system("cd /code/notebook/script ; streamlit run run_streamlit_evaluate.py --server.port 1234")

if option == 'model interpretability':
    st.write('the visualization is on http://172.19.0.5:1235')
    os.system("cd /code/notebook/script ; streamlit run run_streamlit_interpret.py --server.port 1235")