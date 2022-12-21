import streamlit as st
import streamlit.components.v1 as components
import json

with open('/great_expectation/path.json') as json_file:
    data = json.load(json_file)

with open('/eda_html/path.json') as json_file:
    data1 = json.load(json_file)
    
st.set_page_config(page_title="My App waaa",layout='wide')
st.header
tab1, tab2, tab3, tab4 = st.tabs(["Data Quality", "EDA"])
with tab1:
    HtmlFile = open(f"{data['flask_path']}/{data['html_path']}", 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    components.html(source_code, width = 2250, height = 1444, scrolling = True)
with tab2:
    HtmlFile = open(f"{data1['flask_path']}/{data1['html_path']}", 'r', encoding='utf-8')
    source_code_1 = HtmlFile.read() 
    components.html(source_code_1, width = 2250, height = 1444, scrolling = True)