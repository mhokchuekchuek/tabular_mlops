import streamlit as st
import streamlit.components.v1 as components
import json

with open('/eda_html/path.json') as json_file:
    data_html_1 = json.load(json_file)
    
st.set_page_config(page_title="EDA",layout='wide')
HtmlFile = open(f"{data_html_1['flask_path']}/{data_html_1['html_path']}", 'r', encoding='utf-8')
source_code_1 = HtmlFile.read() 
components.html(source_code_1, width = 1846, height = 1053, scrolling = True)