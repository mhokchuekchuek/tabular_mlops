import streamlit as st
import streamlit.components.v1 as components
import json

with open('/great_expectation/path.json') as json_file:
    data_html = json.load(json_file)
    
st.set_page_config(page_title="Data quality checking",layout='wide')
HtmlFile = open(f"{data_html['flask_path']}/{data_html['html_path']}", 'r', encoding='utf-8')
source_code = HtmlFile.read() 
components.html(source_code, width = 1846, height = 1053, scrolling = True)