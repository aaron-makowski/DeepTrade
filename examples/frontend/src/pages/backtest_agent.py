import streamlit as st
from src.pages.training_functions import *

def write():
    """Method used to write page in app.py"""

    st.header('Agent Testing')
    
    st.set_option('deprecation.showfileUploaderEncoding', False)
    #import io
    # file_buffer = st.file_uploader(...)
    # text_io = io.TextIOWrapper(file_buffer)
    
    uploaded_file = st.file_uploader("Choose a DeepTrade Agent file", type="pb")
    if uploaded_file is not None:
        print("uploaded_file:", uploaded_file)

    #if AGENT := return_cached_agent_model():
        #AGENT.

    load_button = st.button("Load File", key="load_data")
    if load_button is not None:
        dataset = load_agent(uploaded_file)
         
    
    #loading data
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        print("uploaded_file:", uploaded_file)
        dataset = load_csv(uploaded_file)
     

    backtest_button = st.button("Start Agent Test", key="backtest_data")
    if backtest_button is not None:
        pass