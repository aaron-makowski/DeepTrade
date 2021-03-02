"""Main module for the streamlit app"""
import streamlit as st
import src.st_extensions
 
# Pages
import src.pages.train_agent
import src.pages.live_trading

PAGES = {
    "Live AI Trading": src.pages.live_trading,
    "AI Training": src.pages.train_agent,
}

selection = st.sidebar.radio("Go to", list(PAGES.keys()))

page = PAGES[selection]
src.st_extensions.write_page(page)