import homepage
import app2
import newpage
import streamlit as st
def calling():

    PAGES = {
        "Home": homepage,
        "Metrics": newpage,
        "Plots": app2
    }
    st.sidebar.title('Navigation')
    selection = st.sidebar.selectbox("Go to", list(PAGES.keys()))
    page = PAGES[selection]
    page.app()