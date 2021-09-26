"""
Python module for the webapp.

Author: Kkyn
"""
import streamlit as st

from home_page import home_page
from prediction_page import  prediction_page

st.set_page_config('Cancer Dectection Application', 
    page_icon='random', initial_sidebar_state='expanded', layout='wide')

def go_to_page(choice):
    if choice == 'Home':
        home_page()
    elif choice == 'Prediction':
        prediction_page()

if __name__ == '__main__':
    main_menu = ['Home', 'Prediction']
    user_choice = st.sidebar.radio("Menu", main_menu)
    go_to_page(user_choice)