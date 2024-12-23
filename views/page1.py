import streamlit as st
from utils.utils import initialize_session_state

# Initialize session state variables
initialize_session_state()

# Page title
st.title("Page 1: Enter User Details")

# Collect user input
st.session_state["user_name"] = st.text_input("Enter your name:", value=st.session_state["user_name"])
st.session_state["user_age"] = st.text_input("Enter your age:", value=st.session_state["user_age"])

# Show entered details
st.write("Name:", st.session_state["user_name"])
st.write("Age:", st.session_state["user_age"])
