import streamlit as st
from utils.utils import initialize_session_state

# Initialize session state variables
initialize_session_state()

# Page title
st.title("Page 2: Display User Details")

# Access and display the session state variables
if st.session_state["user_name"] and st.session_state["user_age"]:
    st.write(f"Hello, {st.session_state['user_name']}! You are {st.session_state['user_age']} years old.")
else:
    st.warning("No user details found. Please go to Page 1 to enter your details.")
