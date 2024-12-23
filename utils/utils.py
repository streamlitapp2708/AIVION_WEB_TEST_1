import streamlit as st

def initialize_session_state():
    """
    Initialize session state variables for file uploads and data.
    """
    # Check if the session state variable 'uploaded_files' exists
    if "uploaded_files" not in st.session_state:
        # Initialize 'uploaded_files' as an empty list to store DataFrames
        st.session_state["uploaded_files"] = []
    
    # Check if the session state variable 'file_names' exists
    if "file_names" not in st.session_state:
        # Initialize 'file_names' as an empty list to store uploaded file names
        st.session_state["file_names"] = []
