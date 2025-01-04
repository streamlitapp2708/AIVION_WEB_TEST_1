import streamlit as st

def initialize_session_state():
    """
    Initialize session state variables for file uploads, data, and chat history.
    """
    # Initialize session state variables for uploaded files and their names
    if "uploaded_files" not in st.session_state:
        st.session_state["uploaded_files"] = []

    if "file_names" not in st.session_state:
        st.session_state["file_names"] = []

    # Initialize 'combined_df' to store the final database
    if "combined_df" not in st.session_state:
        st.session_state["combined_df"] = None

    # Initialize 'chat_history' to store the conversation between the user and the assistant
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Initialize 'question_history' to store the input questions asked by the user
    if "question_history" not in st.session_state:
        st.session_state["question_history"] = []

    # Initialize 'last_input' to store the last entered value in the input box
    if "last_input" not in st.session_state:
        st.session_state["last_input"] = ""
        