import streamlit as st
import pandas as pd
from utils.utils import initialize_session_state

# Initialize session state
initialize_session_state()

# Page title
st.title("Upload Files")

# Radio button to trigger file upload
if st.radio("Upload files", ["No", "Yes"]) == "Yes":
    uploaded_files = st.file_uploader(
        "Upload your CSV files",
        type=["csv"],
        accept_multiple_files=True,
        key="file_uploader",
    )

    # Process uploaded files
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Read each file into a DataFrame
            df = pd.read_csv(uploaded_file)
            # Store the DataFrame and file name in session state
            st.session_state["uploaded_files"].append(df)
            st.session_state["file_names"].append(uploaded_file.name)

        st.success("Files uploaded successfully!")
