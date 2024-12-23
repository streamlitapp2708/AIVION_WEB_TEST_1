import streamlit as st
from utils.utils import initialize_session_state

# Initialize session state
initialize_session_state()

# Page title
st.title("Database")

# Display uploaded file names and content
if st.session_state["file_names"]:
    st.write("Uploaded Files:")
    for i, file_name in enumerate(st.session_state["file_names"]):
        st.subheader(f"File {i+1}: {file_name}")
        st.dataframe(st.session_state["uploaded_files"][i])  # Display the DataFrame
else:
    st.warning("No files uploaded yet. Go to 'Upload Files' to upload files.")
