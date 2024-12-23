import streamlit as st
from utils.utils import initialize_session_state

# Initialize session state
initialize_session_state()

# Page title
st.title("Talk to Data")

# Check if there are uploaded files
if st.session_state["file_names"]:
    # Input a question
    question = st.text_input("Ask a question about the data:")

    if question:
        st.write("Uploaded files and their content:")
        for i, (file_name, df) in enumerate(
            zip(st.session_state["file_names"], st.session_state["uploaded_files"])
        ):
            st.subheader(f"File {i+1}: {file_name}")
            st.dataframe(df)  # Display the DataFrame
else:
    st.warning("No files uploaded yet. Go to 'Upload Files' to upload files.")
