import streamlit as st
from utils.utils import initialize_session_state

# Initialize session state variables
initialize_session_state()

# Sidebar navigation
# page = st.sidebar.radio("Navigation", ["Home", "Database"])

# if page == "Home":
#     st.title("Home")
#     st.write("Welcome to the Home page!")

    # # Example file upload
    # uploaded_files = st.file_uploader("Upload Files", accept_multiple_files=True)
    # if uploaded_files:
    #     st.session_state["uploaded_files"] = uploaded_files
    #     st.session_state["file_names"] = [file.name for file in uploaded_files]
    #     # Example: Combine file content into combined_df (mock logic here)
    #     st.session_state["combined_df"] = f"Mock combined data for: {', '.join(st.session_state['file_names'])}"

# elif page == "Database":
st.title("Database")

# # Display list of uploaded files
# if st.session_state["file_names"]:
#     st.subheader("List of Uploaded Files")
#     for file_name in st.session_state["file_names"]:
#         st.write(file_name)


# Display content of combined_df
if st.session_state["combined_df"] is not None:
    st.write(st.session_state["combined_df"])
else:
    st.write("No processed data available.")
