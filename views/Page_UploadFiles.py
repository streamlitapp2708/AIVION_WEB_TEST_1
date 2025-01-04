import streamlit as st
import pandas as pd
import PyPDF2
import numpy as np
from openai import AzureOpenAI


# Variable assignment from config file
azure_openai_endpoint = st.secrets["openai_api_base"]  # Azure OpenAI endpoint
azure_openai_key = st.secrets["openai_api_key"]        # Azure OpenAI key
azure_openai_version = st.secrets["openai_api_version"]  # Azure API version

azure_embedding_model_name = 'text-embedding-3-large'


client = AzureOpenAI(
  azure_endpoint = azure_openai_endpoint , 
  api_key= azure_openai_key,  
  api_version="2024-02-01"
)
 

# #Creating Embedding Function to vectorize the Querry
# def generate_embeddings(text): # model = "deployment_name"
#     return client.embeddings.create(input = [text], model=azure_embedding_model_name).data[0].embedding
 
def normalize_l2(x):
    x = np.array(x)
    if x.ndim == 1:
        norm = np.linalg.norm(x)
        if norm == 0:
            return x
        return x / norm
    else:
        norm = np.linalg.norm(x, 2, axis=1, keepdims=True)
        return np.where(norm == 0, x, x / norm)
 
#Creating Embedding Function to vectorize the Querry
def generate_embeddings(text): # model = "deployment_name"
    response = client.embeddings.create(input = [text], model=azure_embedding_model_name,encoding_format="float")
    cut_dim = response.data[0].embedding[:1536]
    norm_dim = normalize_l2(cut_dim)
    embeddings = list(norm_dim)
    return embeddings


# Function to initialize session state variables
def initialize_session_state():
    """
    Initialize session state variables for file uploads, data, and chat history.
    """
    if "combined_df" not in st.session_state:
        st.session_state["combined_df"] = pd.DataFrame()  # Initialize as an empty dataframe

# Page layout
st.title("Knowledge Repository")

# Initialize session state if it's not already initialized
initialize_session_state()

# Option to upload either PDF or TXT files
file_type = st.radio("Select File Type", ("PDF", "TXT"))

uploaded_files = st.file_uploader("Upload your file(s)", type=["pdf", "txt"], accept_multiple_files=True)

if uploaded_files:
    # Prepare a temporary dataframe to store the data for current files
    temp_data = []

    # Processing each uploaded file
    for uploaded_file in uploaded_files:
        # if file_type == "PDF":
        #     # Process PDF file
        #     pdf_reader = PyPDF2.PdfReader(uploaded_file)
        #     for page_num, page in enumerate(pdf_reader.pages):
        #         text = page.extract_text()
        #         if text:
        #             embeddings = generate_embeddings(text)
        #             temp_data.append({"file_name": uploaded_file.name, "page_num": page_num + 1, "text": text, "embedding": embeddings})

        if file_type == "PDF":
            try:
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    # st.write (f"text is {text}")
                    if text:
                        embeddings = generate_embeddings(text)
                        temp_data.append({"file_name": uploaded_file.name, "page_num": page_num + 1, "text": text, "embedding": embeddings})
            except PyPDF2.errors.PdfReadError:
                st.error(f"The file '{uploaded_file.name}' is not a valid PDF or is corrupt.")

        elif file_type == "TXT":
            # Process TXT file
            # text = uploaded_file.read().decode("utf-8")
            text = uploaded_file.read().decode("utf-8")

            embeddings = generate_embeddings(text)
            temp_data.append({"file_name": uploaded_file.name, "page_num": 1, "text": text, "embedding": embeddings})

    # Convert temporary data into a pandas dataframe
    temp_df = pd.DataFrame(temp_data)

    # Concatenate the temporary dataframe with the existing 'combined_df' in session state
    st.session_state["combined_df"] = pd.concat([st.session_state["combined_df"], temp_df], ignore_index=True)

    # Display the updated dataframe
    st.write("Processed Data", st.session_state["combined_df"])
