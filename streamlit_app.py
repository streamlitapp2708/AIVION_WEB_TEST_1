import streamlit as st

# Load custom CSS
def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style/style.css")

# --- PAGE SETUP ---
about_page = st.Page(
    "views/about_me.py",
    title="About Us",
    icon=":material/account_circle:",
    default=True,
)
project_1_page = st.Page(
    "views/our_services.py",
    title="Our Services",
    icon=":material/bar_chart:",
)
project_2_page = st.Page(
    "views/Page_UploadFiles.py",
    title="Upload Files",
    icon=":material/smart_toy:",
)
project_3_page = st.Page(
    "views/Page_Database.py",
    title="Database",
    icon=":material/smart_toy:",
)
project_4_page = st.Page(
    "views/Page_TalkToData.py",
    title="Talk To Data",
    icon=":material/smart_toy:",
)



# --- NAVIGATION SETUP [WITHOUT SECTIONS] ---
# pg = st.navigation(pages=[about_page, project_1_page, project_2_page])

# --- NAVIGATION SETUP [WITH SECTIONS]---
pg = st.navigation(
    {
        "Info": [about_page],
        "Services ": [project_1_page],
        "Database Upload": [project_2_page],
        "Database": [project_3_page],
        "Talk to Data": [project_4_page],
    }
)


# --- SHARED ON ALL PAGES ---
st.logo("assets/AIVION.jpg")


# --- RUN NAVIGATION ---
pg.run()
