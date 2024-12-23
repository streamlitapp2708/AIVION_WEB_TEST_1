import streamlit as st





# --- HERO SECTION ---
col1, col2 = st.columns(2, gap="small", vertical_alignment="top")
with col2:
    st.image("./assets/AIVION.jpg", width=230)

with col1:
    st.title("Our Services", anchor=False)
    # st.write(
    #     "Our Services"
    # )
    # if st.button("✉️ Contact Me"):
    #     show_contact_form()


# --- EXPERIENCE & QUALIFICATIONS ---
st.write("\n")
st.subheader("Data Engineering", anchor=False)
st.write(
    """
Build a robust foundation for 
your data strategy with 
scalable, high-performance 
infrastructure. Our data 
engineering services go 
beyond data architecture and 
data integration to provide 
data management, data 
governance and self-service 
access to the data for all 
business users..    """
)

# --- SKILLS ---
st.write("\n")
st.subheader("Data Science", anchor=False)
st.write(
    """
Unlock the power of your 
data through advanced 
analytics and predictive 
modelling. Our data science 
solutions transform raw data 
into actionable insights, 
empowering businesses to 
make data-driven decisions, 
optimize operations, and 
uncover new growth 
opportunities.    """
)

# --- SKILLS ---
st.write("\n")
st.subheader("Generative AI", anchor=False)
st.write(
    """
We enable businesses to 
streamline content 
creation, enhance customer 
experiences, and develop 
AI-powered solutions. 
Harness the potential of 
cutting-edge Generative AI 
to create innovative 
solutions to create value 
for your organisation. """
)

# --- SKILLS ---
st.write("\n")
st.subheader("Automation", anchor=False)
st.write(
    """
Simplify complex workflows 
and drive operational 
efficiency through intelligent 
automation. We implement 
AI-driven automation 
solutions that reduce manual 
efforts, improve accuracy, 
and accelerate business 
processes, helping you stay 
ahead in today’s fast-paced 
environment. """
)