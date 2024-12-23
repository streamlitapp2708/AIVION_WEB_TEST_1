import streamlit as st





# --- HERO SECTION ---
col1, col2 = st.columns(2, gap="small", vertical_alignment="top")
with col2:
    st.image("./assets/AIVION.jpg", width=230)

# with col1:
#     st.title("", anchor=False)
#     st.write(
#         "Transformative. AI. Solutions"
#     )
    # if st.button("✉️ Contact Me"):
    #     show_contact_form()


# --- EXPERIENCE & QUALIFICATIONS ---
st.write("\n")
st.subheader("Introduction", anchor=False)
st.write(
    """
Welcome to M2 AIVION. We are on a mission to democratize AI 
technology, making it accessible and affordable for businesses that aim 
to thrive in a data-driven world. In today’s competitive landscape, 
businesses shouldn’t have to invest millions to unlock the transformative 
power of AI and Data Analytics. At M2 AIVION, we bring cutting-edge AI 
capabilities, proven expertise in data science, and deep industry insights 
to empower organizations of all sizes. We bridge the gap between 
ambition and execution by providing scalable, cost-effective solutions 
that deliver real, measurable outcomes. Our consultancy blends 
expertise in data science, artificial intelligence, and industry-specific 
insights to empower organizations in their digital journey.    """
)

# --- SKILLS ---
st.write("\n")
st.subheader("Our Vision", anchor=False)
st.write(
    """
To be a trusted partner in data and AI transformation, helping 
organizations harness the power of Data & AI innovation. Our vision is to 
simplify AI adoption, solve complex business challenges, and empower 
organizations to innovate faster, operate smarter, and achieve 
sustainable growth.
 At M2 AIVION, we believe that AI isn’t just for the privileged few; it’s for 
everyone ready to embrace the future.    """
)
