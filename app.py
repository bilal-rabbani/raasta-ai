import streamlit as st

st.set_page_config(
    page_title="RAASTA AI",
    page_icon="🛣️",
    layout="centered"
)

st.title("RAASTA AI")

st.write(
    "Tell RAASTA AI what government-related task you want to accomplish "
    "in Pakistan."
)

user_goal = st.text_area(
    "What do you want to accomplish?",
    placeholder="Example: I want to start a construction business in Lahore."
)

if st.button("Ask RAASTA AI"):
    if user_goal.strip():
        st.write("You asked:")
        st.write(user_goal)
    else:
        st.warning("Please tell us what you want to accomplish.")
