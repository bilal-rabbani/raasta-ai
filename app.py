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

# ---------- Scope Detector ----------
GOVERNMENT_KEYWORDS = [
    "register", "registration", "license", "licence", "permit", "tax",
    "fbr", "secp", "nadra", "business", "company", "firm", "authority",
    "government", "govt", "application", "apply", "certificate", "fee",
    "documents", "requirement", "department", "ministry", "form",
    "regulation", "notification", "circular", "municipal", "authority",
    "passport", "cnic", "id card", "property", "land", "construction",
    "import", "export", "customs"
]

def is_government_related(text: str) -> bool:
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in GOVERNMENT_KEYWORDS)

# ---------- UI ----------
user_goal = st.text_area(
    "What do you want to accomplish?",
    placeholder="Example: I want to start a construction business in Lahore."
)

if st.button("Ask RAASTA AI"):
    if not user_goal.strip():
        st.warning("Please tell us what you want to accomplish.")
    elif not is_government_related(user_goal):
        st.info(
            "RAASTA AI is designed to help with government procedures and "
            "services in Pakistan. Please ask about a government "
            "registration, license, permit, application, tax, service, "
            "or other government procedure."
        )
    else:
        st.success("This looks like a government-related request.")
        st.write("You asked:")
        st.write(user_goal)
