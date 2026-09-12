import streamlit as st
import numpy as np
from datetime import date
from sentence_transformers import SentenceTransformer

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
    "regulation", "notification", "circular", "municipal",
    "passport", "cnic", "id card", "property", "land", "construction",
    "import", "export", "customs"
]

def is_government_related(text: str) -> bool:
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in GOVERNMENT_KEYWORDS)

def mentions_business(text: str) -> bool:
    business_words = ["business", "company", "firm", "startup", "start up", "construction"]
    text_lower = text.lower()
    return any(word in text_lower for word in business_words)

# ---------- Tiny Practice Knowledge Base ----------
KNOWLEDGE_BASE = [
    {
        "institution": "SECP (Securities and Exchange Commission of Pakistan)",
        "title": "Company Registration Overview",
        "url": "https://www.secp.gov.pk/",
        "text": (
            "To register a company in Pakistan, you must apply through SECP's "
            "e-Services portal. Required documents typically include CNIC "
            "copies of directors, a proposed company name, and a memorandum "
            "of association."
        ),
        "source_type": "Official Government Portal",
        "verification_status": "Verified / Current",
        "publication_date": "2023-01-15",
        "applies_to_structure": ["company"],
        "classification": "mandatory",
        "reason": "Registering as a company legally requires SECP incorporation before the business can operate.",
    },
    {
        "institution": "FBR (Federal Board of Revenue)",
        "title": "National Tax Number (NTN) Registration",
        "url": "https://www.fbr.gov.pk/",
        "text": (
            "Businesses operating in Pakistan must register for a National "
            "Tax Number (NTN) with FBR. This is required for filing income "
            "tax and is typically done online through the IRIS portal."
        ),
        "source_type": "Official Government Portal",
        "verification_status": "Verified / Current",
        "publication_date": "2023-03-10",
        "applies_to_structure": ["sole_proprietorship", "partnership", "company"],
        "classification": "mandatory",
        "reason": "All business structures must have an NTN to file taxes, regardless of size or type.",
    },
    {
        "institution": "Punjab Government - PBIT",
        "title": "Business Setup Guidance for Punjab",
        "url": "https://invest.punjab.gov.pk/",
        "text": (
            "Businesses setting up in Punjab, including construction-related "
            "businesses, may need approvals from local development "
            "authorities depending on the nature and location of the "
            "business activity."
        ),
        "source_type": "Official Government Portal",
        "verification_status": "Official but date unclear",
        "publication_date": "Unknown",
        "applies_to_structure": ["sole_proprietorship", "partnership", "company"],
        "classification": "conditional",
        "reason": "This only applies if your specific business activity or location requires local development authority approval.",
    },
    {
        "institution": "PEC (Pakistan Engineering Council)",
        "title": "Construction Firm Registration",
        "url": "https://www.pec.org.pk/",
        "text": (
            "Construction companies undertaking engineering works in "
            "Pakistan are generally required to register with the Pakistan "
            "Engineering Council (PEC) to be eligible for certain "
            "government and private contracts."
        ),
        "source_type": "Official Government Portal",
        "verification_status": "Verified / Current",
        "publication_date": "2022-11-05",
        "applies_to_structure": ["partnership", "company"],
        "classification": "conditional",
        "reason": "Required only if you plan to bid on government or PEC-regulated engineering contracts, not for all construction work.",
    },
    {
        "institution": "FBR (Federal Board of Revenue)",
        "title": "Registering as a Sole Proprietor",
        "url": "https://www.fbr.gov.pk/",
        "text": (
            "A sole proprietorship does not require SECP registration. "
            "The owner registers directly with FBR for an NTN under their "
            "own CNIC, and this is generally the simplest business structure "
            "to set up in Pakistan."
        ),
        "source_type": "Official Government Portal",
        "verification_status": "Verified / Current",
        "publication_date": "2023-02-01",
        "applies_to_structure": ["sole_proprietorship"],
        "classification": "mandatory",
        "reason": "As a sole proprietor, NTN registration under your own CNIC is the primary legal registration step.",
    },
    {
        "institution": "Punjab Chamber of Commerce",
        "title": "Chamber of Commerce Membership",
        "url": "https://example-lcci.pk/",
        "text": (
            "Businesses may optionally join their local Chamber of Commerce "
            "and Industry for networking, trade certificates, and business "
            "advocacy support. This is not a legal requirement to operate."
        ),
        "source_type": "Secondary / Industry Body",
        "verification_status": "Secondary",
        "publication_date": "Unknown",
        "applies_to_structure": ["sole_proprietorship", "partnership", "company"],
        "classification": "optional",
        "reason": "Chamber membership provides business benefits but is not legally required to operate.",
    },
]

# ---------- Load embedding model (cached so it only loads once) ----------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def build_embeddings(_model):
    texts = [doc["text"] for doc in KNOWLEDGE_BASE]
    return _model.encode(texts)

model = load_model()
doc_embeddings = build_embeddings(model)

def retrieve_relevant_docs(query: str, structure: str = None, top_k: int = 6):
    query_embedding = model.encode([query])[0]
    similarities = np.dot(doc_embeddings, query_embedding) / (
        np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    ranked_indices = np.argsort(similarities)[::-1]

    results = []
    for i in ranked_indices:
        doc = KNOWLEDGE_BASE[i]
        if structure and structure not in doc["applies_to_structure"]:
            continue
        results.append(doc)
        if len(results) >= top_k:
            break
    return results

def render_source_expander(doc):
    with st.expander(f"📄 {doc['title']} ({doc['institution']})"):
        st.write(f"**Institution:** {doc['institution']}")
        st.write(f"**Title:** {doc['title']}")
        st.write(f"**URL:** {doc['url']}")
        st.write(f"**Source Type:** {doc['source_type']}")
        st.write(f"**Publication Date:** {doc['publication_date']}")
        st.write(f"**Status:** {doc['verification_status']}")
        st.write(f"**Retrieved:** {date.today().isoformat()}")

def render_requirement(doc):
    st.markdown(f"**{doc['title']}** — *{doc['institution']}*")
    st.write(doc["text"])
    st.caption(f"Why this applies: {doc['reason']}")
    render_source_expander(doc)
    st.divider()

# ---------- Session State ----------
if "user_goal" not in st.session_state:
    st.session_state.user_goal = ""
if "business_structure" not in st.session_state:
    st.session_state.business_structure = None
if "submitted" not in st.session_state:
    st.session_state.submitted = False

# ---------- UI ----------
user_goal_input = st.text_area(
    "What do you want to accomplish?",
    placeholder="Example: I want to start a construction business in Lahore.",
    value=st.session_state.user_goal,
)

if st.button("Ask RAASTA AI"):
    st.session_state.user_goal = user_goal_input
    st.session_state.submitted = True
    st.session_state.business_structure = None

if st.session_state.submitted:
    goal = st.session_state.user_goal

    if not goal.strip():
        st.warning("Please tell us what you want to accomplish.")
    elif not is_government_related(goal):
        st.info(
            "RAASTA AI is designed to help with government procedures and "
            "services in Pakistan. Please ask about a government "
            "registration, license, permit, application, tax, service, "
            "or other government procedure."
        )
    else:
        st.success("This looks like a government-related request.")
        st.write("You asked:")
        st.write(goal)

        if mentions_business(goal):
            st.subheader("One quick question")
            st.write("What business structure are you planning to use?")

            with st.expander("Why are you asking me this?"):
                st.write(
                    "Your business structure changes which registrations "
                    "are required. For example, a sole proprietorship does "
                    "not require SECP registration, but a company does."
                )

            structure_choice = st.radio(
                "Choose one:",
                options=["Sole Proprietorship", "Partnership", "Company"],
                index=None,
                key="structure_radio",
            )

            structure_map = {
                "Sole Proprietorship": "sole_proprietorship",
                "Partnership": "partnership",
                "Company": "company",
            }

            if structure_choice:
                st.session_state.business_structure = structure_map[structure_choice]

        structure = st.session_state.business_structure

        if mentions_business(goal) and not structure:
            st.info("Please select a business structure above to see personalized results.")
        else:
            results = retrieve_relevant_docs(goal, structure=structure)

            mandatory = [d for d in results if d["classification"] == "mandatory"]
            conditional = [d for d in results if d["classification"] == "conditional"]
            optional = [d for d in results if d["classification"] == "optional"]

            if structure:
                st.caption(f"Personalized for: {structure.replace('_', ' ').title()}")

            st.subheader("✅ Mandatory Requirements")
            if mandatory:
                for doc in mandatory:
                    render_requirement(doc)
            else:
                st.write("No mandatory requirements found for this query.")

            st.subheader("⚠️ Conditional Requirements")
            if conditional:
                for doc in conditional:
                    render_requirement(doc)
            else:
                st.write("No conditional requirements found for this query.")

            st.subheader("ℹ️ Optional")
            if optional:
                for doc in optional:
                    render_requirement(doc)
            else:
                st.write("No optional items found for this query.")
