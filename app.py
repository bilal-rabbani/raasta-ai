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

# ---------- Tiny Practice Knowledge Base ----------
# SMALL SAMPLE to prove the pipeline works.
# Real government sources will be added in a later part.
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

def retrieve_relevant_docs(query: str, top_k: int = 2):
    query_embedding = model.encode([query])[0]
    similarities = np.dot(doc_embeddings, query_embedding) / (
        np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [KNOWLEDGE_BASE[i] for i in top_indices]

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

        results = retrieve_relevant_docs(user_goal)

        st.subheader("Relevant Information Found")
        for doc in results:
            st.markdown(f"**{doc['title']}** — *{doc['institution']}*")
            st.write(doc["text"])
            st.divider()

        st.subheader("Sources")
        for doc in results:
            with st.expander(f"📄 {doc['title']} ({doc['institution']})"):
                st.write(f"**Institution:** {doc['institution']}")
                st.write(f"**Title:** {doc['title']}")
                st.write(f"**URL:** {doc['url']}")
                st.write(f"**Source Type:** {doc['source_type']}")
                st.write(f"**Publication Date:** {doc['publication_date']}")
                st.write(f"**Status:** {doc['verification_status']}")
                st.write(f"**Retrieved:** {date.today().isoformat()}")
