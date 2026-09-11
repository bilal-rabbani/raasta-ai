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

# ============================================================
# SCOPE DETECTOR
# ============================================================
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

# ============================================================
# TINY PRACTICE KNOWLEDGE BASE
# ------------------------------------------------------------
# Each doc now has an "id" and a "depends_on" list so the
# Roadmap Agent can build a real ordering, not just
# mandatory-before-conditional.
# ============================================================
KNOWLEDGE_BASE = [
    {
        "id": "secp_company_reg",
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
        "depends_on": [],  # first step for companies
    },
    {
        "id": "fbr_ntn",
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
        # a company must be incorporated before it can get its own NTN
        "depends_on": ["secp_company_reg"],
    },
    {
        "id": "fbr_sole_prop",
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
        "depends_on": [],
    },
    {
        "id": "punjab_local_approval",
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
        # needs the business to have its tax registration sorted first
        "depends_on": ["fbr_ntn", "fbr_sole_prop"],
    },
    {
        "id": "pec_construction_reg",
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
        "depends_on": ["fbr_ntn"],
    },
    {
        "id": "chamber_membership",
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
        "depends_on": [],
    },
]

KB_BY_ID = {doc["id"]: doc for doc in KNOWLEDGE_BASE}

# ============================================================
# EMBEDDING MODEL (cached)
# ============================================================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def build_embeddings(_model):
    texts = [doc["text"] for doc in KNOWLEDGE_BASE]
    return _model.encode(texts)

model = load_model()
doc_embeddings = build_embeddings(model)


# ============================================================
# AGENT 1: INTENT & PROFILE AGENT
# ============================================================
def intent_profile_agent(user_goal: str, business_structure: str = None) -> dict:
    profile = {
        "goal_text": user_goal,
        "in_scope": is_government_related(user_goal),
        "is_business_related": mentions_business(user_goal),
        "business_structure": business_structure,
        "missing": [],
    }
    if profile["is_business_related"] and not business_structure:
        profile["missing"].append("business_structure")
    return profile


# ============================================================
# AGENT 2: GOVERNMENT RESEARCH & RETRIEVAL AGENT
# ============================================================
def research_agent(query: str, structure: str = None, top_k: int = 6) -> list:
    query_embedding = model.encode([query])[0]
    similarities = np.dot(doc_embeddings, query_embedding) / (
        np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    ranked_indices = np.argsort(similarities)[::-1]

    evidence = []
    for i in ranked_indices:
        doc = KNOWLEDGE_BASE[i]
        if structure and structure not in doc["applies_to_structure"]:
            continue
        evidence.append(doc)
        if len(evidence) >= top_k:
            break
    return evidence


# ============================================================
# AGENT 3: REQUIREMENT ANALYSIS AGENT
# ============================================================
def requirement_agent(evidence: list) -> dict:
    return {
        "mandatory": [d for d in evidence if d["classification"] == "mandatory"],
        "conditional": [d for d in evidence if d["classification"] == "conditional"],
        "optional": [d for d in evidence if d["classification"] == "optional"],
    }


# ============================================================
# AGENT 4: VERIFICATION AGENT
# ============================================================
WEAK_STATUSES = {"Official but date unclear", "Secondary", "Unverified", "Potentially outdated"}

def verification_agent(requirements: dict) -> dict:
    verified = {"mandatory": [], "conditional": [], "optional": [], "flags": []}
    for bucket in ["mandatory", "conditional", "optional"]:
        for doc in requirements[bucket]:
            doc = dict(doc)
            if doc["verification_status"] in WEAK_STATUSES:
                doc["flagged"] = True
                verified["flags"].append(
                    f"{doc['title']} ({doc['institution']}) has status "
                    f"'{doc['verification_status']}' — treat with caution."
                )
            else:
                doc["flagged"] = False
            verified[bucket].append(doc)
    return verified


# ============================================================
# AGENT 5: ROADMAP AGENT  (Part 8 — upgraded)
# ------------------------------------------------------------
# Responsibility: turn verified requirements into a
# DEPENDENCY-ORDERED sequence (a simple topological sort),
# attach source backing to every step, and produce a
# "next step" with a reason, not just a title.
# ============================================================
def _topological_order(docs: list) -> list:
    """
    Order docs so that any doc's dependencies (by id) appear
    before it, using only the ids present in `docs`. Falls back
    to KB-declaration order for anything with no dependency info.
    """
    present_ids = {d["id"] for d in docs}
    doc_by_id = {d["id"]: d for d in docs}

    ordered = []
    visited = set()
    visiting = set()

    def visit(doc_id):
        if doc_id in visited or doc_id not in present_ids:
            return
        if doc_id in visiting:
            return  # cycle guard — skip rather than crash
        visiting.add(doc_id)
        for dep_id in doc_by_id[doc_id]["depends_on"]:
            visit(dep_id)
        visiting.discard(doc_id)
        visited.add(doc_id)
        ordered.append(doc_by_id[doc_id])

    for doc in docs:
        visit(doc["id"])

    return ordered


def roadmap_agent(verified: dict, business_structure: str = None) -> dict:
    # Mandatory items form the backbone; conditional items are
    # woven in wherever their dependencies place them.
    all_docs = verified["mandatory"] + verified["conditional"]
    ordered_docs = _topological_order(all_docs)

    steps = []
    step_num = 1

    # If we know the structure, make that decision explicit as Step 1
    # only when it's implied but not itself in the KB (e.g. sole prop
    # skips SECP, so surfacing the decision keeps the roadmap honest).
    if business_structure:
        steps.append({
            "number": step_num,
            "title": f"Confirm business structure: {business_structure.replace('_', ' ').title()}",
            "institution": "N/A — your decision",
            "status": "done",  # user already answered this in the UI
            "type": "decision",
            "source": None,
            "depends_on_titles": [],
        })
        step_num += 1

    id_to_step_title = {}
    for doc in ordered_docs:
        title = doc["title"] if doc["classification"] == "mandatory" else f"{doc['title']} (if applicable)"
        id_to_step_title[doc["id"]] = title
        steps.append({
            "number": step_num,
            "title": title,
            "institution": doc["institution"],
            "status": "pending",
            "type": doc["classification"],
            "source": {
                "title": doc["title"],
                "institution": doc["institution"],
                "url": doc["url"],
                "verification_status": doc["verification_status"],
            },
            "depends_on_titles": [
                id_to_step_title.get(dep_id, dep_id) for dep_id in doc["depends_on"]
                if dep_id in id_to_step_title
            ],
        })
        step_num += 1

    # Next step = first pending step, with a reason
    next_step = None
    for step in steps:
        if step["status"] == "pending":
            if step["depends_on_titles"]:
                dep_text = ", ".join(step["depends_on_titles"])
                reason = f"This comes next because it depends on: {dep_text}."
            else:
                reason = "This has no unmet dependencies, so you can start here."
            next_step = {"title": step["title"], "reason": reason}
            break

    return {"steps": steps, "next_step": next_step}


# ============================================================
# ORCHESTRATOR
# ============================================================
def orchestrator(user_goal: str, business_structure: str = None) -> dict:
    state = {
        "user_goal": user_goal,
        "business_structure": business_structure,
        "profile": None,
        "evidence": [],
        "requirements": {},
        "verified": {},
        "roadmap": {},
    }

    state["profile"] = intent_profile_agent(user_goal, business_structure)

    if not state["profile"]["in_scope"]:
        return state

    if state["profile"]["missing"]:
        return state

    state["evidence"] = research_agent(user_goal, structure=business_structure)
    state["requirements"] = requirement_agent(state["evidence"])
    state["verified"] = verification_agent(state["requirements"])
    state["roadmap"] = roadmap_agent(state["verified"], business_structure)
    return state


# ============================================================
# RENDER HELPERS
# ============================================================
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
    flag = " ⚠️" if doc.get("flagged") else ""
    st.markdown(f"**{doc['title']}**{flag} — *{doc['institution']}*")
    st.write(doc["text"])
    st.caption(f"Why this applies: {doc['reason']}")
    render_source_expander(doc)
    st.divider()

def render_roadmap(roadmap: dict):
    st.subheader("🗺️ Step-by-Step Roadmap")
    if not roadmap["steps"]:
        st.write("No roadmap could be generated for this query.")
        return

    for step in roadmap["steps"]:
        if step["type"] == "decision":
            icon = "✅"
        elif step["type"] == "mandatory":
            icon = "🔲"
        else:
            icon = "◽"

        st.write(f"{icon} **Step {step['number']}: {step['title']}** — {step['institution']}")

        if step["depends_on_titles"]:
            dep_text = ", ".join(step["depends_on_titles"])
            st.caption(f"Depends on: {dep_text}")

        if step["source"]:
            st.caption(
                f"Source: {step['source']['institution']} — "
                f"{step['source']['verification_status']} "
                f"([link]({step['source']['url']}))"
            )

    if roadmap["next_step"]:
        st.subheader("👉 Your Next Step")
        st.success(roadmap["next_step"]["title"])
        st.caption(roadmap["next_step"]["reason"])


# ============================================================
# SESSION STATE
# ============================================================
if "user_goal" not in st.session_state:
    st.session_state.user_goal = ""
if "business_structure" not in st.session_state:
    st.session_state.business_structure = None
if "submitted" not in st.session_state:
    st.session_state.submitted = False

# ============================================================
# UI
# ============================================================
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
    else:
        preview_profile = intent_profile_agent(goal, st.session_state.business_structure)

        if not preview_profile["in_scope"]:
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

            if "business_structure" in preview_profile["missing"]:
                st.subheader("One quick question")
                st.write("What business structure are you planning to use?")

                with st.expander("Why are you asking me this?"):
                    st.write(
                        "Your business structure changes which registrations "
                        "are required, and in what order. For example, a "
                        "sole proprietorship skips SECP entirely, while a "
                        "company must incorporate with SECP before it can "
                        "even apply for its own NTN."
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

            if "business_structure" in preview_profile["missing"] and not st.session_state.business_structure:
                st.info("Please select a business structure above to see personalized results.")
            else:
                state = orchestrator(goal, st.session_state.business_structure)
                verified = state["verified"]
                roadmap = state["roadmap"]

                if state["business_structure"]:
                    st.caption(f"Personalized for: {state['business_structure'].replace('_', ' ').title()}")

                if verified.get("flags"):
                    with st.expander("⚠️ Verification notes"):
                        for f in verified["flags"]:
                            st.write(f"- {f}")

                st.subheader("✅ Mandatory Requirements")
                if verified.get("mandatory"):
                    for doc in verified["mandatory"]:
                        render_requirement(doc)
                else:
                    st.write("No mandatory requirements found for this query.")

                st.subheader("⚠️ Conditional Requirements")
                if verified.get("conditional"):
                    for doc in verified["conditional"]:
                        render_requirement(doc)
                else:
                    st.write("No conditional requirements found for this query.")

                st.subheader("ℹ️ Optional")
                if verified.get("optional"):
                    for doc in verified["optional"]:
                        render_requirement(doc)
                else:
                    st.write("No optional items found for this query.")

                if roadmap:
                    render_roadmap(roadmap)
