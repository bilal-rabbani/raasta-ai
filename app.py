import streamlit as st
import numpy as np
from datetime import date, datetime
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client

st.set_page_config(
    page_title="RAASTA AI",
    page_icon="🛣️",
    layout="centered"
)

# ============================================================
# SUPABASE CLIENT
# ============================================================
@st.cache_resource
def get_supabase_client() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_ANON_KEY"]
    return create_client(url, key)

supabase = get_supabase_client()

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
        "depends_on": [],
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
# AGENTS (unchanged from Part 8)
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


def requirement_agent(evidence: list) -> dict:
    return {
        "mandatory": [d for d in evidence if d["classification"] == "mandatory"],
        "conditional": [d for d in evidence if d["classification"] == "conditional"],
        "optional": [d for d in evidence if d["classification"] == "optional"],
    }


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


def _topological_order(docs: list) -> list:
    present_ids = {d["id"] for d in docs}
    doc_by_id = {d["id"]: d for d in docs}
    ordered = []
    visited = set()
    visiting = set()

    def visit(doc_id):
        if doc_id in visited or doc_id not in present_ids:
            return
        if doc_id in visiting:
            return
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
    all_docs = verified["mandatory"] + verified["conditional"]
    ordered_docs = _topological_order(all_docs)

    steps = []
    step_num = 1

    if business_structure:
        steps.append({
            "step_id": "decision_structure",
            "number": step_num,
            "title": f"Confirm business structure: {business_structure.replace('_', ' ').title()}",
            "institution": "N/A — your decision",
            "status": "done",
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
            "step_id": doc["id"],
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

    return {"steps": steps, "next_step": None}  # next_step recalculated after load/merge


def recalculate_next_step(roadmap: dict) -> dict:
    """Recompute 'Your Next Step' based on current step statuses (post-load or post-checkbox)."""
    next_step = None
    for step in roadmap["steps"]:
        if step["status"] != "done":
            if step["depends_on_titles"]:
                dep_text = ", ".join(step["depends_on_titles"])
                reason = f"This comes next because it depends on: {dep_text}."
            else:
                reason = "This has no unmet dependencies, so you can start here."
            next_step = {"title": step["title"], "reason": reason}
            break
    roadmap["next_step"] = next_step
    return roadmap


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
    roadmap = roadmap_agent(state["verified"], business_structure)
    state["roadmap"] = recalculate_next_step(roadmap)
    return state


# ============================================================
# PERSISTENCE LAYER (Supabase)
# ============================================================
def save_progress(user_id: str, goal_text: str, business_structure: str, roadmap: dict, existing_id: str = None):
    payload = {
        "user_id": user_id,
        "goal_text": goal_text,
        "business_structure": business_structure,
        "roadmap_json": roadmap,
        "updated_at": datetime.utcnow().isoformat(),
    }
    if existing_id:
        supabase.table("raasta_progress").update(payload).eq("id", existing_id).execute()
        return existing_id
    else:
        result = supabase.table("raasta_progress").insert(payload).execute()
        return result.data[0]["id"] if result.data else None


def load_user_goals(user_id: str) -> list:
    result = (
        supabase.table("raasta_progress")
        .select("*")
        .eq("user_id", user_id)
        .order("updated_at", desc=True)
        .execute()
    )
    return result.data or []


def compute_progress_percent(roadmap: dict) -> int:
    steps = roadmap.get("steps", [])
    if not steps:
        return 0
    done = sum(1 for s in steps if s["status"] == "done")
    return round((done / len(steps)) * 100)


# ============================================================
# AUTH HELPERS
# ============================================================
def sign_up(email: str, password: str):
    return supabase.auth.sign_up({"email": email, "password": password})

def sign_in(email: str, password: str):
    return supabase.auth.sign_in_with_password({"email": email, "password": password})

def sign_out():
    supabase.auth.sign_out()


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

def render_roadmap_with_progress(roadmap: dict, editable: bool = True):
    st.subheader("🗺️ Step-by-Step Roadmap")
    if not roadmap["steps"]:
        st.write("No roadmap could be generated for this query.")
        return roadmap

    percent = compute_progress_percent(roadmap)
    st.progress(percent / 100, text=f"Progress: {percent}%")

    changed = False
    for step in roadmap["steps"]:
        icon = "✅" if step["status"] == "done" else ("🔲" if step["type"] == "mandatory" else "◽")
        col1, col2 = st.columns([0.08, 0.92])
        with col1:
            if editable and step["type"] != "decision":
                checked = st.checkbox(
                    "",
                    value=(step["status"] == "done"),
                    key=f"step_{step['step_id']}",
                    label_visibility="collapsed",
                )
                new_status = "done" if checked else "pending"
                if new_status != step["status"]:
                    step["status"] = new_status
                    changed = True
            else:
                st.write(icon)
        with col2:
            st.write(f"**Step {step['number']}: {step['title']}** — {step['institution']}")
            if step["depends_on_titles"]:
                st.caption(f"Depends on: {', '.join(step['depends_on_titles'])}")
            if step["source"]:
                st.caption(
                    f"Source: {step['source']['institution']} — "
                    f"{step['source']['verification_status']} "
                    f"([link]({step['source']['url']}))"
                )

    if changed:
        recalculate_next_step(roadmap)

    if roadmap["next_step"]:
        st.subheader("👉 Your Next Step")
        st.success(roadmap["next_step"]["title"])
        st.caption(roadmap["next_step"]["reason"])
    else:
        st.success("🎉 All steps complete for this goal!")

    return roadmap


# ============================================================
# SESSION STATE
# ============================================================
defaults = {
    "user": None,
    "user_goal": "",
    "business_structure": None,
    "submitted": False,
    "active_roadmap": None,
    "active_goal_id": None,
    "active_goal_text": None,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ============================================================
# AUTH UI (sidebar)
# ============================================================
with st.sidebar:
    st.header("Account")

    if st.session_state.user is None:
        auth_mode = st.radio("Choose:", ["Log In", "Sign Up"], horizontal=True)
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        if auth_mode == "Sign Up":
            if st.button("Create Account"):
                try:
                    res = sign_up(email, password)
                    if res.user:
                        st.success("Account created. Please log in.")
                    else:
                        st.error("Sign up failed. Try a different email/password.")
                except Exception as e:
                    st.error(f"Sign up error: {e}")
        else:
            if st.button("Log In"):
                try:
                    res = sign_in(email, password)
                    if res.user:
                        st.session_state.user = res.user
                        st.rerun()
                    else:
                        st.error("Login failed. Check your credentials.")
                except Exception as e:
                    st.error(f"Login error: {e}")
    else:
        st.write(f"Logged in as **{st.session_state.user.email}**")
        if st.button("Log Out"):
            sign_out()
            st.session_state.user = None
            st.session_state.active_roadmap = None
            st.session_state.active_goal_id = None
            st.rerun()

        st.divider()
        st.subheader("My Saved Goals")
        saved_goals = load_user_goals(st.session_state.user.id)
        if not saved_goals:
            st.caption("No saved goals yet.")
        for g in saved_goals:
            pct = compute_progress_percent(g["roadmap_json"])
            label = f"{g['goal_text'][:35]}... ({pct}%)" if len(g["goal_text"]) > 35 else f"{g['goal_text']} ({pct}%)"
            if st.button(label, key=f"load_{g['id']}"):
                st.session_state.active_roadmap = g["roadmap_json"]
                st.session_state.active_goal_id = g["id"]
                st.session_state.active_goal_text = g["goal_text"]
                st.session_state.business_structure = g["business_structure"]
                st.session_state.submitted = False  # skip re-running the full pipeline
                st.rerun()

# ============================================================
# MAIN UI
# ============================================================
st.title("RAASTA AI")
st.write(
    "Tell RAASTA AI what government-related task you want to accomplish "
    "in Pakistan."
)

if st.session_state.user is None:
    st.info("Log in or sign up in the sidebar to save your progress across sessions.")

# ---- If a saved goal was loaded from the sidebar, render it directly ----
if st.session_state.active_roadmap and not st.session_state.submitted:
    st.subheader("Loaded goal:")
    st.write(st.session_state.active_goal_text)
    updated_roadmap = render_roadmap_with_progress(st.session_state.active_roadmap, editable=True)

    if st.button("💾 Save Progress"):
        save_progress(
            user_id=st.session_state.user.id,
            goal_text=st.session_state.active_goal_text,
            business_structure=st.session_state.business_structure,
            roadmap=updated_roadmap,
            existing_id=st.session_state.active_goal_id,
        )
        st.success("Progress saved.")

    if st.button("Start a new goal"):
        st.session_state.active_roadmap = None
        st.session_state.active_goal_id = None
        st.rerun()

else:
    user_goal_input = st.text_area(
        "What do you want to accomplish?",
        placeholder="Example: I want to start a construction business in Lahore.",
        value=st.session_state.user_goal,
    )

    if st.button("Ask RAASTA AI"):
        st.session_state.user_goal = user_goal_input
        st.session_state.submitted = True
        st.session_state.business_structure = None
        st.session_state.active_roadmap = None
        st.session_state.active_goal_id = None

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
                            "are required, and in what order."
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
                    for doc in verified.get("mandatory", []):
                        render_requirement(doc)
                    if not verified.get("mandatory"):
                        st.write("No mandatory requirements found for this query.")

                    st.subheader("⚠️ Conditional Requirements")
                    for doc in verified.get("conditional", []):
                        render_requirement(doc)
                    if not verified.get("conditional"):
                        st.write("No conditional requirements found for this query.")

                    st.subheader("ℹ️ Optional")
                    for doc in verified.get("optional", []):
                        render_requirement(doc)
                    if not verified.get("optional"):
                        st.write("No optional items found for this query.")

                    if roadmap and roadmap.get("steps"):
                        updated_roadmap = render_roadmap_with_progress(roadmap, editable=True)

                        if st.session_state.user is not None:
                            if st.button("💾 Save this goal"):
                                new_id = save_progress(
                                    user_id=st.session_state.user.id,
                                    goal_text=goal,
                                    business_structure=state["business_structure"],
                                    roadmap=updated_roadmap,
                                )
                                st.session_state.active_goal_id = new_id
                                st.session_state.active_goal_text = goal
                                st.success("Goal saved! You'll find it in 'My Saved Goals' next time you log in.")
                        else:
                            st.info("Log in to save this roadmap and track your progress across sessions.")
