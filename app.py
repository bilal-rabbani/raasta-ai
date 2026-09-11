import streamlit as st
import numpy as np
import requests
from datetime import date, datetime
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client

st.set_page_config(
    page_title="RAASTA AI",
    page_icon="🛣️",
    layout="centered"
)

# ============================================================
# TRANSLATIONS (UI text)
# ============================================================
TRANSLATIONS = {
    "app_title": {"en": "RAASTA AI", "ur": "راستہ اے آئی"},
    "app_intro": {
        "en": "Tell RAASTA AI what government-related task you want to accomplish in Pakistan.",
        "ur": "راستہ اے آئی کو بتائیں کہ آپ پاکستان میں کون سا سرکاری کام مکمل کرنا چاہتے ہیں۔",
    },
    "account_header": {"en": "Account", "ur": "اکاؤنٹ"},
    "log_in": {"en": "Log In", "ur": "لاگ ان"},
    "sign_up": {"en": "Sign Up", "ur": "سائن اپ"},
    "email": {"en": "Email", "ur": "ای میل"},
    "password": {"en": "Password", "ur": "پاس ورڈ"},
    "create_account": {"en": "Create Account", "ur": "اکاؤنٹ بنائیں"},
    "account_created": {"en": "Account created. Please log in.", "ur": "اکاؤنٹ بن گیا۔ براہ کرم لاگ ان کریں۔"},
    "signup_failed": {"en": "Sign up failed. Try a different email/password.", "ur": "سائن اپ ناکام۔ دوسرا ای میل/پاس ورڈ آزمائیں۔"},
    "login_failed": {"en": "Login failed. Check your credentials.", "ur": "لاگ ان ناکام۔ اپنی معلومات چیک کریں۔"},
    "logged_in_as": {"en": "Logged in as", "ur": "لاگ ان بطور"},
    "log_out": {"en": "Log Out", "ur": "لاگ آؤٹ"},
    "my_saved_goals": {"en": "My Saved Goals", "ur": "میرے محفوظ کردہ اہداف"},
    "no_saved_goals": {"en": "No saved goals yet.", "ur": "ابھی تک کوئی ہدف محفوظ نہیں۔"},
    "login_info": {"en": "Log in or sign up in the sidebar to save your progress across sessions.", "ur": "اپنی پیش رفت محفوظ کرنے کے لیے سائیڈبار میں لاگ ان یا سائن اپ کریں۔"},
    "loaded_goal": {"en": "Loaded goal:", "ur": "لوڈ کیا گیا ہدف:"},
    "save_progress": {"en": "💾 Save Progress", "ur": "💾 پیش رفت محفوظ کریں"},
    "progress_saved": {"en": "Progress saved.", "ur": "پیش رفت محفوظ ہو گئی۔"},
    "start_new_goal": {"en": "Start a new goal", "ur": "نیا ہدف شروع کریں"},
    "goal_input_label": {"en": "What do you want to accomplish?", "ur": "آپ کیا حاصل کرنا چاہتے ہیں؟"},
    "goal_input_placeholder": {
        "en": "Example: I want to start a construction business in Lahore.",
        "ur": "مثال: میں لاہور میں تعمیراتی کاروبار شروع کرنا چاہتا ہوں۔",
    },
    "ask_button": {"en": "Ask RAASTA AI", "ur": "راستہ اے آئی سے پوچھیں"},
    "please_tell_us": {"en": "Please tell us what you want to accomplish.", "ur": "براہ کرم بتائیں کہ آپ کیا حاصل کرنا چاہتے ہیں۔"},
    "out_of_scope": {
        "en": "RAASTA AI is designed to help with government procedures and services in Pakistan. Please ask about a government registration, license, permit, application, tax, service, or other government procedure.",
        "ur": "راستہ اے آئی پاکستان میں سرکاری امور اور خدمات میں مدد کے لیے بنایا گیا ہے۔ براہ کرم سرکاری رجسٹریشن، لائسنس، پرمٹ، درخواست، ٹیکس، یا کسی اور سرکاری طریقہ کار کے بارے میں پوچھیں۔",
    },
    "looks_government": {"en": "This looks like a government-related request.", "ur": "یہ ایک سرکاری نوعیت کی درخواست لگتی ہے۔"},
    "you_asked": {"en": "You asked:", "ur": "آپ نے پوچھا:"},
    "one_quick_question": {"en": "One quick question", "ur": "ایک مختصر سوال"},
    "business_structure_question": {"en": "What business structure are you planning to use?", "ur": "آپ کس قسم کا کاروباری ڈھانچہ استعمال کرنے کا ارادہ رکھتے ہیں؟"},
    "why_asking": {"en": "Why are you asking me this?", "ur": "آپ مجھ سے یہ کیوں پوچھ رہے ہیں؟"},
    "why_asking_explanation": {
        "en": "Your business structure changes which registrations are required, and in what order.",
        "ur": "آپ کا کاروباری ڈھانچہ اس بات کا تعین کرتا ہے کہ کون سی رجسٹریشنز درکار ہیں اور کس ترتیب میں۔",
    },
    "choose_one": {"en": "Choose one:", "ur": "ایک منتخب کریں:"},
    "sole_proprietorship": {"en": "Sole Proprietorship", "ur": "واحد ملکیت"},
    "partnership": {"en": "Partnership", "ur": "شراکت داری"},
    "company": {"en": "Company", "ur": "کمپنی"},
    "select_structure_prompt": {"en": "Please select a business structure above to see personalized results.", "ur": "ذاتی نتائج دیکھنے کے لیے براہ کرم اوپر کاروباری ڈھانچہ منتخب کریں۔"},
    "personalized_for": {"en": "Personalized for", "ur": "کے لیے موزوں کردہ"},
    "verification_notes": {"en": "⚠️ Verification notes", "ur": "⚠️ تصدیقی نوٹس"},
    "mandatory_requirements": {"en": "✅ Mandatory Requirements", "ur": "✅ لازمی تقاضے"},
    "no_mandatory": {"en": "No mandatory requirements found for this query.", "ur": "اس سوال کے لیے کوئی لازمی تقاضے نہیں ملے۔"},
    "conditional_requirements": {"en": "⚠️ Conditional Requirements", "ur": "⚠️ مشروط تقاضے"},
    "no_conditional": {"en": "No conditional requirements found for this query.", "ur": "اس سوال کے لیے کوئی مشروط تقاضے نہیں ملے۔"},
    "optional_requirements": {"en": "ℹ️ Optional", "ur": "ℹ️ اختیاری"},
    "no_optional": {"en": "No optional items found for this query.", "ur": "اس سوال کے لیے کوئی اختیاری آئٹم نہیں ملا۔"},
    "why_this_applies": {"en": "Why this applies", "ur": "یہ کیوں لاگو ہوتا ہے"},
    "roadmap_header": {"en": "🗺️ Step-by-Step Roadmap", "ur": "🗺️ مرحلہ وار روڈ میپ"},
    "no_roadmap": {"en": "No roadmap could be generated for this query.", "ur": "اس سوال کے لیے کوئی روڈ میپ نہیں بن سکا۔"},
    "progress_label": {"en": "Progress", "ur": "پیش رفت"},
    "depends_on": {"en": "Depends on", "ur": "کا انحصار ہے"},
    "source_label": {"en": "Source", "ur": "ماخذ"},
    "your_next_step": {"en": "👉 Your Next Step", "ur": "👉 آپ کا اگلا قدم"},
    "all_complete": {"en": "🎉 All steps complete for this goal!", "ur": "🎉 اس ہدف کے تمام مراحل مکمل ہو گئے!"},
    "save_goal_button": {"en": "💾 Save this goal", "ur": "💾 یہ ہدف محفوظ کریں"},
    "goal_saved": {"en": "Goal saved! You'll find it in 'My Saved Goals' next time you log in.", "ur": "ہدف محفوظ ہو گیا! اگلی بار لاگ ان کرنے پر یہ 'میرے محفوظ کردہ اہداف' میں ملے گا۔"},
    "login_to_save": {"en": "Log in to save this roadmap and track your progress across sessions.", "ur": "یہ روڈ میپ محفوظ کرنے اور پیش رفت ٹریک کرنے کے لیے لاگ ان کریں۔"},
    "language_label": {"en": "Language", "ur": "زبان"},
    "institution_label": {"en": "Institution", "ur": "ادارہ"},
    "title_label": {"en": "Title", "ur": "عنوان"},
    "url_label": {"en": "URL", "ur": "یو آر ایل"},
    "source_type_label": {"en": "Source Type", "ur": "ماخذ کی قسم"},
    "publication_date_label": {"en": "Publication Date", "ur": "اشاعت کی تاریخ"},
    "status_label": {"en": "Status", "ur": "حیثیت"},
    "retrieved_label": {"en": "Retrieved", "ur": "حاصل کردہ تاریخ"},
    "ai_summary_header": {"en": "🤖 Plain-Language Summary", "ur": "🤖 سادہ زبان میں خلاصہ"},
    "ai_summary_unavailable": {
        "en": "A plain-language AI summary isn't available right now, but the verified information above is complete and accurate.",
        "ur": "اس وقت اے آئی خلاصہ دستیاب نہیں، لیکن اوپر دی گئی تصدیق شدہ معلومات مکمل اور درست ہیں۔",
    },
}

def t(key: str) -> str:
    lang = st.session_state.get("language", "en")
    return TRANSLATIONS.get(key, {}).get(lang, key)

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
# LLM ROUTER (Primary + Fallback) — Part 11
# ============================================================
def _call_openai_compatible(base_url, api_key, model, prompt, timeout=20):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are RAASTA AI, a Pakistani government procedure assistant. "
                    "Explain the given information clearly and briefly in plain language. "
                    "Only use the facts given to you — never invent requirements, fees, or dates."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 300,
        "temperature": 0.3,
    }
    response = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def llm_router(prompt: str) -> dict:
    """
    Tries the primary LLM provider first, then falls back to the next
    configured provider if it fails. Providers are only used if their
    API key exists in Streamlit secrets. If none succeed, returns
    success=False instead of guessing — no fabricated output.
    """
    providers = []

    if "GROQ_API_KEY" in st.secrets:
        providers.append({
            "name": "Groq",
            "base_url": "https://api.groq.com/openai/v1",
            "api_key": st.secrets["GROQ_API_KEY"],
            "model": st.secrets.get("GROQ_MODEL", "openai/gpt-oss-20b"),
        })

    if "OPENAI_API_KEY" in st.secrets:
        providers.append({
            "name": "OpenAI",
            "base_url": "https://api.openai.com/v1",
            "api_key": st.secrets["OPENAI_API_KEY"],
            "model": st.secrets.get("OPENAI_MODEL", "gpt-4o-mini"),
        })

    if "OPENROUTER_API_KEY" in st.secrets:
        providers.append({
            "name": "OpenRouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": st.secrets["OPENROUTER_API_KEY"],
            "model": st.secrets.get("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free"),
        })

    if not providers:
        return {"success": False, "text": None, "provider_used": None, "error": "No LLM provider configured."}

    last_error = None
    for provider in providers:
        try:
            text = _call_openai_compatible(
                provider["base_url"], provider["api_key"], provider["model"], prompt
            )
            return {"success": True, "text": text, "provider_used": provider["name"], "error": None}
        except Exception as e:
            last_error = str(e)
            continue

    return {"success": False, "text": None, "provider_used": None, "error": last_error}


def build_summary_prompt(goal: str, structure: str, verified: dict) -> str:
    lines = [f"User's goal: {goal}"]
    if structure:
        lines.append(f"Business structure: {structure.replace('_', ' ')}")
    lines.append("Mandatory requirements:")
    for d in verified.get("mandatory", []):
        lines.append(f"- {d['title']} ({d['institution']})")
    lines.append("Conditional requirements:")
    for d in verified.get("conditional", []):
        lines.append(f"- {d['title']} ({d['institution']}) — {d['reason']}")
    lines.append(
        "Write a short, friendly 3-4 sentence plain-language summary of what "
        "this person needs to do, in the order they should do it. Do not add "
        "any requirement not listed above."
    )
    return "\n".join(lines)

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
        "title_ur": "کمپنی رجسٹریشن کا جائزہ",
        "url": "https://www.secp.gov.pk/",
        "text": (
            "To register a company in Pakistan, you must apply through SECP's "
            "e-Services portal. Required documents typically include CNIC "
            "copies of directors, a proposed company name, and a memorandum "
            "of association."
        ),
        "text_ur": (
            "پاکستان میں کمپنی رجسٹر کرنے کے لیے آپ کو SECP کے ای-سروسز پورٹل کے "
            "ذریعے درخواست دینی ہوگی۔ عام طور پر درکار دستاویزات میں ڈائریکٹرز کے "
            "شناختی کارڈ کی کاپیاں، تجویز کردہ کمپنی کا نام، اور میمورنڈم آف ایسوسی ایشن شامل ہیں۔"
        ),
        "source_type": "Official Government Portal",
        "verification_status": "Verified / Current",
        "publication_date": "2023-01-15",
        "applies_to_structure": ["company"],
        "classification": "mandatory",
        "reason": "Registering as a company legally requires SECP incorporation before the business can operate.",
        "reason_ur": "کمپنی کے طور پر رجسٹریشن کے لیے کاروبار شروع کرنے سے پہلے قانونی طور پر SECP سے انکارپوریشن ضروری ہے۔",
        "depends_on": [],
    },
    {
        "id": "fbr_ntn",
        "institution": "FBR (Federal Board of Revenue)",
        "title": "National Tax Number (NTN) Registration",
        "title_ur": "قومی ٹیکس نمبر (این ٹی این) کی رجسٹریشن",
        "url": "https://www.fbr.gov.pk/",
        "text": (
            "Businesses operating in Pakistan must register for a National "
            "Tax Number (NTN) with FBR. This is required for filing income "
            "tax and is typically done online through the IRIS portal."
        ),
        "text_ur": (
            "پاکستان میں کام کرنے والے کاروباروں کو FBR کے ساتھ قومی ٹیکس نمبر "
            "(NTN) رجسٹر کروانا ضروری ہے۔ یہ انکم ٹیکس فائل کرنے کے لیے درکار ہے اور "
            "عام طور پر آئی آر آئی ایس پورٹل کے ذریعے آن لائن کیا جاتا ہے۔"
        ),
        "source_type": "Official Government Portal",
        "verification_status": "Verified / Current",
        "publication_date": "2023-03-10",
        "applies_to_structure": ["sole_proprietorship", "partnership", "company"],
        "classification": "mandatory",
        "reason": "All business structures must have an NTN to file taxes, regardless of size or type.",
        "reason_ur": "ہر قسم کے کاروباری ڈھانچے کے لیے ٹیکس فائل کرنے کے لیے این ٹی این ضروری ہے۔",
        "depends_on": ["secp_company_reg"],
    },
    {
        "id": "fbr_sole_prop",
        "institution": "FBR (Federal Board of Revenue)",
        "title": "Registering as a Sole Proprietor",
        "title_ur": "واحد ملکیت کے طور پر رجسٹریشن",
        "url": "https://www.fbr.gov.pk/",
        "text": (
            "A sole proprietorship does not require SECP registration. "
            "The owner registers directly with FBR for an NTN under their "
            "own CNIC, and this is generally the simplest business structure "
            "to set up in Pakistan."
        ),
        "text_ur": (
            "واحد ملکیت کے لیے SECP رجسٹریشن درکار نہیں۔ مالک اپنے شناختی کارڈ "
            "کے تحت براہ راست FBR کے ساتھ NTN رجسٹر کرواتا ہے، اور یہ پاکستان میں "
            "قائم کرنے کا سب سے آسان کاروباری ڈھانچہ ہے۔"
        ),
        "source_type": "Official Government Portal",
        "verification_status": "Verified / Current",
        "publication_date": "2023-02-01",
        "applies_to_structure": ["sole_proprietorship"],
        "classification": "mandatory",
        "reason": "As a sole proprietor, NTN registration under your own CNIC is the primary legal registration step.",
        "reason_ur": "واحد ملکیت میں، اپنے شناختی کارڈ کے تحت NTN رجسٹریشن بنیادی قانونی قدم ہے۔",
        "depends_on": [],
    },
    {
        "id": "punjab_local_approval",
        "institution": "Punjab Government - PBIT",
        "title": "Business Setup Guidance for Punjab",
        "title_ur": "پنجاب کے لیے کاروبار قائم کرنے کی رہنمائی",
        "url": "https://invest.punjab.gov.pk/",
        "text": (
            "Businesses setting up in Punjab, including construction-related "
            "businesses, may need approvals from local development "
            "authorities depending on the nature and location of the "
            "business activity."
        ),
        "text_ur": (
            "پنجاب میں کاروبار قائم کرنے والوں، بشمول تعمیراتی کاروبار، کو کاروبار "
            "کی نوعیت اور مقام کے مطابق مقامی ترقیاتی اداروں سے منظوری کی ضرورت ہو سکتی ہے۔"
        ),
        "source_type": "Official Government Portal",
        "verification_status": "Official but date unclear",
        "publication_date": "Unknown",
        "applies_to_structure": ["sole_proprietorship", "partnership", "company"],
        "classification": "conditional",
        "reason": "This only applies if your specific business activity or location requires local development authority approval.",
        "reason_ur": "یہ صرف اس صورت میں لاگو ہوتا ہے جب آپ کی کاروباری سرگرمی یا مقام کو مقامی ترقیاتی ادارے کی منظوری درکار ہو۔",
        "depends_on": ["fbr_ntn", "fbr_sole_prop"],
    },
    {
        "id": "pec_construction_reg",
        "institution": "PEC (Pakistan Engineering Council)",
        "title": "Construction Firm Registration",
        "title_ur": "تعمیراتی فرم کی رجسٹریشن",
        "url": "https://www.pec.org.pk/",
        "text": (
            "Construction companies undertaking engineering works in "
            "Pakistan are generally required to register with the Pakistan "
            "Engineering Council (PEC) to be eligible for certain "
            "government and private contracts."
        ),
        "text_ur": (
            "پاکستان میں انجینئرنگ کے کام کرنے والی تعمیراتی کمپنیوں کو بعض سرکاری "
            "اور نجی ٹھیکوں کے لیے اہل ہونے کے لیے عام طور پر پاکستان انجینئرنگ کونسل "
            "(PEC) کے ساتھ رجسٹر ہونا ضروری ہوتا ہے۔"
        ),
        "source_type": "Official Government Portal",
        "verification_status": "Verified / Current",
        "publication_date": "2022-11-05",
        "applies_to_structure": ["partnership", "company"],
        "classification": "conditional",
        "reason": "Required only if you plan to bid on government or PEC-regulated engineering contracts, not for all construction work.",
        "reason_ur": "یہ صرف اس صورت میں درکار ہے جب آپ سرکاری یا PEC کے زیرِ انتظام انجینئرنگ ٹھیکوں کے لیے بولی دینا چاہتے ہوں۔",
        "depends_on": ["fbr_ntn"],
    },
    {
        "id": "chamber_membership",
        "institution": "Punjab Chamber of Commerce",
        "title": "Chamber of Commerce Membership",
        "title_ur": "چیمبر آف کامرس کی رکنیت",
        "url": "https://example-lcci.pk/",
        "text": (
            "Businesses may optionally join their local Chamber of Commerce "
            "and Industry for networking, trade certificates, and business "
            "advocacy support. This is not a legal requirement to operate."
        ),
        "text_ur": (
            "کاروبار اختیاری طور پر اپنے مقامی چیمبر آف کامرس اینڈ انڈسٹری میں شامل "
            "ہو سکتے ہیں تاکہ رابطہ کاری، تجارتی سرٹیفکیٹس، اور کاروباری معاونت حاصل "
            "کی جا سکے۔ یہ کاروبار چلانے کے لیے قانونی طور پر لازمی نہیں۔"
        ),
        "source_type": "Secondary / Industry Body",
        "verification_status": "Secondary",
        "publication_date": "Unknown",
        "applies_to_structure": ["sole_proprietorship", "partnership", "company"],
        "classification": "optional",
        "reason": "Chamber membership provides business benefits but is not legally required to operate.",
        "reason_ur": "چیمبر کی رکنیت کاروباری فوائد دیتی ہے لیکن قانونی طور پر لازمی نہیں۔",
        "depends_on": [],
    },
]

KB_BY_ID = {doc["id"]: doc for doc in KNOWLEDGE_BASE}

def loc_text(doc, field_en, field_ur_key):
    lang = st.session_state.get("language", "en")
    if lang == "ur" and doc.get(field_ur_key):
        return doc[field_ur_key]
    return doc[field_en]

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
# AGENTS
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
            "depends_on_ids": [],
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
            "depends_on_ids": doc["depends_on"],
        })
        step_num += 1

    return {"steps": steps, "next_step": None}


def recalculate_next_step(roadmap: dict) -> dict:
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
    title_display = loc_text(doc, "title", "title_ur")
    with st.expander(f"📄 {title_display} ({doc['institution']})"):
        st.write(f"**{t('institution_label')}:** {doc['institution']}")
        st.write(f"**{t('title_label')}:** {title_display}")
        st.write(f"**{t('url_label')}:** {doc['url']}")
        st.write(f"**{t('source_type_label')}:** {doc['source_type']}")
        st.write(f"**{t('publication_date_label')}:** {doc['publication_date']}")
        st.write(f"**{t('status_label')}:** {doc['verification_status']}")
        st.write(f"**{t('retrieved_label')}:** {date.today().isoformat()}")

def render_requirement(doc):
    flag = " ⚠️" if doc.get("flagged") else ""
    title_display = loc_text(doc, "title", "title_ur")
    text_display = loc_text(doc, "text", "text_ur")
    reason_display = loc_text(doc, "reason", "reason_ur")
    st.markdown(f"**{title_display}**{flag} — *{doc['institution']}*")
    st.write(text_display)
    st.caption(f"{t('why_this_applies')}: {reason_display}")
    render_source_expander(doc)
    st.divider()


def render_roadmap_with_progress(roadmap: dict, editable: bool = True):
    st.subheader(t("roadmap_header"))
    if not roadmap["steps"]:
        st.write(t("no_roadmap"))
        return roadmap

    # Order of togglable steps (excludes the "decision" step), used both
    # for the cascade-uncheck logic and for drawing the checkboxes.
    toggle_order = [s["step_id"] for s in roadmap["steps"] if s["type"] != "decision"]

    def cascade_uncheck(step_id):
        # Runs the instant a checkbox is clicked, BEFORE the page redraws.
        # If it was unchecked, force every step listed below it to unchecked too.
        key = f"step_{step_id}"
        if not st.session_state.get(key, False):
            idx = toggle_order.index(step_id)
            for later_id in toggle_order[idx + 1:]:
                st.session_state[f"step_{later_id}"] = False

    for step in roadmap["steps"]:
        icon = "✅" if step["status"] == "done" else ("🔲" if step["type"] == "mandatory" else "◽")
        col1, col2 = st.columns([0.08, 0.92])
        with col1:
            if editable and step["type"] != "decision":
                st.checkbox(
                    "",
                    value=(step["status"] == "done"),
                    key=f"step_{step['step_id']}",
                    label_visibility="collapsed",
                    on_change=cascade_uncheck,
                    args=(step["step_id"],),
                )
            else:
                st.write(icon)
        with col2:
            st.write(f"**Step {step['number']}: {step['title']}** — {step['institution']}")
            if step["depends_on_titles"]:
                st.caption(f"{t('depends_on')}: {', '.join(step['depends_on_titles'])}")
            if step["source"]:
                st.caption(
                    f"{t('source_label')}: {step['source']['institution']} — "
                    f"{step['source']['verification_status']} "
                    f"([link]({step['source']['url']}))"
                )

    # IMPORTANT: sync statuses from the checkbox widgets AFTER all checkboxes
    # are drawn (and after any cascade), THEN compute the percentage — this
    # is the fix for the progress bar being stuck.
    for step in roadmap["steps"]:
        if step["type"] != "decision":
            checked = st.session_state.get(f"step_{step['step_id']}", False)
            step["status"] = "done" if checked else "pending"

    percent = compute_progress_percent(roadmap)
    st.progress(percent / 100, text=f"{t('progress_label')}: {percent}%")

    recalculate_next_step(roadmap)

    if roadmap["next_step"]:
        st.subheader(t("your_next_step"))
        st.success(roadmap["next_step"]["title"])
        st.caption(roadmap["next_step"]["reason"])
    else:
        st.success(t("all_complete"))

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
    "language": "en",
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ============================================================
# SIDEBAR: LANGUAGE + AUTH
# ============================================================
with st.sidebar:
    lang_choice = st.selectbox(
        t("language_label"),
        options=["English", "اردو"],
        index=0 if st.session_state.language == "en" else 1,
    )
    st.session_state.language = "en" if lang_choice == "English" else "ur"

    st.divider()
    st.header(t("account_header"))

    if st.session_state.user is None:
        auth_mode = st.radio("", [t("log_in"), t("sign_up")], horizontal=True, label_visibility="collapsed")
        email = st.text_input(t("email"))
        password = st.text_input(t("password"), type="password")

        if auth_mode == t("sign_up"):
            if st.button(t("create_account")):
                try:
                    res = sign_up(email, password)
                    if res.user:
                        st.success(t("account_created"))
                    else:
                        st.error(t("signup_failed"))
                except Exception as e:
                    st.error(f"{t('signup_failed')} ({e})")
        else:
            if st.button(t("log_in")):
                try:
                    res = sign_in(email, password)
                    if res.user:
                        st.session_state.user = res.user
                        st.rerun()
                    else:
                        st.error(t("login_failed"))
                except Exception as e:
                    st.error(f"{t('login_failed')} ({e})")
    else:
        st.write(f"{t('logged_in_as')} **{st.session_state.user.email}**")
        if st.button(t("log_out")):
            sign_out()
            st.session_state.user = None
            st.session_state.active_roadmap = None
            st.session_state.active_goal_id = None
            st.rerun()

        st.divider()
        st.subheader(t("my_saved_goals"))
        saved_goals = load_user_goals(st.session_state.user.id)
        if not saved_goals:
            st.caption(t("no_saved_goals"))
        for g in saved_goals:
            pct = compute_progress_percent(g["roadmap_json"])
            label = f"{g['goal_text'][:35]}... ({pct}%)" if len(g["goal_text"]) > 35 else f"{g['goal_text']} ({pct}%)"
            if st.button(label, key=f"load_{g['id']}"):
                st.session_state.active_roadmap = g["roadmap_json"]
                st.session_state.active_goal_id = g["id"]
                st.session_state.active_goal_text = g["goal_text"]
                st.session_state.business_structure = g["business_structure"]
                st.session_state.submitted = False
                st.rerun()

# ============================================================
# MAIN UI
# ============================================================
st.title(t("app_title"))
st.write(t("app_intro"))

if st.session_state.user is None:
    st.info(t("login_info"))

if st.session_state.active_roadmap and not st.session_state.submitted:
    st.subheader(t("loaded_goal"))
    st.write(st.session_state.active_goal_text)
    updated_roadmap = render_roadmap_with_progress(st.session_state.active_roadmap, editable=True)

    if st.button(t("save_progress")):
        save_progress(
            user_id=st.session_state.user.id,
            goal_text=st.session_state.active_goal_text,
            business_structure=st.session_state.business_structure,
            roadmap=updated_roadmap,
            existing_id=st.session_state.active_goal_id,
        )
        st.success(t("progress_saved"))

    if st.button(t("start_new_goal")):
        st.session_state.active_roadmap = None
        st.session_state.active_goal_id = None
        st.rerun()

else:
    user_goal_input = st.text_area(
        t("goal_input_label"),
        placeholder=t("goal_input_placeholder"),
        value=st.session_state.user_goal,
    )

    if st.button(t("ask_button")):
        st.session_state.user_goal = user_goal_input
        st.session_state.submitted = True
        st.session_state.business_structure = None
        st.session_state.active_roadmap = None
        st.session_state.active_goal_id = None

    if st.session_state.submitted:
        goal = st.session_state.user_goal

        if not goal.strip():
            st.warning(t("please_tell_us"))
        else:
            preview_profile = intent_profile_agent(goal, st.session_state.business_structure)

            if not preview_profile["in_scope"]:
                st.info(t("out_of_scope"))
            else:
                st.success(t("looks_government"))
                st.write(t("you_asked"))
                st.write(goal)

                if "business_structure" in preview_profile["missing"]:
                    st.subheader(t("one_quick_question"))
                    st.write(t("business_structure_question"))

                    with st.expander(t("why_asking")):
                        st.write(t("why_asking_explanation"))

                    structure_choice = st.radio(
                        t("choose_one"),
                        options=[t("sole_proprietorship"), t("partnership"), t("company")],
                        index=None,
                        key="structure_radio",
                    )
                    structure_map = {
                        t("sole_proprietorship"): "sole_proprietorship",
                        t("partnership"): "partnership",
                        t("company"): "company",
                    }
                    if structure_choice:
                        st.session_state.business_structure = structure_map[structure_choice]

                if "business_structure" in preview_profile["missing"] and not st.session_state.business_structure:
                    st.info(t("select_structure_prompt"))
                else:
                    state = orchestrator(goal, st.session_state.business_structure)
                    verified = state["verified"]
                    roadmap = state["roadmap"]

                    if state["business_structure"]:
                        st.caption(f"{t('personalized_for')}: {state['business_structure'].replace('_', ' ').title()}")

                    if verified.get("flags"):
                        with st.expander(t("verification_notes")):
                            for f in verified["flags"]:
                                st.write(f"- {f}")

                    # ---------- Part 11: AI plain-language summary ----------
                    llm_result = llm_router(build_summary_prompt(goal, state["business_structure"], verified))
                    st.subheader(t("ai_summary_header"))
                    if llm_result["success"]:
                        st.info(llm_result["text"])
                        st.caption(f"Generated by: {llm_result['provider_used']}")
                    else:
                        st.caption(t("ai_summary_unavailable"))

                    st.subheader(t("mandatory_requirements"))
                    for doc in verified.get("mandatory", []):
                        render_requirement(doc)
                    if not verified.get("mandatory"):
                        st.write(t("no_mandatory"))

                    st.subheader(t("conditional_requirements"))
                    for doc in verified.get("conditional", []):
                        render_requirement(doc)
                    if not verified.get("conditional"):
                        st.write(t("no_conditional"))

                    st.subheader(t("optional_requirements"))
                    for doc in verified.get("optional", []):
                        render_requirement(doc)
                    if not verified.get("optional"):
                        st.write(t("no_optional"))

                    if roadmap and roadmap.get("steps"):
                        updated_roadmap = render_roadmap_with_progress(roadmap, editable=True)

                        if st.session_state.user is not None:
                            if st.button(t("save_goal_button")):
                                new_id = save_progress(
                                    user_id=st.session_state.user.id,
                                    goal_text=goal,
                                    business_structure=state["business_structure"],
                                    roadmap=updated_roadmap,
                                )
                                st.session_state.active_goal_id = new_id
                                st.session_state.active_goal_text = goal
                                st.success(t("goal_saved"))
                        else:
                            st.info(t("login_to_save"))
