import os
import re
import streamlit as st

# =========================
# SAFE IMPORT FOR GEMINI SDK
# =========================
GENAI_AVAILABLE = True
GENAI_IMPORT_ERROR = ""

try:
    from google import genai
    from google.genai import types
except Exception as e:
    GENAI_AVAILABLE = False
    GENAI_IMPORT_ERROR = str(e)


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AgriVision AI Chat",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# STYLES
# =========================
st.markdown("""
<style>
.main {background: linear-gradient(135deg, #0f1f0f 0%, #1a2f1a 100%);}
.block-container {padding-top: 1rem; padding-bottom: 1rem;}
[data-testid="stSidebar"] {background: linear-gradient(180deg, #14532d 0%, #166534 50%, #14532d 100%);}
[data-testid="stSidebar"] * {color: white !important;}
.hero-dashboard {
    background: linear-gradient(90deg, #166534, #22c55e, #16a34a);
    color: white;
    padding: 1.8rem 1.5rem;
    border-radius: 24px;
    box-shadow: 0 12px 32px rgba(22, 101, 52, 0.3);
    margin-bottom: 1.5rem;
}
.info-card {
    background: rgba(255,255,255,0.97);
    border-radius: 20px;
    padding: 1.2rem;
    border: 1px solid #dcfce7;
    box-shadow: 0 8px 24px rgba(0,0,0,0.12);
    margin-bottom: 1rem;
}
.farm-title {
    font-size: 2.3rem;
    font-weight: 800;
    margin: 0;
}
.subtitle {
    margin: 0.4rem 0 0 0;
    font-size: 1.05rem;
    opacity: 0.96;
}
.badge-good {
    background: #dcfce7;
    color: #166534;
    padding: 0.35rem 0.75rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 700;
}
.small-muted {
    color: #64748b;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# =========================
# SESSION STATE
# =========================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "district" not in st.session_state:
    st.session_state.district = "Hyderabad"

if "season" not in st.session_state:
    st.session_state.season = "Kharif"

if "language" not in st.session_state:
    st.session_state.language = "English"

# =========================
# API KEY
# =========================
def get_gemini_api_key():
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"].strip()
    except Exception:
        pass

    key1 = os.getenv("GEMINI_API_KEY", "").strip()
    if key1:
        return key1

    key2 = os.getenv("GOOGLE_API_KEY", "").strip()
    if key2:
        return key2

    return ""

# =========================
# CLIENT
# =========================
def get_gemini_client():
    if not GENAI_AVAILABLE:
        return None, (
            "Google Gen AI SDK is not installed.\n\n"
            "Fix:\n"
            "1. Create a requirements.txt file.\n"
            "2. Add:\n"
            "   streamlit\n"
            "   google-genai\n"
            "3. Rebuild/redeploy the app.\n\n"
            f"Import error: {GENAI_IMPORT_ERROR}"
        )

    api_key = get_gemini_api_key()
    if not api_key:
        return None, "Missing GEMINI_API_KEY. Add it in Streamlit secrets or environment variables."

    try:
        client = genai.Client(api_key=api_key)
        return client, None
    except Exception as e:
        return None, f"Failed to initialize Gemini client: {e}"

# =========================
# ERROR FORMATTER
# =========================
def format_gemini_error(error_text):
    text = str(error_text)

    if "429" in text or "quota" in text.lower():
        retry_match = re.search(r"retry in\s+([\d\.]+)s", text, re.IGNORECASE)
        retry_hint = f" Retry after about {retry_match.group(1)} seconds." if retry_match else ""

        return (
            "Gemini quota exceeded for this project."
            + retry_hint
            + "\n\nPossible reasons:\n"
            "- Free-tier limit finished.\n"
            "- Billing not enabled.\n"
            "- Requests per minute/day exceeded.\n\n"
            "Fix:\n"
            "1. Check your Google AI Studio project.\n"
            "2. Enable billing if needed.\n"
            "3. Use a valid API key from a project with quota."
        )

    if "invalid" in text.lower() and "api" in text.lower():
        return "Invalid Gemini API key. Please verify GEMINI_API_KEY."

    if "not found" in text.lower() and "model" in text.lower():
        return "The Gemini model is not available for your project or API version."

    return f"Gemini chatbot error: {text}"

# =========================
# CHATBOT REPLY
# =========================
def get_gemini_reply(user_text, district, season, language):
    client, error = get_gemini_client()
    if error:
        return error

    system_instruction = f"""
You are AgriVision AI, an agriculture chatbot for India.

User district: {district}, Telangana
Season: {season}
Preferred language: {language}

Rules:
- Give practical farming guidance.
- Use simple beginner-friendly language.
- Focus on crop planning, irrigation, fertilizer, pests, disease, soil health, and farm management.
- Keep answers clear and useful.
- If user writes in Telugu, reply in simple Telugu.
- Otherwise reply in English.
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_text,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.4,
                max_output_tokens=700
            )
        )

        if hasattr(response, "text") and response.text:
            return response.text.strip()

        return "No response generated."

    except Exception as e:
        return format_gemini_error(str(e))

# =========================
# UI
# =========================
st.sidebar.title("🌾 AgriVision AI")

st.session_state.district = st.sidebar.selectbox(
    "District",
    ["Hyderabad", "Ranga Reddy", "Medchal", "Sangareddy", "Mahabubnagar"]
)

st.session_state.season = st.sidebar.selectbox(
    "Season",
    ["Kharif", "Rabi", "Summer"]
)

st.session_state.language = st.sidebar.selectbox(
    "Language",
    ["English", "Telugu"]
)

st.sidebar.markdown("### Setup Status")
if GENAI_AVAILABLE:
    st.sidebar.success("google-genai import OK")
else:
    st.sidebar.error("google-genai not installed")

if get_gemini_api_key():
    st.sidebar.success("API key found")
else:
    st.sidebar.error("API key missing")

st.sidebar.markdown("### Example prompts")
st.sidebar.markdown("""
- Best crop for Telangana in Kharif?
- How to manage yellow leaves in paddy?
- Give me drip irrigation advice for chilli.
- Which fertilizer is best at vegetative stage?
- How do I improve soil fertility naturally?
- Tell me a weekly plan for maize crop.
""")

st.markdown(
    f"""
    <div class="hero-dashboard">
        <div style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:1rem;'>
            <div>
                <h1 class="farm-title">AgriVision AI Chat</h1>
                <p class="subtitle">Simple Gemini-powered farming assistant</p>
            </div>
            <div style='text-align:right;'>
                <div><span class="badge-good">gemini-2.5-flash</span></div>
                <div style='margin-top:0.5rem;font-size:0.95rem;'>
                    📍 {st.session_state.district} | 🌾 {st.session_state.season} | 🗣️ {st.session_state.language}
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="info-card">
        Ask any farming question about crops, irrigation, fertilizers, pests, diseases, soil health, or seasonal planning.
    </div>
    """,
    unsafe_allow_html=True
)

if not GENAI_AVAILABLE:
    st.error(
        "Import failed for google-genai.\n\n"
        "Add this to requirements.txt:\n"
        "streamlit\n"
        "google-genai"
    )

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask your agriculture question...")

if prompt:
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = get_gemini_reply(
                user_text=prompt,
                district=st.session_state.district,
                season=st.session_state.season,
                language=st.session_state.language
            )
            st.markdown(reply)

    st.session_state.chat_history.append({"role": "assistant", "content": reply})

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#6b7280;padding:18px;font-size:14px;'>🌾 AgriVision AI Chat | Streamlit + Gemini 2.5 Flash</div>",
    unsafe_allow_html=True
)
