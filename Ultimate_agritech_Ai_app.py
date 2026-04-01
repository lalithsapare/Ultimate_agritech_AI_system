import os
import re
import time
import streamlit as st
from google import genai
from google.genai import types

# =========================
# STREAMLIT PAGE SETTINGS
# =========================
st.set_page_config(
    page_title="AgriVision AI Chat",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CUSTOM CSS
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
.small-muted {
    color: #64748b;
    font-size: 0.9rem;
}
.badge-good {
    background: #dcfce7;
    color: #166534;
    padding: 0.35rem 0.75rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 700;
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

    env_key = os.getenv("GEMINI_API_KEY", "").strip()
    if env_key:
        return env_key

    env_key_2 = os.getenv("GOOGLE_API_KEY", "").strip()
    if env_key_2:
        return env_key_2

    return ""

# =========================
# CLIENT
# =========================
def get_gemini_client():
    api_key = get_gemini_api_key()
    if not api_key:
        return None, "Missing API key. Add GEMINI_API_KEY in Streamlit secrets or environment variables."
    try:
        client = genai.Client(api_key=api_key)
        return client, None
    except Exception as e:
        return None, f"Failed to initialize Gemini client: {e}"

# =========================
# ERROR FORMATTER
# =========================
def format_gemini_error(error_text: str) -> str:
    text = str(error_text)

    if "429" in text or "quota" in text.lower() or "rate limit" in text.lower():
        retry_match = re.search(r"retry in\s+([\d\.]+)s", text, re.IGNORECASE)
        retry_hint = ""
        if retry_match:
            retry_hint = f" Try again after about {retry_match.group(1)} seconds."

        return (
            "Gemini quota exceeded for this project. "
            "This usually means your Google AI project has no free-tier quota left, "
            "billing is not enabled, or the daily/request/token limit has been reached."
            + retry_hint
            + "\n\nFix:\n"
            "1. Open Google AI Studio.\n"
            "2. Check the project linked to your API key.\n"
            "3. Enable billing or use a project with active quota.\n"
            "4. Then retry."
        )

    if "API key" in text or "invalid" in text.lower():
        return "Invalid Gemini API key. Please check your GEMINI_API_KEY."

    if "not found" in text.lower() and "model" in text.lower():
        return "The selected Gemini model is not available for your current API/version/project."

    return f"Gemini chatbot error: {text}"

# =========================
# CHAT FUNCTION
# =========================
def get_gemini_reply(user_text, district, season, language):
    client, client_error = get_gemini_client()
    if client_error:
        return client_error

    system_prompt = f"""
You are AgriVision AI, a practical agriculture assistant for farmers and beginners in India.

User location: {district}, Telangana, India
Season: {season}
Preferred language: {language}

Rules:
- Reply in simple, beginner-friendly language.
- Focus on agriculture, crops, irrigation, fertilizer, disease prevention, soil health, market planning, and weather-based farm advice.
- Give practical action steps.
- Keep answers concise but useful.
- If the user asks in English, reply in English.
- If the user asks in Telugu, reply in simple Telugu.
- If something depends on local field conditions, say so clearly.
"""

    messages = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=system_prompt)]
        ),
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_text)]
        )
    ]

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=messages,
            config=types.GenerateContentConfig(
                temperature=0.4,
                max_output_tokens=700
            )
        )

        if hasattr(response, "text") and response.text:
            return response.text.strip()

        return "No response generated from Gemini."

    except Exception as e:
        return format_gemini_error(str(e))

# =========================
# HEADER
# =========================
def render_header():
    st.markdown(
        f"""
        <div class="hero-dashboard">
            <div style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:1rem;'>
                <div>
                    <h1 class="farm-title">AgriVision AI Chat</h1>
                    <p class="subtitle">Gemini-powered agriculture assistant</p>
                </div>
                <div style='text-align:right;'>
                    <div><span class="badge-good">Gemini 2.5 Flash</span></div>
                    <div style='margin-top:0.5rem;font-size:0.95rem;'>
                        📍 {st.session_state.district} | 🌾 {st.session_state.season}
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================
# SIDEBAR
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

api_key_available = bool(get_gemini_api_key())
st.sidebar.markdown("### API Status")
st.sidebar.success("Gemini API key detected") if api_key_available else st.sidebar.error("Gemini API key missing")

st.sidebar.markdown("### Suggested questions")
st.sidebar.markdown("""
- Which crop is suitable for Telangana in Kharif?
- How often should I irrigate maize?
- What fertilizer is good for paddy at early stage?
- My cotton leaves are turning yellow, why?
- How can I improve soil health naturally?
- Tell me a 7-day farm care plan.
""")

# =========================
# MAIN UI
# =========================
render_header()

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("## Agriculture Chat Assistant")
    st.markdown(
        "<div class='info-card'>Ask crop, soil, irrigation, disease, fertilizer, market, or seasonal farming questions.</div>",
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div class='info-card'>
            <div class='small-muted'>Current setup</div>
            <h4 style='margin-top:8px;color:#166534;'>District: {st.session_state.district}</h4>
            <h4 style='margin-top:8px;color:#166534;'>Season: {st.session_state.season}</h4>
            <h4 style='margin-top:8px;color:#166534;'>Language: {st.session_state.language}</h4>
        </div>
        """,
        unsafe_allow_html=True
    )

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_prompt = st.chat_input("Ask your farming question here...")

if user_prompt:
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = get_gemini_reply(
                user_text=user_prompt,
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
