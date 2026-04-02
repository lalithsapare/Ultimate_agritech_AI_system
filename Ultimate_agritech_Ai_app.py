import os
import re
import streamlit as st
from google import genai

st.set_page_config(
    page_title="🌾 AgriVision AI Chat",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
.info-box {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    color: #166534;
    padding: 1rem;
    border-radius: 14px;
    margin-bottom: 1rem;
}
.footer {
    text-align: center;
    color: #6b7280;
    padding: 18px;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "district" not in st.session_state:
    st.session_state.district = "Hyderabad"
if "season" not in st.session_state:
    st.session_state.season = "Kharif"
if "language" not in st.session_state:
    st.session_state.language = "English"

def get_gemini_api_key():
    """Get API key from secrets or environment variables"""
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

def get_gemini_client():
    """Initialize Gemini client with error handling"""
    api_key = get_gemini_api_key()
    if not api_key:
        return None, "❌ Missing GEMINI_API_KEY. Add it in Streamlit secrets."
    
    try:
        client = genai.Client(api_key=api_key)
        # Test connection
        client.models.generate_content(model="gemini-2.5-flash", contents="test")
        return client, None
    except Exception as e:
        return None, f"❌ Client error: {str(e)}"

def format_gemini_error(error_text):
    """Format common Gemini errors for users"""
    text = str(error_text).lower()
    
    if "429" in text or "quota" in text:
        return "⚠️ Rate limit exceeded. Please wait 1-2 minutes and try again."
    if "invalid" in text and "api" in text:
        return "❌ Invalid API key. Check your GEMINI_API_KEY."
    if "not found" in text and "model" in text:
        return "❌ Gemini model unavailable. Try a different API key."
    
    return f"🤖 Chatbot error: {str(error_text)}"

def get_gemini_reply(client, user_text, district, season, language):
    """Get AI response with system context"""
    system_prompt = f"""
You are AgriVision AI, Telangana's agriculture chatbot.

📍 District: {district}
🌾 Season: {season}
🗣️ Language: {language}

RULES:
1. Answer in SIMPLE farmer language
2. Focus on: crops, irrigation, fertilizer, soil, pests, diseases
3. If user writes in Telugu → reply in Telugu
4. Structure answers as: Problem → Cause → Solution → Next step
5. Be practical and actionable
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=system_prompt + "\n\nUser: " + user_text,
            config=genai.types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=800
            )
        )
        return response.text.strip() if response.text else "No response generated."
    except Exception as e:
        return format_gemini_error(e)

# Sidebar
st.sidebar.title("🌾 AgriVision AI")
st.session_state.district = st.sidebar.selectbox(
    "📍 District",
    ["Hyderabad", "Ranga Reddy", "Medchal", "Sangareddy", "Mahabubnagar", "Nalgonda"]
)
st.session_state.season = st.sidebar.selectbox(
    "🌾 Season", 
    ["Kharif", "Rabi", "Summer"]
)
st.session_state.language = st.sidebar.selectbox(
    "🗣️ Language",
    ["English", "Telugu"]
)

# Status
st.sidebar.markdown("### 🔧 Status")
client, client_error = get_gemini_client()
if client:
    st.sidebar.success("✅ Gemini Ready")
else:
    st.sidebar.error(client_error)

# Quick questions
st.sidebar.markdown("### 🚀 Quick Questions")
quick_questions = [
    "Best Kharif crop for Telangana?",
    "Maize irrigation schedule?",
    "Paddy yellow leaves?",
    "Cotton fertilizer?",
    "Soil fertility tips?",
    "Chilli leaf spot?"
]
for q in quick_questions:
    if st.sidebar.button(q, key=f"quick_{q[:20]}"):
        st.session_state.chat_history.append({"role": "user", "content": q})
        st.rerun()

# Main UI
st.markdown(f"""
<div class="hero-dashboard">
    <div style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:1rem;'>
        <div>
            <h1 class="farm-title">AgriVision AI Chat</h1>
            <p class="subtitle">Telangana farming assistant powered by Gemini 2.5</p>
        </div>
        <div style='text-align:right;'>
            <div><span class="badge-good">Live Chat</span></div>
            <div style='margin-top:0.5rem;font-size:0.95rem;'>
                📍 {st.session_state.district} | 🌾 {st.session_state.season} | 🗣️ {st.session_state.language}
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-card">
    <h3>💬 Ask about farming</h3>
    <p>Crop planning • Irrigation • Fertilizer • Soil health • Pests • Diseases • Weather advice</p>
</div>
""", unsafe_allow_html=True)

if client_error:
    st.error(client_error)

# Chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
prompt = st.chat_input("Ask your agriculture question... 🌾")

if prompt:
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("🤖 AgriVision AI is thinking..."):
            if client:
                reply = get_gemini_reply(
                    client,
                    prompt,
                    st.session_state.district,
                    st.session_state.season,
                    st.session_state.language
                )
            else:
                reply = client_error or "Chatbot unavailable."
            st.markdown(reply)
    
    st.session_state.chat_history.append({"role": "assistant", "content": reply})

# Footer
st.markdown("""
<div class="footer">
    🌾 AgriVision AI Chat | Streamlit + Gemini 2.5 Flash | Deployment Ready
</div>
""", unsafe_allow_html=True)
