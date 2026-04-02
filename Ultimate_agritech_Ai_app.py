import os
import io
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from openai import OpenAI
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

st.set_page_config(page_title="AgriVision AI Pro", page_icon="🌾", layout="wide")

st.markdown("""
<style>
.main {background: linear-gradient(135deg, #08111f 0%, #020817 100%);}
[data-testid="stSidebar"] {background: linear-gradient(180deg, #166534 0%, #14532d 100%);}
[data-testid="stSidebar"] * {color: white !important;}
.hero {background: linear-gradient(90deg, #166534, #22c55e); color: white; padding: 24px; border-radius: 22px; margin-bottom: 16px;}
.card {background: rgba(255,255,255,0.96); padding: 16px; border-radius: 18px; margin-bottom: 12px;}
</style>
""", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "latest_result" not in st.session_state:
    st.session_state.latest_result = None


def get_secret(name):
    try:
        return str(st.secrets[name]).strip()
    except Exception:
        return os.getenv(name, "").strip()


def fetch_weather(city):
    api_key = get_secret("OPENWEATHER_API_KEY")
    if not api_key:
        return None

    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"q": city, "appid": api_key, "units": "metric"}
        r = requests.get(url, params=params, timeout=15)
        data = r.json()
        if "main" not in data:
            return None
        return {
            "temperature": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "rainfall": data.get("rain", {}).get("1h", 0)
        }
    except Exception:
        return None


def call_backend(payload):
    base_url = get_secret("FASTAPI_URL") or "http://127.0.0.1:8000"
    try:
        r = requests.post(f"{base_url}/predict/all", json=payload, timeout=30)
        return r.json()
    except Exception as e:
        st.error(f"Backend connection failed: {e}")
        return None


def get_ai_reply(user_text, latest_result, district, season):
    api_key = get_secret("OPENROUTER_API_KEY")
    if not api_key:
        return "OpenRouter API key missing. Add OPENROUTER_API_KEY in secrets."

    try:
        client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        context = f"District: {district}, Season: {season}, Farm Result: {latest_result}"
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an agriculture AI assistant. Give short, practical, farmer-friendly advice."},
                {"role": "user", "content": f"{context}\n\nQuestion: {user_text}"}
            ],
            temperature=0.4,
            extra_headers={
                "HTTP-Referer": "https://agrivisionai.streamlit.app",
                "X-Title": "AgriVision AI Pro"
            }
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI error: {e}"


def build_pdf_report(result):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 800, "AgriVision AI Farm Report")
    c.setFont("Helvetica", 11)

    lines = [
        f"Crop Recommendation: {result['crop']['name']}",
        f"Crop Confidence: {result['crop']['confidence']}%",
        f"Predicted Yield: {result['yield']['value']} t/ha",
        f"Irrigation Need: {result['irrigation']['need']}",
        f"Irrigation Action: {result['irrigation']['action']}",
        f"NDVI: {result['ndvi']}",
        f"Health Score: {result['health']}",
        f"Fertilizer Advice: {result['fertilizer']['advice']}",
    ]

    y = 760
    for line in lines:
        c.drawString(50, y, line)
        y -= 22

    c.save()
    buffer.seek(0)
    return buffer


st.markdown("<div class='hero'><h1>🌾 AgriVision AI Pro</h1><p>FastAPI + Weather + XAI + AI Assistant + Report Generation</p></div>", unsafe_allow_html=True)

st.sidebar.title("AgriVision AI Pro")
page = st.sidebar.radio("Choose Module", ["Smart Advisor", "AI Assistant"])

district = st.sidebar.selectbox("District", ["Hyderabad", "Ranga Reddy", "Medchal", "Sangareddy"])
season = st.sidebar.selectbox("Season", ["Kharif", "Rabi"])

weather = fetch_weather(district)

if page == "Smart Advisor":
    st.subheader("Farm Input")

    col1, col2, col3 = st.columns(3)
    with col1:
        n = st.number_input("Nitrogen (N)", value=88.0)
        p = st.number_input("Phosphorus (P)", value=40.0)
        k = st.number_input("Potassium (K)", value=41.0)
        ph = st.number_input("Soil pH", value=6.7)

    with col2:
        if weather:
            st.success("Real-time weather loaded")
            temperature = st.number_input("Temperature (°C)", value=float(weather["temperature"]), disabled=True)
            humidity = st.number_input("Humidity (%)", value=float(weather["humidity"]), disabled=True)
            rainfall = st.number_input("Rainfall (mm)", value=float(weather["rainfall"]), disabled=True)
        else:
            st.warning("Weather API unavailable, using manual values")
            temperature = st.number_input("Temperature (°C)", value=29.0)
            humidity = st.number_input("Humidity (%)", value=78.0)
            rainfall = st.number_input("Rainfall (mm)", value=12.0)

    with col3:
        soil_moisture = st.number_input("Soil Moisture (%)", value=42.0)
        red_band = st.number_input("Red Band", min_value=0.0, max_value=1.0, value=0.30)
        nir_band = st.number_input("NIR Band", min_value=0.0, max_value=1.0, value=0.72)

    if st.button("Run Full Analysis", use_container_width=True):
        payload = {
            "n": n,
            "p": p,
            "k": k,
            "ph": ph,
            "temperature": temperature,
            "humidity": humidity,
            "rainfall": rainfall,
            "soil_moisture": soil_moisture,
            "red_band": red_band,
            "nir_band": nir_band,
            "season": season
        }

        result = call_backend(payload)
        st.session_state.latest_result = result

        if result:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Crop", result["crop"]["name"], f"{result['crop']['confidence']}%")
            m2.metric("Yield", f"{result['yield']['value']} t/ha")
            m3.metric("NDVI", result["ndvi"])
            m4.metric("Health", f"{result['health']}%")

            c1, c2 = st.columns(2)

            with c1:
                crop_df = pd.DataFrame(result["crop"]["scores"])
                fig1 = px.bar(crop_df, x="crop", y="score", color="score", title="Crop Suitability")
                st.plotly_chart(fig1, use_container_width=True)

                yield_df = pd.DataFrame({
                    "Month": result["yield"]["months"],
                    "Yield": result["yield"]["trend"]
                })
                fig2 = px.line(yield_df, x="Month", y="Yield", markers=True, title="NDVI / Yield Trend View")
                st.plotly_chart(fig2, use_container_width=True)

                radar = go.Figure()
                radar.add_trace(go.Scatterpolar(
                    r=[n, p, k, soil_moisture, humidity],
                    theta=["N", "P", "K", "Moisture", "Humidity"],
                    fill='toself',
                    name='Soil Profile'
                ))
                radar.update_layout(polar=dict(radialaxis=dict(visible=True)), title="Soil Radar Chart")
                st.plotly_chart(radar, use_container_width=True)

            with c2:
                irr_df = pd.DataFrame({
                    "Day": result["irrigation"]["days"],
                    "Liters": result["irrigation"]["liters"]
                })
                fig3 = px.area(irr_df, x="Day", y="Liters", title="Irrigation Plan")
                st.plotly_chart(fig3, use_container_width=True)

                fig4 = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=result["health"],
                    title={'text': "Risk / Health Gauge"},
                    gauge={'axis': {'range': [0, 100]}}
                ))
                st.plotly_chart(fig4, use_container_width=True)

                rain_df = pd.DataFrame({
                    "Metric": ["Yield", "Rainfall"],
                    "Value": [result["yield"]["value"], rainfall]
                })
                fig5 = px.bar(rain_df, x="Metric", y="Value", color="Metric", title="Yield vs Rainfall")
                st.plotly_chart(fig5, use_container_width=True)

            st.subheader("Explainable AI")
            xai_df = pd.DataFrame({
                "Feature": ["N", "P", "K", "Temperature", "Humidity", "pH", "Rainfall"],
                "Importance": [0.24, 0.12, 0.11, 0.19, 0.14, 0.09, 0.11]
            })
            fig_xai = px.bar(xai_df, x="Importance", y="Feature", orientation="h", title="Why this crop was selected")
            st.plotly_chart(fig_xai, use_container_width=True)
            st.info("SHAP-ready section added. Replace demo importance values with real SHAP values from your trained model.")

            pdf = build_pdf_report(result)
            st.download_button(
                "Download Report (PDF)",
                data=pdf,
                file_name="farm_report.pdf",
                mime="application/pdf",
                use_container_width=True
            )

            st.subheader("Voice Assistant")
            st.caption("Add speech-to-text and text-to-speech here for farmer voice interaction.")
            audio_file = st.file_uploader("Upload voice query (wav/mp3)", type=["wav", "mp3"])
            if audio_file:
                st.audio(audio_file)

elif page == "AI Assistant":
    st.subheader("Farm Chatbot with Memory")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask about crop, fertilizer, disease, irrigation or weather...")
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            reply = get_ai_reply(prompt, st.session_state.latest_result, district, season)
            st.markdown(reply)
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
