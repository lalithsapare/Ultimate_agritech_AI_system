import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
from datetime import datetime

st.set_page_config(
    page_title="Farm Health AI Dashboard",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root {
    --bg: #0b1220;
    --panel: rgba(17, 25, 40, 0.82);
    --panel-2: rgba(15, 23, 42, 0.92);
    --border: rgba(148, 163, 184, 0.18);
    --text: #e2e8f0;
    --muted: #94a3b8;
    --accent: #14b8a6;
    --accent-2: #22c55e;
    --danger: #ef4444;
    --warning: #f59e0b;
}

.stApp {
    background: radial-gradient(circle at top left, #123b33 0%, #0b1220 35%, #020617 100%);
    color: var(--text);
}

.main .block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    max-width: 1450px;
}

h1, h2, h3 {
    color: #dffcf6 !important;
    letter-spacing: 0.2px;
}

[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(20, 184, 166, 0.16), rgba(15, 23, 42, 0.95));
    border: 1px solid var(--border);
    padding: 18px;
    border-radius: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.22);
}

[data-testid="stMetricLabel"], [data-testid="stMetricValue"], [data-testid="stMetricDelta"] {
    color: #ecfeff !important;
}

div[data-testid="stPlotlyChart"], div[data-testid="stDeckGlJsonChart"] {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 22px;
    padding: 8px;
    backdrop-filter: blur(12px);
    box-shadow: 0 12px 28px rgba(0,0,0,0.18);
}

[data-testid="stChatMessage"] {
    background: var(--panel-2);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 8px 10px;
    margin-bottom: 10px;
}

.alert-card {
    background: linear-gradient(135deg, rgba(30,41,59,0.85), rgba(15,23,42,0.95));
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 18px;
    margin-bottom: 12px;
}

.hero {
    background: linear-gradient(135deg, rgba(20,184,166,0.16), rgba(34,197,94,0.10), rgba(15,23,42,0.95));
    border: 1px solid rgba(148,163,184,0.16);
    border-radius: 24px;
    padding: 20px 22px;
    margin-bottom: 16px;
    box-shadow: 0 15px 40px rgba(0,0,0,0.22);
}

.caption-pill {
    display:inline-block;
    padding: 6px 12px;
    border-radius: 999px;
    background: rgba(20,184,166,0.14);
    color: #99f6e4;
    border: 1px solid rgba(45,212,191,0.18);
    font-size: 0.85rem;
    margin-bottom: 10px;
}

hr {
    border-color: rgba(148, 163, 184, 0.15);
}
</style>
""", unsafe_allow_html=True)

# Sidebar controls
st.sidebar.title("⚙ Control Center")
crop = st.sidebar.selectbox("Select Crop", ["Sesame", "Paddy", "Maize", "Cotton"])
health_score = st.sidebar.slider("Health Score", 0, 100, 10)
moisture = st.sidebar.slider("Soil Moisture (%)", 0, 100, 24)
ndvi = st.sidebar.slider("Current NDVI", 0.0, 1.0, 0.35, 0.01)
risk_level = "HIGH" if health_score < 30 else "MEDIUM" if health_score < 70 else "LOW"
latitude = st.sidebar.number_input("Latitude", value=19.67, format="%.4f")
longitude = st.sidebar.number_input("Longitude", value=78.53, format="%.4f")
last_refresh = datetime.now().strftime("%d %b %Y, %I:%M %p")

# Header
st.markdown(f"""
<div class='hero'>
    <div class='caption-pill'>LIVE FARM MONITORING</div>
    <h1>🌾 Farm Health Dashboard</h1>
    <p style='color:#cbd5e1; font-size:1rem;'>Premium AI dashboard for crop health, soil diagnostics, weather relationship, risk analytics, smart alerts, and assistant guidance. Latest dashboard status refresh: <b>{last_refresh}</b>.</p>
</div>
""", unsafe_allow_html=True)

# KPI Cards
col1, col2, col3, col4 = st.columns(4)
col1.metric("🌱 Crop", crop)
col2.metric("📊 Health Score", f"{health_score}/100", delta=f"{health_score-90}")
col3.metric("⚠ Risk Level", risk_level)
col4.metric("💧 Soil Moisture", f"{moisture}%")

st.markdown("## Analytics Overview")

# NDVI trend
ndvi_data = pd.DataFrame({
    "Days": list(range(1, 11)),
    "NDVI": [0.60, 0.58, 0.55, 0.50, 0.45, 0.42, 0.40, 0.39, 0.38, ndvi]
})
fig_ndvi = px.line(
    ndvi_data,
    x="Days",
    y="NDVI",
    title="NDVI Trend (Crop Health)",
    markers=True,
)
fig_ndvi.update_traces(line_color="#14b8a6", marker=dict(size=8))
fig_ndvi.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

# Weather + yield graph
weather_yield = pd.DataFrame({
    "Rainfall": [50, 80, 100, 120, 140],
    "Yield": [10, 15, 20, 23, 25]
})
fig_weather = px.scatter(
    weather_yield,
    x="Rainfall",
    y="Yield",
    size="Yield",
    color="Yield",
    title="Yield vs Rainfall",
    color_continuous_scale="Viridis",
)
fig_weather.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

# NPK chart
npk = pd.DataFrame({
    "Nutrient": ["Nitrogen", "Phosphorus", "Potassium"],
    "Value": [42, 22, 24]
})
fig_npk = px.bar(
    npk,
    x="Nutrient",
    y="Value",
    color="Nutrient",
    title="Soil Nutrient Levels",
    color_discrete_sequence=["#22c55e", "#eab308", "#3b82f6"],
)
fig_npk.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

# Gauge
fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=health_score,
    title={'text': "Health Score"},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': "#ef4444" if health_score < 30 else "#f59e0b" if health_score < 70 else "#22c55e"},
        'steps': [
            {'range': [0, 30], 'color': 'rgba(239,68,68,0.35)'},
            {'range': [30, 70], 'color': 'rgba(245,158,11,0.35)'},
            {'range': [70, 100], 'color': 'rgba(34,197,94,0.35)'}
        ]
    }
))
fig_gauge.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")

row1_col1, row1_col2 = st.columns([1.5, 1])
with row1_col1:
    st.plotly_chart(fig_ndvi, use_container_width=True)
with row1_col2:
    st.plotly_chart(fig_gauge, use_container_width=True)

row2_col1, row2_col2 = st.columns(2)
with row2_col1:
    st.plotly_chart(fig_weather, use_container_width=True)
with row2_col2:
    st.plotly_chart(fig_npk, use_container_width=True)

st.markdown("## Smart Alerts")

if health_score < 30:
    st.error("🚨 High Risk: Immediate action required")
if moisture < 30:
    st.warning("⚠ Soil moisture low - irrigation needed")
if ndvi < 0.4:
    st.warning("⚠ Crop stress detected")
if health_score >= 70 and moisture >= 40 and ndvi >= 0.6:
    st.success("✅ Crop health looks stable and field conditions are favorable")

st.markdown("## Field Location")
map_df = pd.DataFrame({"lat": [latitude], "lon": [longitude]})
st.pydeck_chart(pdk.Deck(
    map_style="mapbox://styles/mapbox/dark-v10",
    initial_view_state=pdk.ViewState(
        latitude=latitude,
        longitude=longitude,
        zoom=8,
        pitch=40,
    ),
    layers=[
        pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position='[lon, lat]',
            get_color='[20, 184, 166, 220]',
            get_radius=18000,
            pickable=True,
        )
    ],
    tooltip={"text": f"Crop: {crop}\nRisk: {risk_level}\nMoisture: {moisture}%"}
))

st.markdown("## AI Agronomy Chat")
user_input = st.chat_input("Ask about disease, irrigation, NPK, or crop stress...")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        ("assistant", f"Hello! I am your farm assistant. Current crop is {crop}, health score is {health_score}/100, and risk level is {risk_level}.")
    ]

if user_input:
    question = user_input.lower()
    if "irrigation" in question or "water" in question:
        answer = f"Soil moisture is {moisture}%. Start irrigation planning immediately." if moisture < 30 else f"Soil moisture is {moisture}%. Irrigation is moderate priority right now."
    elif "ndvi" in question or "stress" in question:
        answer = f"Current NDVI is {ndvi:.2f}. This indicates crop stress and likely reduced vigor." if ndvi < 0.4 else f"Current NDVI is {ndvi:.2f}. Crop greenness is within a safer range."
    elif "npk" in question or "soil" in question:
        answer = "Nitrogen is strongest at 42, while phosphorus and potassium are lower. Balanced fertilization and soil testing are recommended."
    elif "risk" in question:
        answer = f"Risk level is {risk_level}. The score combines crop health, soil moisture, and NDVI trend behavior."
    else:
        answer = "Based on the dashboard, focus first on moisture management, regular field scouting, and correcting nutrient imbalance to improve crop health."

    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("assistant", answer))

for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(msg)

st.markdown("## Recommendations")
recommendations = [
    "Increase irrigation frequency if soil moisture remains below 30%.",
    "Inspect leaves for visible stress symptoms because NDVI is trending down.",
    "Improve nutrient planning with follow-up soil testing for phosphorus and potassium.",
    "Track dashboard values daily to compare recovery after intervention."
]
for item in recommendations:
    st.markdown(f"- {item}")
