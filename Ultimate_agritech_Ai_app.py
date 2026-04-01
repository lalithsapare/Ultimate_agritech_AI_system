import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Farm Health AI Dashboard",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Theme / Styling ----------
st.markdown("""
<style>
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 1rem;
    padding-left: 1.2rem;
    padding-right: 1.2rem;
}
.main-title {
    font-size: 2.2rem;
    font-weight: 800;
    color: #2e7d32;
    margin-bottom: 0rem;
}
.sub-title {
    font-size: 0.95rem;
    color: #6b7280;
    margin-top: -0.3rem;
    margin-bottom: 1rem;
}
.card {
    background: #ffffff;
    padding: 1rem 1rem;
    border-radius: 16px;
    box-shadow: 0 1px 10px rgba(0,0,0,0.06);
    border: 1px solid #edf2e9;
}
.metric-title {
    font-size: 0.85rem;
    color: #6b7280;
    font-weight: 600;
}
.metric-value {
    font-size: 2rem;
    font-weight: 800;
    color: #111827;
}
.metric-delta-up {
    color: #2e7d32;
    font-size: 0.85rem;
    font-weight: 600;
}
.metric-delta-down {
    color: #c62828;
    font-size: 0.85rem;
    font-weight: 600;
}
.metric-delta-neutral {
    color: #b26a00;
    font-size: 0.85rem;
    font-weight: 600;
}
.section-head {
    font-size: 1.15rem;
    font-weight: 700;
    margin-top: 0.5rem;
    margin-bottom: 0.6rem;
    color: #1f2937;
}
.small-note {
    color: #6b7280;
    font-size: 0.85rem;
}
.badge-green {
    background-color: #e8f5e9;
    color: #2e7d32;
    padding: 0.2rem 0.6rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 700;
}
.badge-red {
    background-color: #ffebee;
    color: #c62828;
    padding: 0.2rem 0.6rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 700;
}
.badge-orange {
    background-color: #fff3e0;
    color: #ef6c00;
    padding: 0.2rem 0.6rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 700;
}
div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #edf2e9;
    padding: 12px 14px;
    border-radius: 16px;
    box-shadow: 0 1px 10px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 🌾 Agri Vision AI")
    st.caption("Farm Health Dashboard")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Crop Monitor",
            "Soil Analysis",
            "Weather Intel",
            "Disease Detection",
            "Yield Prediction",
            "Alert Center",
            "Reports",
            "Settings",
        ],
    )

    st.markdown("---")
    st.markdown("### Farm Filters")
    farm_name = st.selectbox("Farm", ["Hyderabad Demo Farm", "Ranga Reddy Farm", "Medchal Farm"])
    crop_type = st.selectbox("Crop", ["Rice", "Maize", "Tomato", "Cotton", "Chilli"])
    season = st.selectbox("Season", ["Kharif", "Rabi", "Summer"])
    st.markdown("---")
    st.success("AI Models Active")

# ---------- Mock Data ----------
trend_df = pd.DataFrame({
    "Day": [f"Day {i}" for i in range(1, 31)],
    "Health Index": [72, 74, 73, 75, 77, 76, 78, 79, 80, 78, 81, 83, 82, 84, 85, 86, 84, 87, 88, 89, 87, 90, 91, 89, 92, 93, 91, 94, 95, 96]
})

zone_df = pd.DataFrame({
    "Zone": ["Healthy", "Moderate Stress", "High Risk", "Critical"],
    "Value": [52, 28, 14, 6]
})

alerts_df = pd.DataFrame({
    "Field / Zone": ["Zone A - Block 3", "Zone B - Block 1", "Zone C - Block 2", "Zone D - Block 5"],
    "Issue Detected": ["Leaf Blight (Alternaria)", "Nitrogen Deficiency", "Aphid Infestation", "Overwatering Detected"],
    "Severity": ["Critical", "Warning", "Warning", "Stable"],
    "AI Confidence": ["94.2%", "87.8%", "79.3%", "91.0%"],
    "Detected": ["2 hrs ago", "5 hrs ago", "1 day ago", "2 days ago"]
})

pred_df = pd.DataFrame({
    "Crop": ["Rice", "Maize", "Tomato"],
    "Forecast": ["Healthy growth", "Moderate stress", "High disease risk"],
    "Confidence": [96, 73, 88]
})

soil_df = pd.DataFrame({
    "Nutrient": ["Nitrogen (N)", "Phosphorus (P)", "Potassium (K)", "pH Level", "Organic Carbon"],
    "Level": [72, 45, 88, 64, 22],
    "Value": ["72 kg/ha", "45 kg/ha", "88 kg/ha", "6.4 (Optimal)", "22%"]
})

# ---------- Header ----------
st.markdown('<div class="main-title">Agri Vision AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Real-time crop intelligence and farm health monitoring dashboard</div>', unsafe_allow_html=True)

top_a, top_b, top_c = st.columns([2, 1, 1])
with top_a:
    st.markdown(f"**Farm:** {farm_name}  |  **Crop:** {crop_type}  |  **Season:** {season}")
with top_b:
    st.markdown('<span class="badge-green">AI Active</span>', unsafe_allow_html=True)
with top_c:
    st.markdown("**Location:** Hyderabad")

# ---------- Pages ----------
if page == "Dashboard":
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Crop Health Score", "87.4%", "+3.2%")
    m2.metric("Active Alerts", "7", "+2")
    m3.metric("Yield Forecast", "4.2 t/ha", "+8.5%")
    m4.metric("Soil Moisture", "62%", "Optimal")

    st.markdown("### Farm Health Analytics")

    c1, c2 = st.columns([2, 1])

    with c1:
        fig_line = px.line(
            trend_df,
            x="Day",
            y="Health Index",
            markers=False,
            title="Crop Health Trend - Last 30 Days"
        )
        fig_line.update_traces(line=dict(color="#2e7d32", width=3))
        fig_line.update_layout(
            height=360,
            template="plotly_white",
            margin=dict(l=10, r=10, t=50, b=10),
            title_font=dict(size=18),
        )
        st.plotly_chart(fig_line, use_container_width=True)

    with c2:
        fig_pie = px.pie(
            zone_df,
            names="Zone",
            values="Value",
            hole=0.6,
            title="Zone Distribution",
            color="Zone",
            color_discrete_map={
                "Healthy": "#43a047",
                "Moderate Stress": "#fb8c00",
                "High Risk": "#ef5350",
                "Critical": "#b71c1c"
            }
        )
        fig_pie.update_layout(
            height=360,
            template="plotly_white",
            margin=dict(l=10, r=10, t=50, b=10),
            title_font=dict(size=18),
            legend_title_text=""
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("### Active Field Alerts")
    def color_severity(val):
        if val == "Critical":
            return "background-color: #ffebee; color: #c62828; font-weight: 700;"
        elif val == "Warning":
            return "background-color: #fff3e0; color: #ef6c00; font-weight: 700;"
        else:
            return "background-color: #e8f5e9; color: #2e7d32; font-weight: 700;"

    st.dataframe(
        alerts_df.style.map(color_severity, subset=["Severity"]),
        use_container_width=True,
        hide_index=True
    )

    b1, b2 = st.columns(2)

    with b1:
        st.markdown("### AI Predictions")
        for _, row in pred_df.iterrows():
            st.markdown(f"**{row['Crop']}** — {row['Forecast']}")
            st.progress(int(row["Confidence"]))
            st.caption(f"Confidence: {row['Confidence']}%")

    with b2:
        st.markdown("### Soil Nutrient Levels")
        for _, row in soil_df.iterrows():
            st.write(f"**{row['Nutrient']}** — {row['Value']}")
            st.progress(int(row["Level"]))

elif page == "Crop Monitor":
    st.markdown("## Crop Monitor")
    st.info("Track crop growth, canopy vigor, NDVI patterns, and field-level health conditions.")
    crop_health = pd.DataFrame({
        "Zone": ["A", "B", "C", "D", "E"],
        "Health Score": [92, 78, 70, 88, 81]
    })
    fig = px.bar(crop_health, x="Zone", y="Health Score", color="Health Score",
                 color_continuous_scale="Greens", title="Zone-wise Crop Health")
    st.plotly_chart(fig, use_container_width=True)

elif page == "Soil Analysis":
    st.markdown("## Soil Analysis")
    st.info("Analyze nutrient levels, pH balance, organic carbon, and fertilizer recommendations.")
    fig = go.Figure(go.Scatterpolar(
        r=[72, 45, 88, 64, 22],
        theta=["Nitrogen", "Phosphorus", "Potassium", "pH", "Organic Carbon"],
        fill='toself',
        line_color="#2e7d32"
    ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True)), height=500, title="Soil Health Radar")
    st.plotly_chart(fig, use_container_width=True)

elif page == "Weather Intel":
    st.markdown("## Weather Intel")
    st.info("Monitor temperature, humidity, rainfall, and irrigation planning indicators.")
    weather_df = pd.DataFrame({
        "Day": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        "Temp": [31, 32, 34, 33, 30, 29, 31],
        "Rainfall": [0, 2, 5, 1, 0, 12, 3]
    })
    fig = px.line(weather_df, x="Day", y=["Temp", "Rainfall"], markers=True, title="7-Day Weather Signals")
    st.plotly_chart(fig, use_container_width=True)

elif page == "Disease Detection":
    st.markdown("## Disease Detection")
    st.warning("Upload crop leaf images and run CNN-based disease classification.")
    uploaded_file = st.file_uploader("Upload crop image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Crop Image", use_container_width=True)
        st.success("Prediction: Leaf Blight detected with 94.2% confidence.")

elif page == "Yield Prediction":
    st.markdown("## Yield Prediction")
    st.info("Forecast expected yield using crop, weather, and soil features.")
    area = st.number_input("Farm area (acres)", min_value=1.0, value=5.0)
    ndvi = st.slider("Average NDVI", 0.0, 1.0, 0.72)
    moisture = st.slider("Soil Moisture (%)", 0, 100, 62)
    if st.button("Predict Yield"):
        pred = round((area * ndvi * moisture) / 8.5, 2)
        st.success(f"Estimated Yield Forecast: {pred} tons")

elif page == "Alert Center":
    st.markdown("## Alert Center")
    st.error("Critical and warning alerts are listed below.")
    st.dataframe(alerts_df, use_container_width=True, hide_index=True)

elif page == "Reports":
    st.markdown("## Reports")
    st.info("Download and review weekly or monthly AI farm reports.")
    csv = alerts_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Alerts Report CSV", data=csv, file_name="farm_alerts_report.csv", mime="text/csv")

elif page == "Settings":
    st.markdown("## Settings")
    st.text_input("Farm Name", value=farm_name)
    st.selectbox("Preferred Alert Mode", ["SMS", "Email", "Dashboard Only"])
    st.slider("Critical Alert Threshold", 50, 100, 80)
    st.checkbox("Enable AI Recommendations", value=True)
    st.button("Save Settings")
