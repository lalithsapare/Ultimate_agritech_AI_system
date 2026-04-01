import os
from datetime import datetime

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

st.set_page_config(page_title="Agri Vision AI – Telangana Edition", page_icon="🌾", layout="wide")

APP_NAME = "Agri Vision AI – Telangana Edition"

TELANGANA_CROPS = {
    "Kharif": ["Cotton", "Maize", "Paddy", "Red Gram", "Soybean", "Green Gram", "Black Gram", "Jowar", "Bajra", "Sesame", "Groundnut", "Turmeric", "Chilli"],
    "Rabi": ["Paddy", "Maize", "Groundnut", "Bengal Gram", "Black Gram", "Sesame", "Jowar", "Sunflower", "Safflower", "Horse Gram", "Castor"],
    "Summer": ["Paddy", "Maize", "Groundnut", "Sesame", "Sunflower", "Vegetables", "Fodder Maize"],
}

TELANGANA_DISTRICTS = [
    "Adilabad", "Bhadradri Kothagudem", "Hanamkonda", "Hyderabad", "Jagtial", "Jangaon",
    "Jayashankar Bhupalpally", "Jogulamba Gadwal", "Kamareddy", "Karimnagar", "Khammam",
    "Komaram Bheem Asifabad", "Mahabubabad", "Mahabubnagar", "Mancherial", "Medak",
    "Medchal-Malkajgiri", "Mulugu", "Nagarkurnool", "Nalgonda", "Narayanpet", "Nirmal",
    "Nizamabad", "Peddapalli", "Rajanna Sircilla", "Rangareddy", "Sangareddy", "Siddipet",
    "Suryapet", "Vikarabad", "Wanaparthy", "Warangal", "Yadadri Bhuvanagiri"
]

SOIL_TYPES = ["Red Sandy Loam", "Black Cotton Soil", "Alluvial Soil", "Lateritic Soil", "Clay Loam"]
FARM_TYPES = ["Dry land", "Irrigated", "Rain-fed"]

SIMULATION_DEFAULTS = {
    "Dry land": {"soil_moisture": 24.0, "rainfall": 60.0, "humidity": 42.0, "temperature": 33.0, "nitrogen": 42, "phosphorus": 22, "potassium": 24, "ph": 7.2, "ndvi": 0.39},
    "Irrigated": {"soil_moisture": 71.0, "rainfall": 220.0, "humidity": 70.0, "temperature": 28.0, "nitrogen": 74, "phosphorus": 39, "potassium": 44, "ph": 6.8, "ndvi": 0.78},
    "Rain-fed": {"soil_moisture": 49.0, "rainfall": 510.0, "humidity": 76.0, "temperature": 29.0, "nitrogen": 58, "phosphorus": 31, "potassium": 34, "ph": 6.9, "ndvi": 0.58},
}

CROP_GUIDE = {
    "Cotton": {"water": "Medium", "soil": "Black Cotton Soil", "base_yield": 18, "fert": "Balanced NPK with split nitrogen application"},
    "Maize": {"water": "Medium", "soil": "Red Sandy Loam", "base_yield": 32, "fert": "Nitrogen rich basal + top dressing"},
    "Paddy": {"water": "High", "soil": "Clay Loam", "base_yield": 26, "fert": "High nitrogen with zinc attention"},
    "Red Gram": {"water": "Low-Medium", "soil": "Red Sandy Loam", "base_yield": 10, "fert": "Starter phosphorus, low nitrogen"},
    "Soybean": {"water": "Medium", "soil": "Alluvial Soil", "base_yield": 14, "fert": "Rhizobium support + phosphorus"},
    "Green Gram": {"water": "Low", "soil": "Red Sandy Loam", "base_yield": 8, "fert": "Light phosphorus and sulphur"},
    "Black Gram": {"water": "Low", "soil": "Red Sandy Loam", "base_yield": 8, "fert": "Low nitrogen, phosphorus support"},
    "Jowar": {"water": "Low", "soil": "Black Cotton Soil", "base_yield": 14, "fert": "Moderate NPK with micronutrient watch"},
    "Bajra": {"water": "Low", "soil": "Lateritic Soil", "base_yield": 12, "fert": "Low input crop, moderate nitrogen"},
    "Sesame": {"water": "Low", "soil": "Red Sandy Loam", "base_yield": 6, "fert": "Sulphur and phosphorus support"},
    "Groundnut": {"water": "Medium", "soil": "Red Sandy Loam", "base_yield": 16, "fert": "Gypsum, calcium and phosphorus"},
    "Turmeric": {"water": "High", "soil": "Clay Loam", "base_yield": 80, "fert": "Organic matter rich with potassium support"},
    "Chilli": {"water": "Medium", "soil": "Red Sandy Loam", "base_yield": 22, "fert": "Potash and micronutrient support"},
    "Bengal Gram": {"water": "Low", "soil": "Black Cotton Soil", "base_yield": 10, "fert": "Phosphorus dominant recommendation"},
    "Sunflower": {"water": "Low-Medium", "soil": "Alluvial Soil", "base_yield": 9, "fert": "Sulphur and boron support"},
    "Safflower": {"water": "Low", "soil": "Black Cotton Soil", "base_yield": 7, "fert": "Low irrigation, balanced phosphorus"},
    "Horse Gram": {"water": "Low", "soil": "Lateritic Soil", "base_yield": 6, "fert": "Very low input, organic matter preferred"},
    "Castor": {"water": "Low-Medium", "soil": "Black Cotton Soil", "base_yield": 11, "fert": "Moderate NPK with sulphur"},
    "Vegetables": {"water": "Medium-High", "soil": "Clay Loam", "base_yield": 95, "fert": "Crop-specific soluble nutrition"},
    "Fodder Maize": {"water": "Medium", "soil": "Red Sandy Loam", "base_yield": 140, "fert": "Nitrogen rich fodder nutrition"},
}

DISEASE_RULES = {
    "Paddy": [(0.45, 78, "Leaf Blight", 88), (0.60, 70, "Blast Risk", 81)],
    "Cotton": [(0.40, 65, "Leaf Spot", 84), (0.55, 75, "Wilt Risk", 76)],
    "Maize": [(0.42, 70, "Leaf Blight", 85)],
    "Chilli": [(0.48, 72, "Leaf Curl Risk", 82)],
    "Groundnut": [(0.43, 68, "Tikka Disease Risk", 80)],
}


def recommend_crop(season, soil_type, temperature, humidity, rainfall, ph, nitrogen, phosphorus, potassium):
    scored = []
    for crop in TELANGANA_CROPS[season]:
        score = 50
        guide = CROP_GUIDE.get(crop, {})
        if guide.get("soil") == soil_type:
            score += 18
        if crop == "Cotton" and 24 <= temperature <= 32 and rainfall >= 500:
            score += 24
        if crop == "Paddy" and humidity >= 70 and rainfall >= 600:
            score += 25
        if crop == "Maize" and 20 <= temperature <= 32 and 400 <= rainfall <= 900:
            score += 22
        if crop == "Red Gram" and 22 <= temperature <= 34 and rainfall >= 450:
            score += 20
        if crop == "Groundnut" and 22 <= temperature <= 30 and 300 <= rainfall <= 700:
            score += 20
        if crop == "Bengal Gram" and season == "Rabi" and rainfall <= 150:
            score += 25
        if crop == "Sesame" and rainfall <= 500:
            score += 17
        if crop == "Turmeric" and humidity >= 75 and rainfall >= 800:
            score += 18
        if 6.0 <= ph <= 7.8:
            score += 8
        if nitrogen >= 50:
            score += 5
        if phosphorus >= 25:
            score += 5
        if potassium >= 25:
            score += 5
        confidence = min(97, max(68, score))
        scored.append({"crop": crop, "score": score, "confidence": confidence})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:5]


def predict_yield(crop, area, rainfall, temperature, soil_moisture, nitrogen, phosphorus, potassium):
    base = CROP_GUIDE.get(crop, {}).get("base_yield", 12)
    weather_factor = 1 + ((rainfall - 300) / 2000) + ((soil_moisture - 40) / 300)
    nutrient_factor = 1 + ((nitrogen + phosphorus + potassium) - 100) / 500
    temp_penalty = 1 - abs(28 - temperature) / 60
    per_acre = max(3, base * weather_factor * nutrient_factor * temp_penalty)
    total = per_acre * area
    confidence = max(72, min(96, int(78 + (soil_moisture / 10) + (nitrogen + phosphorus + potassium) / 20)))
    return round(per_acre, 2), round(total, 2), confidence


def predict_disease(crop, ndvi, humidity, soil_moisture):
    rules = DISEASE_RULES.get(crop, [])
    for ndvi_limit, humidity_limit, disease_name, conf in rules:
        if ndvi <= ndvi_limit and humidity >= humidity_limit:
            return disease_name, conf
    if soil_moisture < 25 and ndvi < 0.45:
        return "Stress Wilt Risk", 79
    return "Healthy", 91


def weather_label(rainfall, humidity):
    if rainfall >= 700:
        return "High Rainfall"
    if rainfall >= 300:
        return "Moderate Rainfall"
    if humidity < 40:
        return "Dry Weather"
    return "Low Rainfall"


def irrigation_advice(crop, soil_moisture, temperature, rainfall, disease_name):
    if disease_name != "Healthy":
        return "Low", "Reduce irrigation and avoid extra leaf wetness because disease risk is active."
    if soil_moisture < 30 or (temperature > 34 and rainfall < 50):
        return "High", "Immediate irrigation needed; use drip or early morning watering."
    if soil_moisture < 45:
        return "Medium", "Irrigation needed in 1-2 days; protect crop during flowering and grain filling stage."
    return "Low", "Moisture is acceptable; avoid over-irrigation in Telangana field conditions."


def fertilizer_advice(crop, nitrogen, phosphorus, potassium, disease_name):
    low = []
    if nitrogen < 50:
        low.append("nitrogen")
    if phosphorus < 25:
        low.append("phosphorus")
    if potassium < 25:
        low.append("potassium")
    if disease_name != "Healthy":
        return "Reduced usage", "Disease detected, so avoid heavy fertilizer immediately. Use balanced corrective nutrition only after field inspection."
    if not low:
        return "Balanced usage", f"NPK looks acceptable for {crop}. Follow split doses and crop-stage based application."
    return "Targeted usage", f"Low {', '.join(low)} detected. For {crop}: {CROP_GUIDE.get(crop, {}).get('fert', 'Balanced fertilization recommended')}."


def nutrient_health(nitrogen, phosphorus, potassium):
    avg = (nitrogen + phosphorus + potassium) / 3
    if avg >= 55:
        return "Strong"
    if avg >= 35:
        return "Moderate"
    return "Weak"


def ndvi_status(ndvi):
    if ndvi >= 0.70:
        return "Healthy canopy"
    if ndvi >= 0.50:
        return "Moderate canopy"
    return "Weak canopy"


def build_intelligence_layer(crop, disease_name, soil_moisture, rainfall, temperature, nitrogen, phosphorus, potassium):
    irrigation_level, irrigation_text = irrigation_advice(crop, soil_moisture, temperature, rainfall, disease_name)
    fertilizer_level, fertilizer_text = fertilizer_advice(crop, nitrogen, phosphorus, potassium, disease_name)
    if disease_name != "Healthy":
        irrigation_level = "Low"
        fertilizer_level = "Reduced usage"
        irrigation_text = "Reduce irrigation because disease pressure is active and extra moisture can worsen infection."
        fertilizer_text = "Reduce fertilizer usage for now and avoid heavy nitrogen until disease pressure comes down."
    return {
        "irrigation_level": irrigation_level,
        "irrigation_text": irrigation_text,
        "fertilizer_level": fertilizer_level,
        "fertilizer_text": fertilizer_text,
    }


def calculate_risk_and_health(disease_name, ndvi, soil_moisture, nutrient_state):
    risk_score = 0
    if disease_name != "Healthy":
        risk_score += 50
    if ndvi < 0.5:
        risk_score += 30
    if soil_moisture < 30:
        risk_score += 10
    if nutrient_state == "Weak":
        risk_score += 10
    if risk_score > 60:
        risk = "HIGH"
    elif risk_score >= 30:
        risk = "MEDIUM"
    else:
        risk = "LOW"
    health_score = max(10, 100 - risk_score)
    return risk_score, risk, health_score


def combined_decision_score(crop_confidence, yield_confidence, disease_confidence, health_score, risk_score):
    score = (crop_confidence * 0.25) + (yield_confidence * 0.20) + (disease_confidence * 0.15) + (health_score * 0.25) + ((100 - risk_score) * 0.15)
    return round(score, 2)


def final_priority(risk_level, disease_name, soil_moisture, nutrient_state):
    if risk_level == "HIGH":
        return "URGENT"
    if disease_name != "Healthy" or soil_moisture < 30 or nutrient_state == "Weak":
        return "ATTENTION NEEDED"
    return "STABLE"


def build_smart_farm_report(location, season, crop, crop_conf, yield_total, yield_conf, weather_name, disease_name, disease_conf, risk, health_score, final_score, priority, intelligence):
    actions = []
    if intelligence["irrigation_level"] == "Low":
        actions.append("Reduce irrigation")
    elif intelligence["irrigation_level"] == "High":
        actions.append("Increase irrigation immediately")
    else:
        actions.append("Monitor irrigation in 1-2 days")
    if disease_name != "Healthy":
        actions.append("Apply fungicide or crop-protection spray after field inspection")
    if intelligence["fertilizer_level"] == "Reduced usage":
        actions.append("Avoid heavy nitrogen fertilizer now")
    elif intelligence["fertilizer_level"] == "Targeted usage":
        actions.append("Add balanced nitrogen-phosphorus-potassium correction")
    else:
        actions.append("Continue split fertilizer schedule")
    return {
        "location": location,
        "season": season,
        "crop": crop,
        "crop_conf": crop_conf,
        "yield_total": yield_total,
        "yield_conf": yield_conf,
        "weather": weather_name,
        "disease": disease_name,
        "disease_conf": disease_conf,
        "risk": risk,
        "health_score": health_score,
        "final_score": final_score,
        "priority": priority,
        "actions": actions[:3],
    }


def predict_all_modules(location, season, soil_type, area, temperature, humidity, rainfall, soil_moisture, ph, nitrogen, phosphorus, potassium, ndvi):
    top_recommendations = recommend_crop(season, soil_type, temperature, humidity, rainfall, ph, nitrogen, phosphorus, potassium)
    selected_crop = top_recommendations[0]["crop"]
    crop_confidence = top_recommendations[0]["confidence"]
    yield_per_acre, yield_total, yield_confidence = predict_yield(selected_crop, area, rainfall, temperature, soil_moisture, nitrogen, phosphorus, potassium)
    disease_name, disease_confidence = predict_disease(selected_crop, ndvi, humidity, soil_moisture)
    nutrient_state = nutrient_health(nitrogen, phosphorus, potassium)
    weather_name = weather_label(rainfall, humidity)
    intelligence = build_intelligence_layer(selected_crop, disease_name, soil_moisture, rainfall, temperature, nitrogen, phosphorus, potassium)
    risk_score, risk_level, health_score = calculate_risk_and_health(disease_name, ndvi, soil_moisture, nutrient_state)
    final_score = combined_decision_score(crop_confidence, yield_confidence, disease_confidence, health_score, risk_score)
    priority = final_priority(risk_level, disease_name, soil_moisture, nutrient_state)
    report = build_smart_farm_report(location, season, selected_crop, crop_confidence, yield_total, yield_confidence, weather_name, disease_name, disease_confidence, risk_level, health_score, final_score, priority, intelligence)
    return {
        "recommendations": top_recommendations,
        "selected_crop": selected_crop,
        "crop_confidence": crop_confidence,
        "yield_per_acre": yield_per_acre,
        "yield_total": yield_total,
        "yield_confidence": yield_confidence,
        "disease_name": disease_name,
        "disease_confidence": disease_confidence,
        "nutrient_state": nutrient_state,
        "weather_name": weather_name,
        "intelligence": intelligence,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "health_score": health_score,
        "final_score": final_score,
        "priority": priority,
        "report": report,
    }


def build_openai_prompt(user_input, farm_data, report, intelligence):
    return f"""
You are an expert AI agriculture assistant helping Indian farmers.

Farmer Details:
Location: {farm_data['location']}
Crop: {farm_data['crop']}
Soil: {farm_data['soil']}
Season: {farm_data['season']}
Temperature: {farm_data['temperature']}
Humidity: {farm_data['humidity']}
Rainfall: {farm_data['rainfall']}
Soil Moisture: {farm_data['soil_moisture']}
NDVI: {farm_data['ndvi']}
Disease Status: {report['disease']}
Risk Level: {report['risk']}
Health Score: {report['health_score']}
Final Farm Score: {report['final_score']}
Priority: {report['priority']}
Suggested Irrigation: {intelligence['irrigation_level']}
Suggested Fertilizer: {intelligence['fertilizer_level']}
Priority Actions: {', '.join(report['actions'])}

User Question:
{user_input}

Instructions:
- Give friendly, simple answers.
- Provide step-by-step guidance.
- Use easy farmer-friendly language.
- If priority is URGENT, say that clearly.
- Keep it short, practical, and actionable.
"""


def farmer_chatbot(user_input, farm_data, report, intelligence):
    if OpenAI is None:
        return "OpenAI package is not installed. Run: pip install openai"
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return "OpenAI API key not found. Set OPENAI_API_KEY in your environment before using AI chat."
    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a smart farming assistant."},
                {"role": "user", "content": build_openai_prompt(user_input, farm_data, report, intelligence)}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI assistant error: {e}"


def fallback_chatbot(user_input, report, intelligence, nutrient_state, ndvi_value):
    q = user_input.lower().strip()
    intro = f"🤖 Agri Vision AI Assistant: Crop is {report['crop']}, risk is {report['risk']}, farm priority is {report['priority']}, and health score is {report['health_score']}/100."
    if "what should i do" in q or "what to do" in q:
        return f"{intro}\n\nDo these steps: {', '.join(report['actions'])}. Main reason is disease status {report['disease']} and NDVI {ndvi_value:.2f}."
    if "irrigation" in q or "water" in q:
        return f"{intro}\n\nIrrigation advice: {intelligence['irrigation_text']}"
    if "fertilizer" in q or "npk" in q:
        return f"{intro}\n\nFertilizer advice: {intelligence['fertilizer_text']} Nutrient condition is {nutrient_state}."
    return f"{intro}\n\nPriority actions: {', '.join(report['actions'])}."


def ensure_chat_history():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if not isinstance(st.session_state.chat_history, list):
        st.session_state.chat_history = []
    cleaned = []
    for item in st.session_state.chat_history:
        if isinstance(item, dict) and "question" in item and "answer" in item:
            cleaned.append(item)
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            cleaned.append({"question": str(item[0]), "answer": str(item[1])})
    st.session_state.chat_history = cleaned


def gauge_chart(value, title, color, max_value=100):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title},
        gauge={"axis": {"range": [0, max_value]}, "bar": {"color": color}}
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def trend_chart(ndvi, soil_moisture):
    weeks = ["Week 1", "Week 2", "Week 3", "Week 4"]
    ndvi_series = [max(0.15, ndvi - 0.12), max(0.18, ndvi - 0.06), ndvi, min(1.0, ndvi + 0.03)]
    moisture_series = [max(5, soil_moisture - 10), max(10, soil_moisture - 5), soil_moisture, min(100, soil_moisture + 4)]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=weeks, y=ndvi_series, name="NDVI", mode="lines+markers", line=dict(color="#16a34a", width=3)),
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=weeks, y=moisture_series, name="Soil Moisture %", mode="lines+markers", line=dict(color="#2563eb", width=3)),
        secondary_y=True
    )
    fig.update_layout(title="📊 Field Health Trend", height=340, legend=dict(orientation="h"))
    fig.update_yaxes(title_text="NDVI", secondary_y=False)
    fig.update_yaxes(title_text="Soil Moisture %", secondary_y=True)
    return fig


def yield_forecast_chart(yield_total):
    stages = ["Current", "Next 10 Days", "Flowering", "Harvest"]
    values = [round(yield_total * 0.72, 2), round(yield_total * 0.84, 2), round(yield_total * 0.92, 2), yield_total]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=stages,
        y=values,
        marker_color=["#93c5fd", "#60a5fa", "#3b82f6", "#1d4ed8"],
        text=values,
        textposition="outside"
    ))
    fig.update_layout(title="📈 Yield Forecast", height=340, yaxis_title="Tons")
    return fig


def nutrient_chart(nitrogen, phosphorus, potassium):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Nitrogen", "Phosphorus", "Potassium"],
        y=[nitrogen, phosphorus, potassium],
        marker_color=["#22c55e", "#f59e0b", "#3b82f6"],
        text=[nitrogen, phosphorus, potassium],
        textposition="outside"
    ))
    fig.add_hline(y=50, line_dash="dash", line_color="red", annotation_text="Ideal safety line")
    fig.update_layout(title="📉 Soil Nutrient Status", height=340, yaxis_title="Value")
    return fig


def confidence_chart(crop_conf, yield_conf, disease_conf):
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[crop_conf, yield_conf, disease_conf, crop_conf],
        theta=["Crop", "Yield", "Disease", "Crop"],
        fill="toself",
        line=dict(color="#7c3aed", width=3)
    ))
    fig.update_layout(
        title="🎯 Prediction Confidence",
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        height=340
    )
    return fig


def main():
    st.title(APP_NAME)
    st.caption("One-click integrated Telangana smart farming app with crop, yield, disease, irrigation, fertilizer, risk, health, and AI assistant in one workflow.")

    with st.sidebar:
        st.header("Farm Setup")
        location = st.selectbox("Location", TELANGANA_DISTRICTS)
        season = st.selectbox("Season", ["Kharif", "Rabi", "Summer"])
        farm_type = st.selectbox("Select Farm Type", FARM_TYPES)
        auto_fill = st.checkbox("Use real farm simulation mode", value=True)
        defaults = SIMULATION_DEFAULTS[farm_type]

        if auto_fill:
            temperature_default = float(defaults["temperature"])
            humidity_default = float(defaults["humidity"])
            rainfall_default = float(defaults["rainfall"])
            moisture_default = float(defaults["soil_moisture"])
            ph_default = float(defaults["ph"])
            n_default = int(defaults["nitrogen"])
            p_default = int(defaults["phosphorus"])
            k_default = int(defaults["potassium"])
            ndvi_default = float(defaults["ndvi"])
        else:
            temperature_default = 29.0
            humidity_default = 65.0
            rainfall_default = 350.0
            moisture_default = 50.0
            ph_default = 6.8
            n_default = 55
            p_default = 28
            k_default = 30
            ndvi_default = 0.60

        soil_type = st.selectbox("Soil Type", SOIL_TYPES)
        area = st.number_input("Area (acres)", 0.5, 100.0, 2.0, 0.5)
        temperature = st.slider("Temperature (°C)", 10.0, 45.0, temperature_default, 0.5)
        humidity = st.slider("Humidity (%)", 10.0, 100.0, humidity_default, 1.0)
        rainfall = st.slider("Rainfall (mm)", 0.0, 1200.0, rainfall_default, 5.0)
        soil_moisture = st.slider("Soil Moisture (%)", 5.0, 100.0, moisture_default, 1.0)
        ph = st.slider("Soil pH", 4.0, 9.0, ph_default, 0.1)
        nitrogen = st.slider("Nitrogen (N)", 0, 140, n_default, 1)
        phosphorus = st.slider("Phosphorus (P)", 0, 120, p_default, 1)
        potassium = st.slider("Potassium (K)", 0, 140, k_default, 1)
        ndvi = st.slider("NDVI", 0.0, 1.0, ndvi_default, 0.01)
        run_all = st.button("🚀 Predict All Modules - One Click", use_container_width=True)

    if "results" not in st.session_state:
        st.session_state.results = None

    if run_all:
        st.session_state.results = predict_all_modules(
            location, season, soil_type, area, temperature, humidity, rainfall,
            soil_moisture, ph, nitrogen, phosphorus, potassium, ndvi
        )

    if st.session_state.results is None:
        st.info("Enter farm details in the sidebar and click 'Predict All Modules - One Click' to generate the integrated farm report.")
        return

    results = st.session_state.results
    report = results["report"]
    intelligence = results["intelligence"]

    tabs = st.tabs(["🌾 Smart Farming", "👁 Crop Monitoring", "📊 Dashboard", "🤖 AI Assistant"])

    with tabs[0]:
        st.subheader("🌾 ONE-CLICK SMART FARM REPORT")
        st.markdown(f"""
### 🌾 SMART FARM REPORT
**Location:** {report['location']}  
**Season:** {report['season']}  
**Crop:** {report['crop']} ({report['crop_conf']}%)  
**Yield:** {report['yield_total']} tons ({report['yield_conf']}%)  
**Weather:** {report['weather']}  
**Disease:** {report['disease']} ({report['disease_conf']}%)  
**Risk Level:** {report['risk']}  
**Health Score:** {report['health_score']}/100  
**Final Farm Score:** {report['final_score']}/100  
**Priority:** {report['priority']}
""")
        st.markdown("**✅ Recommended Actions:**")
        for action in report["actions"]:
            st.write(f"- {action}")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Crop", report["crop"], f"{report['crop_conf']}%")
        m2.metric("Yield", f"{report['yield_total']} tons", f"{report['yield_conf']}%")
        m3.metric("Disease", report["disease"], f"{report['disease_conf']}%")
        m4.metric("Risk Score", results["risk_score"], report["risk"])
        m5.metric("Farm Priority", report["priority"], f"Score {report['final_score']}")

        st.success(f"Integrated decision complete. Irrigation: {intelligence['irrigation_level']} | Fertilizer: {intelligence['fertilizer_level']}")

    with tabs[1]:
        st.subheader("👁 Combined Monitoring")
        left, right = st.columns(2)
        left.write(f"Top crop recommendations: {', '.join([item['crop'] for item in results['recommendations']])}")
        left.write(f"NDVI status: {ndvi_status(ndvi)}")
        left.write(f"Nutrient health: {results['nutrient_state']}")
        left.write(f"Weather condition: {results['weather_name']}")
        right.write(f"Disease status: {results['disease_name']}")
        right.write(f"Irrigation plan: {intelligence['irrigation_text']}")
        right.write(f"Fertilizer plan: {intelligence['fertilizer_text']}")
        right.write(f"Yield per acre: {results['yield_per_acre']} tons")
        st.dataframe(results["recommendations"], use_container_width=True)

    with tabs[2]:
        st.subheader("📊 Advanced Combined Dashboard")
        r1, r2 = st.columns(2)
        r1.plotly_chart(trend_chart(ndvi, soil_moisture), use_container_width=True)
        r2.plotly_chart(yield_forecast_chart(results["yield_total"]), use_container_width=True)

        r3, r4 = st.columns(2)
        r3.plotly_chart(nutrient_chart(nitrogen, phosphorus, potassium), use_container_width=True)
        r4.plotly_chart(confidence_chart(results["crop_confidence"], results["yield_confidence"], results["disease_confidence"]), use_container_width=True)

        r5, r6 = st.columns(2)
        r5.plotly_chart(
            gauge_chart(
                results["health_score"],
                "Farm Health Score",
                "#dc2626" if results["risk_level"] == "HIGH" else "#f59e0b" if results["risk_level"] == "MEDIUM" else "#16a34a"
            ),
            use_container_width=True
        )
        r6.plotly_chart(
            gauge_chart(results["final_score"], "Final Combined Farm Score", "#2563eb", 100),
            use_container_width=True
        )

    with tabs[3]:
        st.subheader("🤖 AI Assistant")
        ensure_chat_history()
        st.caption("For real AI answers, set OPENAI_API_KEY in your environment.")
        user_input = st.text_input("Ask your farming question")
        col1, col2 = st.columns(2)
        ask_ai = col1.button("Ask AI")
        clear = col2.button("Clear Chat")

        if clear:
            st.session_state.chat_history = []

        if ask_ai and user_input.strip():
            farm_data = {
                "location": report["location"],
                "crop": report["crop"],
                "soil": soil_type,
                "season": report["season"],
                "temperature": temperature,
                "humidity": humidity,
                "rainfall": rainfall,
                "soil_moisture": soil_moisture,
                "ndvi": ndvi,
            }
            if os.getenv("OPENAI_API_KEY") and OpenAI is not None:
                answer = farmer_chatbot(user_input, farm_data, report, intelligence)
            else:
                answer = fallback_chatbot(user_input, report, intelligence, results["nutrient_state"], ndvi)

            st.session_state.chat_history.append({
                "question": user_input,
                "answer": answer,
                "time": datetime.now().strftime("%H:%M")
            })

        for item in reversed(st.session_state.chat_history):
            st.markdown(f"**Farmer ({item.get('time', '')}):** {item['question']}")
            st.markdown(item["answer"])
            st.markdown("---")


if __name__ == "__main__":
    main()
