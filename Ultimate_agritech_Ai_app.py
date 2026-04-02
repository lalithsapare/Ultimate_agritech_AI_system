import os
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Agri Vision AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main {background: linear-gradient(135deg, #08111f 0%, #020817 100%);}
[data-testid="stSidebar"] {background: linear-gradient(180deg, #166534 0%, #14532d 100%);}
[data-testid="stSidebar"] * {color: white !important;}
.hero {
    background: linear-gradient(90deg, #166534, #22c55e);
    color: white;
    padding: 22px;
    border-radius: 20px;
    margin-bottom: 16px;
}
.card {
    background: rgba(255,255,255,0.97);
    padding: 16px;
    border-radius: 18px;
    margin-bottom: 12px;
}
.alert-high {
    background:#fee2e2;
    color:#991b1b;
    padding:12px 16px;
    border-radius:14px;
    margin-bottom:10px;
    font-weight:600;
}
.alert-med {
    background:#fef3c7;
    color:#92400e;
    padding:12px 16px;
    border-radius:14px;
    margin-bottom:10px;
    font-weight:600;
}
.alert-good {
    background:#dcfce7;
    color:#166534;
    padding:12px 16px;
    border-radius:14px;
    margin-bottom:10px;
    font-weight:600;
}
.small-muted {
    color:#64748b;
    font-size:13px;
}
.crop-pill {
    display:inline-block;
    padding:10px 14px;
    border-radius:999px;
    background:#dcfce7;
    color:#166534;
    font-weight:700;
}
</style>
""", unsafe_allow_html=True)

ALL_CROPS = [
    "Cotton",
    "Maize",
    "Red Gram",
    "Soybean",
    "Turmeric",
    "Jowar",
    "Paddy",
    "Groundnut",
    "Black Gram",
    "Bengal Gram",
    "Sesame",
    "Sunflower"
]

if "latest_result" not in st.session_state:
    st.session_state.latest_result = None

def get_secret(name, default=""):
    try:
        if name in st.secrets:
            value = str(st.secrets[name]).strip()
            if value:
                return value
    except Exception:
        pass

    env_value = os.getenv(name, "").strip()
    if env_value:
        return env_value

    return default

def fetch_weather(city):
    api_key = get_secret("OPENWEATHER_API_KEY")
    debug = {
        "city": city,
        "api_key_present": bool(api_key),
        "status": "unknown",
        "reason": "",
        "fallback_used": False
    }

    if not api_key:
        debug["status"] = "missing_api_key"
        debug["reason"] = "OPENWEATHER_API_KEY not found"
        debug["fallback_used"] = True
        return None, debug

    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": city,
            "appid": api_key,
            "units": "metric"
        }
        response = requests.get(url, params=params, timeout=20)
        debug["http_status"] = response.status_code

        if response.status_code != 200:
            debug["status"] = "api_error"
            try:
                debug["reason"] = response.json()
            except Exception:
                debug["reason"] = response.text
            debug["fallback_used"] = True
            return None, debug

        data = response.json()

        if "main" not in data or "weather" not in data:
            debug["status"] = "invalid_response"
            debug["reason"] = data
            debug["fallback_used"] = True
            return None, debug

        result = {
            "temperature": float(data["main"].get("temp", 28.0)),
            "humidity": float(data["main"].get("humidity", 70.0)),
            "rainfall": float(data.get("rain", {}).get("1h", 0.0)),
            "weather": data["weather"][0].get("main", "Unknown"),
            "city": data.get("name", city)
        }

        debug["status"] = "success"
        return result, debug

    except requests.exceptions.Timeout:
        debug["status"] = "timeout"
        debug["reason"] = "Request timed out"
        debug["fallback_used"] = True
        return None, debug

    except requests.exceptions.ConnectionError:
        debug["status"] = "connection_error"
        debug["reason"] = "Network connection failed"
        debug["fallback_used"] = True
        return None, debug

    except Exception as e:
        debug["status"] = "exception"
        debug["reason"] = str(e)
        debug["fallback_used"] = True
        return None, debug

def crop_score(crop, temp, humidity, rainfall, ph, moisture, n, p, k):
    score = 50.0

    if crop == "Cotton":
        score += 15 if 25 <= temp <= 35 else 0
        score += 10 if 40 <= humidity <= 80 else 0
        score += 10 if rainfall >= 40 else 0
        score += 8 if 6.0 <= ph <= 8.0 else 0
        score += 5 if k >= 40 else 0

    elif crop == "Maize":
        score += 15 if 20 <= temp <= 32 else 0
        score += 8 if humidity >= 50 else 0
        score += 10 if rainfall >= 30 else 0
        score += 8 if 5.5 <= ph <= 7.5 else 0
        score += 7 if n >= 60 else 0

    elif crop == "Red Gram":
        score += 14 if 22 <= temp <= 34 else 0
        score += 10 if rainfall >= 20 else 0
        score += 8 if 6.0 <= ph <= 7.5 else 0
        score += 6 if moisture >= 25 else 0

    elif crop == "Soybean":
        score += 14 if 20 <= temp <= 30 else 0
        score += 8 if humidity >= 55 else 0
        score += 10 if rainfall >= 50 else 0
        score += 8 if 6.0 <= ph <= 7.5 else 0

    elif crop == "Turmeric":
        score += 14 if 20 <= temp <= 32 else 0
        score += 10 if humidity >= 60 else 0
        score += 10 if rainfall >= 60 else 0
        score += 8 if moisture >= 35 else 0

    elif crop == "Jowar":
        score += 14 if 24 <= temp <= 36 else 0
        score += 8 if rainfall >= 20 else 0
        score += 8 if 5.5 <= ph <= 8.0 else 0
        score += 5 if moisture >= 20 else 0

    elif crop == "Paddy":
        score += 14 if 20 <= temp <= 35 else 0
        score += 10 if humidity >= 65 else 0
        score += 12 if rainfall >= 80 else 0
        score += 10 if moisture >= 40 else 0
        score += 7 if 5.5 <= ph <= 7.5 else 0

    elif crop == "Groundnut":
        score += 14 if 22 <= temp <= 32 else 0
        score += 8 if rainfall >= 25 else 0
        score += 8 if 6.0 <= ph <= 7.5 else 0
        score += 5 if k >= 35 else 0

    elif crop == "Black Gram":
        score += 12 if 24 <= temp <= 34 else 0
        score += 8 if rainfall >= 20 else 0
        score += 7 if humidity >= 45 else 0
        score += 8 if 6.0 <= ph <= 7.5 else 0

    elif crop == "Bengal Gram":
        score += 14 if 18 <= temp <= 30 else 0
        score += 10 if rainfall <= 60 else 0
        score += 8 if 6.0 <= ph <= 8.0 else 0
        score += 5 if moisture >= 20 else 0

    elif crop == "Sesame":
        score += 13 if 24 <= temp <= 34 else 0
        score += 8 if rainfall >= 20 else 0
        score += 8 if 5.5 <= ph <= 7.5 else 0

    elif crop == "Sunflower":
        score += 13 if 20 <= temp <= 32 else 0
        score += 8 if rainfall >= 20 else 0
        score += 7 if humidity <= 75 else 0
        score += 8 if 6.0 <= ph <= 7.8 else 0

    score += min(n / 20, 5)
    score += min(p / 20, 5)
    score += min(k / 20, 5)

    return round(min(score, 98), 1)

def predict_yield(crop, temp, humidity, rainfall, ph, moisture, n, p, k):
    base = {
        "Cotton": 22,
        "Maize": 45,
        "Red Gram": 14,
        "Soybean": 20,
        "Turmeric": 65,
        "Jowar": 18,
        "Paddy": 55,
        "Groundnut": 24,
        "Black Gram": 13,
        "Bengal Gram": 16,
        "Sesame": 10,
        "Sunflower": 17
    }.get(crop, 20)

    factor = (
        (temp * 0.18) +
        (humidity * 0.07) +
        (rainfall * 0.03) +
        (moisture * 0.22) +
        ((n + p + k) * 0.04) -
        (abs(ph - 6.8) * 2.0)
    )

    return max(1.0, round((base + factor) / 10, 2))

def irrigation_advice(moisture, temp, rainfall):
    if rainfall > 20 or moisture > 60:
        return "Low", "Skip irrigation today"
    if moisture < 30 and temp > 32:
        return "High", "Provide deep irrigation immediately"
    return "Moderate", "Light irrigation recommended"

def rain_prediction(rainfall, humidity, temp):
    chance = (humidity * 0.7) + (rainfall * 8) - (temp * 0.4)
    return min(100, max(0, round(chance, 1)))

def smart_alerts(crop, moisture, rainfall, temp, suitability):
    alerts = []

    if suitability < 65:
        alerts.append(("high", f"🚨 {crop} suitability is low under current conditions."))
    if moisture < 25:
        alerts.append(("med", "⚠ Low soil moisture detected."))
    if rainfall < 5 and temp > 35:
        alerts.append(("high", "🚨 High heat and low rain risk detected."))
    if suitability >= 75 and 25 <= moisture <= 60:
        alerts.append(("good", f"✅ {crop} is in a favorable growing range."))

    return alerts

st.sidebar.title("🌾 Agri Vision AI")

district = st.sidebar.selectbox(
    "Select District / City",
    ["Hyderabad", "Warangal", "Karimnagar", "Nizamabad", "Khammam", "Mahabubnagar"]
)

selected_crop = st.sidebar.selectbox("Select Crop", ALL_CROPS)

st.sidebar.markdown("### Manual Backup Inputs")
manual_temp = st.sidebar.number_input("Temperature (°C)", value=29.0)
manual_humidity = st.sidebar.number_input("Humidity (%)", value=70.0)
manual_rainfall = st.sidebar.number_input("Rainfall (mm)", value=8.0)
manual_ph = st.sidebar.number_input("Soil pH", value=6.8)
manual_moisture = st.sidebar.number_input("Soil Moisture (%)", value=38.0)
manual_n = st.sidebar.number_input("Nitrogen (N)", value=80.0)
manual_p = st.sidebar.number_input("Phosphorus (P)", value=42.0)
manual_k = st.sidebar.number_input("Potassium (K)", value=40.0)

st.markdown(f"""
<div class="hero">
    <h1>Agri Vision AI</h1>
    <p>Selected crop: <b>{selected_crop}</b> | City: <b>{district}</b> | One-click farm prediction</p>
</div>
""", unsafe_allow_html=True)

weather_data, weather_debug = fetch_weather(district)

if weather_data is not None:
    temp = weather_data["temperature"]
    humidity = weather_data["humidity"]
    rainfall = weather_data["rainfall"]
    weather_label = weather_data["weather"]
    weather_mode = "Live Weather API"
else:
    temp = manual_temp
    humidity = manual_humidity
    rainfall = manual_rainfall
    weather_label = "Manual input active"
    weather_mode = "Manual backup"

ph = manual_ph
moisture = manual_moisture
n = manual_n
p = manual_p
k = manual_k

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Weather Source", weather_mode)
with m2:
    st.metric("Temperature", f"{temp} °C")
with m3:
    st.metric("Humidity", f"{humidity} %")
with m4:
    st.metric("Rainfall", f"{rainfall} mm")

if weather_data is None:
    st.warning("Weather service is currently unavailable. Manual values are being used.")

with st.expander("Weather debug details", expanded=False):
    st.json(weather_debug)

if st.button("🚀 Predict All", type="primary", use_container_width=True):
    suitability = crop_score(selected_crop, temp, humidity, rainfall, ph, moisture, n, p, k)
    yield_pred = predict_yield(selected_crop, temp, humidity, rainfall, ph, moisture, n, p, k)
    irrigation_level, irrigation_action = irrigation_advice(moisture, temp, rainfall)
    rain_chance = rain_prediction(rainfall, humidity, temp)

    st.session_state.latest_result = {
        "crop": str(selected_crop),
        "temperature": float(temp),
        "humidity": float(humidity),
        "rainfall": float(rainfall),
        "ph": float(ph),
        "moisture": float(moisture),
        "n": float(n),
        "p": float(p),
        "k": float(k),
        "weather_label": str(weather_label),
        "weather_mode": str(weather_mode),
        "suitability": float(suitability),
        "yield_pred": float(yield_pred),
        "irrigation_level": str(irrigation_level),
        "irrigation_action": str(irrigation_action),
        "rain_chance": float(rain_chance)
    }

if st.session_state.latest_result:
    result = st.session_state.latest_result

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Selected Crop")
    st.markdown(f'<span class="crop-pill">{result["crop"]}</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    a, b, c = st.columns(3)
    with a:
        st.metric("Suitability", f'{result["suitability"]}%')
    with b:
        st.metric("Yield Prediction", f'{result["yield_pred"]} t/ha')
    with c:
        st.metric("Rain Chance", f'{result["rain_chance"]}%')

    x1, x2 = st.columns(2)

    with x1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Prediction Summary")
        st.write(f"Weather source: {result['weather_mode']}")
        st.write(f"Weather condition: {result['weather_label']}")
        st.write(f"Temperature: {result['temperature']} °C")
        st.write(f"Humidity: {result['humidity']} %")
        st.write(f"Rainfall: {result['rainfall']} mm")
        st.write(f"Soil pH: {result['ph']}")
        st.write(f"Soil moisture: {result['moisture']} %")
        st.write(f"Irrigation need: {result['irrigation_level']}")
        st.write(f"Action: {result['irrigation_action']}")
        st.markdown('</div>', unsafe_allow_html=True)

    with x2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Why this result")
        st.write(f"{result['crop']} was scored using weather, humidity, rainfall, soil moisture, pH, and NPK values.")
        st.write("Suitability shows how well the selected crop matches current farm conditions.")
        st.write("Yield is an estimated value based on crop baseline plus environmental adjustments.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("Smart Alerts")
    alerts = smart_alerts(
        result["crop"],
        result["moisture"],
        result["rainfall"],
        result["temperature"],
        result["suitability"]
    )

    if alerts:
        for level, message in alerts:
            if level == "high":
                st.markdown(f'<div class="alert-high">{message}</div>', unsafe_allow_html=True)
            elif level == "med":
                st.markdown(f'<div class="alert-med">{message}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="alert-good">{message}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert-good">✅ No major alerts detected.</div>', unsafe_allow_html=True)

    st.subheader("Analytics")

    col1, col2 = st.columns(2)

    with col1:
        metric_df = pd.DataFrame({
            "Metric": ["Temperature", "Humidity", "Rainfall", "Moisture", "Suitability", "Rain Chance"],
            "Value": [
                result["temperature"],
                result["humidity"],
                result["rainfall"],
                result["moisture"],
                result["suitability"],
                result["rain_chance"]
            ]
        })
        fig_bar = px.bar(metric_df, x="Metric", y="Value", color="Metric", title="All-in-One Prediction Metrics")
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=result["suitability"],
            title={'text': f'{result["crop"]} Suitability'},
            gauge={'axis': {'range': [0, 100]}}
        ))
        fig_gauge.update_layout(height=400)
        st.plotly_chart(fig_gauge, use_container_width=True)

st.markdown("<div class='small-muted'>Agri Vision AI | Crop selected from sidebar | Weather fallback handled safely</div>", unsafe_allow_html=True)
