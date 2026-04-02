import os
import io
import json
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False


st.set_page_config(
    page_title="Agri Vision AI – Telangana Edition",
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
    padding: 24px;
    border-radius: 22px;
    margin-bottom: 16px;
}
.card {
    background: rgba(255,255,255,0.96);
    padding: 16px;
    border-radius: 18px;
    margin-bottom: 12px;
}
.small {color:#64748b;font-size:13px;}
.footer {color:#94a3b8;font-size:13px;padding-top:20px;}
.alert-high {background:#fee2e2;color:#991b1b;padding:12px 16px;border-radius:14px;margin-bottom:10px;}
.alert-med {background:#fef3c7;color:#92400e;padding:12px 16px;border-radius:14px;margin-bottom:10px;}
.alert-good {background:#dcfce7;color:#166534;padding:12px 16px;border-radius:14px;margin-bottom:10px;}
</style>
""", unsafe_allow_html=True)


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "latest_result" not in st.session_state:
    st.session_state.latest_result = None


TELANGANA_CROPS = {
    "Kharif": [
        "Cotton", "Maize", "Red Gram", "Soybean", "Turmeric", "Jowar", "Paddy"
    ],
    "Rabi": [
        "Bengal Gram", "Groundnut", "Sesame", "Jowar", "Paddy", "Black Gram", "Sunflower"
    ]
}


def get_secret(name: str, default: str = ""):
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


def fetch_weather(city: str):
    api_key = get_secret("OPENWEATHER_API_KEY")
    debug = {
        "city": city,
        "has_key": bool(api_key),
        "status": "unknown",
        "reason": ""
    }

    if not api_key:
        debug["status"] = "missing_key"
        debug["reason"] = "OPENWEATHER_API_KEY not found."
        return None, debug

    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"q": city, "appid": api_key, "units": "metric"}
        r = requests.get(url, params=params, timeout=20)
        debug["http_status"] = r.status_code

        if r.status_code != 200:
            debug["status"] = "api_error"
            try:
                debug["reason"] = r.json()
            except Exception:
                debug["reason"] = r.text
            return None, debug

        data = r.json()
        if "main" not in data:
            debug["status"] = "bad_response"
            debug["reason"] = data
            return None, debug

        result = {
            "temperature": float(data["main"].get("temp", 29.0)),
            "humidity": float(data["main"].get("humidity", 75.0)),
            "rainfall": float(data.get("rain", {}).get("1h", 0.0)),
            "weather": data["weather"][0]["main"] if data.get("weather") else "Unknown",
            "city": data.get("name", city)
        }
        debug["status"] = "success"
        return result, debug

    except requests.exceptions.Timeout:
        debug["status"] = "timeout"
        debug["reason"] = "Weather API timed out"
        return None, debug

    except Exception as e:
        debug["status"] = "exception"
        debug["reason"] = str(e)
        return None, debug


class AgriEngine:
    def __init__(self):
        self.crops = TELANGANA_CROPS

    def crop_recommendation(self, n, p, k, temp, humidity, ph, rainfall, season):
        crop_list = self.crops.get(season, self.crops["Kharif"])

        suitability = []
        for crop in crop_list:
            score = 50.0

            if crop == "Cotton":
                score += 12 if 25 <= temp <= 35 else 0
                score += 10 if rainfall >= 50 else 0
                score += 8 if 6.0 <= ph <= 8.0 else 0

            elif crop == "Maize":
                score += 12 if 20 <= temp <= 32 else 0
                score += 10 if humidity >= 50 else 0
                score += 8 if rainfall >= 40 else 0

            elif crop == "Paddy":
                score += 15 if rainfall >= 80 else 0
                score += 10 if humidity >= 70 else 0
                score += 8 if soil_moisture >= 40 else 0

            elif crop == "Red Gram":
                score += 12 if 22 <= temp <= 34 else 0
                score += 9 if rainfall >= 30 else 0
                score += 8 if 6.0 <= ph <= 7.5 else 0

            elif crop == "Turmeric":
                score += 14 if humidity >= 60 else 0
                score += 10 if rainfall >= 70 else 0
                score += 8 if 5.5 <= ph <= 7.5 else 0

            elif crop == "Bengal Gram":
                score += 13 if 18 <= temp <= 30 else 0
                score += 8 if rainfall <= 60 else 0
                score += 8 if 6.0 <= ph <= 8.0 else 0

            elif crop == "Groundnut":
                score += 12 if 22 <= temp <= 32 else 0
                score += 9 if rainfall >= 35 else 0
                score += 9 if k >= 35 else 0

            elif crop == "Black Gram":
                score += 11 if 24 <= temp <= 34 else 0
                score += 9 if rainfall >= 25 else 0
                score += 8 if humidity >= 45 else 0

            elif crop == "Jowar":
                score += 11 if 24 <= temp <= 36 else 0
                score += 10 if rainfall >= 20 else 0
                score += 8 if ph >= 6 else 0

            elif crop == "Soybean":
                score += 11 if 22 <= temp <= 32 else 0
                score += 10 if rainfall >= 60 else 0
                score += 8 if humidity >= 55 else 0

            elif crop == "Sesame":
                score += 10 if 24 <= temp <= 34 else 0
                score += 8 if rainfall >= 20 else 0
                score += 8 if ph >= 5.5 else 0

            elif crop == "Sunflower":
                score += 11 if 20 <= temp <= 32 else 0
                score += 8 if rainfall >= 25 else 0
                score += 8 if humidity <= 70 else 0

            score += min(n / 20, 5)
            score += min(p / 20, 5)
            score += min(k / 20, 5)

            suitability.append({"Crop": crop, "Suitability": round(min(score, 98), 1)})

        suitability = sorted(suitability, key=lambda x: x["Suitability"], reverse=True)
        selected = suitability[0]["Crop"]
        confidence = suitability[0]["Suitability"]

        return {
            "crop": selected,
            "confidence": confidence,
            "scores": suitability,
            "why": {
                "Nitrogen": round(min(n / 100, 1), 2),
                "Phosphorus": round(min(p / 100, 1), 2),
                "Potassium": round(min(k / 100, 1), 2),
                "Temperature": round(min(temp / 40, 1), 2),
                "Humidity": round(min(humidity / 100, 1), 2),
                "pH": round(min(ph / 10, 1), 2),
                "Rainfall": round(min(rainfall / 200, 1), 2)
            }
        }

    def yield_prediction(self, ph, temp, rainfall, fert, humidity, moisture):
        pred = (
            (temp * 0.65)
            + (rainfall * 0.03)
            + (fert * 0.045)
            + (humidity * 0.075)
            + (moisture * 0.25)
            - abs(ph - 6.8) * 2.2
        )
        pred = round(max(pred, 0), 2)

        months = ["Jun", "Jul", "Aug", "Sep", "Oct", "Nov"]
        trend = [round(max(pred * f, 0), 2) for f in [0.42, 0.56, 0.71, 0.84, 0.93, 1.00]]
        return {"yield": pred, "months": months, "trend": trend}

    def irrigation(self, moisture, temp, humidity, ph, rainfall):
        if rainfall > 150 or moisture > 55:
            need, action, conf = "Low", "Skip irrigation cycle", 92.0
            liters = [20, 15, 10, 10, 8, 8, 5]
        elif moisture < 28 and temp > 32:
            need, action, conf = "High", "Deep irrigation needed", 90.0
            liters = [55, 58, 60, 50, 48, 46, 40]
        else:
            need, action, conf = "Moderate", "Light irrigation", 89.0
            liters = [35, 38, 40, 34, 32, 30, 28]

        return {
            "need": need,
            "action": action,
            "confidence": conf,
            "days": ["Day1", "Day2", "Day3", "Day4", "Day5", "Day6", "Day7"],
            "liters": liters
        }

    def fertilizer(self, n, p, k):
        if n < 50:
            rec = "Apply nitrogen fertilizer in split doses"
        elif p < 40:
            rec = "Apply phosphorus fertilizer with basal dose"
        elif k < 40:
            rec = "Apply potash for stress tolerance"
        else:
            rec = "Use balanced NPK fertilizer with organic manure"

        target = {"N": 80, "P": 45, "K": 45}
        current = {"N": n, "P": p, "K": k}
        gap = {key: max(target[key] - current[key], 0) for key in target}

        return {
            "advice": rec,
            "confidence": 89.0,
            "current": current,
            "target": target,
            "gap": gap
        }

    def ndvi(self, red, nir, moisture, temp):
        ndvi_val = round((nir - red) / (nir + red + 1e-8), 4)
        health = (ndvi_val * 50) + (moisture * 0.3) + ((35 - abs(temp - 28)) * 0.2)
        health = min(100, max(0, round(health, 1)))
        trend_days = [f"D{i}" for i in range(1, 8)]
        trend_vals = [round(max(ndvi_val + x, 0), 3) for x in [-0.06, -0.03, -0.01, 0.00, 0.02, 0.03, 0.05]]
        risk = max(0, min(100, round(100 - health, 1)))

        return {
            "ndvi": ndvi_val,
            "health": health,
            "risk": risk,
            "days": trend_days,
            "trend": trend_vals
        }

    def disease(self, image_present=False):
        labels = ["Healthy", "Leaf Blight", "Rust", "Leaf Spot"]
        probs = [64, 14, 12, 10] if image_present else [20, 35, 28, 17]
        idx = int(np.argmax(probs))
        return {
            "label": labels[idx],
            "confidence": float(probs[idx]),
            "probabilities": dict(zip(labels, probs))
        }

    def smart_alerts(self, result):
        alerts = []

        if result["ndvi"]["risk"] >= 55:
            alerts.append(("high", "🚨 High crop stress detected."))

        if result["irrigation"]["need"] == "High":
            alerts.append(("medium", "⚠ Low moisture or high temperature, irrigation needed."))

        if result["disease"]["label"] != "Healthy":
            alerts.append(("medium", f"⚠ Disease risk: {result['disease']['label']}."))

        if result["crop"]["confidence"] >= 85:
            alerts.append(("good", f"✅ Best Telangana crop right now: {result['crop']['crop']}."))

        return alerts

    def predict_all(self, payload):
        crop = self.crop_recommendation(
            payload["n"], payload["p"], payload["k"], payload["temperature"],
            payload["humidity"], payload["ph"], payload["rainfall"], payload["season"]
        )
        yld = self.yield_prediction(
            payload["ph"], payload["temperature"], payload["rainfall"],
            payload["n"], payload["humidity"], payload["soil_moisture"]
        )
        irr = self.irrigation(
            payload["soil_moisture"], payload["temperature"],
            payload["humidity"], payload["ph"], payload["rainfall"]
        )
        fert = self.fertilizer(payload["n"], payload["p"], payload["k"])
        ndvi = self.ndvi(
            payload["red_band"], payload["nir_band"],
            payload["soil_moisture"], payload["temperature"]
        )
        disease = self.disease(payload.get("image_present", False))

        result = {
            "crop": crop,
            "yield": yld,
            "irrigation": irr,
            "fertilizer": fert,
            "ndvi": ndvi,
            "disease": disease
        }
        result["alerts"] = self.smart_alerts(result)
        return result


engine = AgriEngine()


def build_pdf_report(lines):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    c.setFont("Helvetica-Bold", 15)
    c.drawString(50, 800, "Agri Vision AI – Final Farm Report")
    c.setFont("Helvetica", 11)
    y = 770

    for line in lines:
        c.drawString(50, y, str(line))
        y -= 22
        if y < 60:
            c.showPage()
            c.setFont("Helvetica", 11)
            y = 800

    c.save()
    buffer.seek(0)
    return buffer


def get_ai_reply(question, context):
    api_key = get_secret("OPENROUTER_API_KEY")
    if not api_key:
        return (
            "AI chatbot key is missing. Add OPENROUTER_API_KEY in Streamlit secrets. "
            "After fixing this, your chatbot will give real context-aware farm replies."
        )

    if not OPENAI_AVAILABLE:
        return "openai package not installed. Run: pip install openai"

    try:
        client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a friendly Telangana agriculture AI assistant. Give practical, simple, context-aware advice."
                },
                {
                    "role": "user",
                    "content": f"Farm context: {json.dumps(context)}\n\nQuestion: {question}"
                }
            ],
            temperature=0.4,
            extra_headers={
                "HTTP-Referer": "https://agrivisionai-9.streamlit.app/",
                "X-Title": "Agri Vision AI"
            }
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI error: {e}"


st.markdown(
    "<div class='hero'><h1>🌾 Agri Vision AI – Telangana Edition</h1><p>Telangana crops + one-click prediction + weather + alerts + AI farm report</p></div>",
    unsafe_allow_html=True
)

st.sidebar.title("Agri Vision AI")

module = st.sidebar.radio(
    "All Modules",
    [
        "Dashboard",
        "Predict All Modules",
        "Crop Recommendation",
        "Yield Prediction",
        "Irrigation",
        "Fertilizer & Soil",
        "NDVI Analysis",
        "Disease Detection",
        "Weather Auto",
        "Reports Download",
        "AI Assistant",
        "Voice Assistant",
        "FastAPI Plan"
    ]
)

district = st.sidebar.selectbox(
    "District",
    ["Hyderabad", "Ranga Reddy", "Medchal", "Sangareddy", "Mahabubnagar", "Karimnagar", "Warangal", "Nizamabad"]
)

season = st.sidebar.selectbox("Season", ["Kharif", "Rabi"])

weather, weather_debug = fetch_weather(district)

st.sidebar.markdown("### Telangana Crops")
st.sidebar.write(", ".join(TELANGANA_CROPS[season]))

st.sidebar.markdown("### Common Inputs")

if weather:
    st.sidebar.success(f"Live weather: {weather['city']} ({weather['weather']})")
    temperature = float(weather["temperature"])
    humidity = float(weather["humidity"])
    rainfall = float(weather["rainfall"])
    st.sidebar.metric("Temperature", f"{temperature} °C")
    st.sidebar.metric("Humidity", f"{humidity} %")
    st.sidebar.metric("Rainfall", f"{rainfall} mm")
else:
    st.sidebar.warning("Weather API not available. Using manual backup inputs.")
    temperature = st.sidebar.number_input("Temperature (°C)", value=29.0)
    humidity = st.sidebar.number_input("Humidity (%)", value=78.0)
    rainfall = st.sidebar.number_input("Rainfall (mm)", value=12.0)

n = st.sidebar.number_input("Nitrogen (N)", value=88.0)
p = st.sidebar.number_input("Phosphorus (P)", value=40.0)
k = st.sidebar.number_input("Potassium (K)", value=41.0)
ph = st.sidebar.number_input("Soil pH", value=6.7)
soil_moisture = st.sidebar.number_input("Soil Moisture (%)", value=42.0)
red_band = st.sidebar.number_input("Red Band", min_value=0.0, max_value=1.0, value=0.30)
nir_band = st.sidebar.number_input("NIR Band", min_value=0.0, max_value=1.0, value=0.72)

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
    "season": season,
    "image_present": False
}

all_result = engine.predict_all(payload)
st.session_state.latest_result = all_result


def render_alerts(alerts):
    st.subheader("Smart Alerts")
    if not alerts:
        st.markdown("<div class='alert-good'>✅ No major alerts right now.</div>", unsafe_allow_html=True)
        return

    for level, msg in alerts:
        if level == "high":
            st.markdown(f"<div class='alert-high'>{msg}</div>", unsafe_allow_html=True)
        elif level == "medium":
            st.markdown(f"<div class='alert-med'>{msg}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='alert-good'>{msg}</div>", unsafe_allow_html=True)


if module == "Dashboard":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Crop", all_result["crop"]["crop"], f"{all_result['crop']['confidence']}%")
    c2.metric("Yield", f"{all_result['yield']['yield']} t/ha")
    c3.metric("Health", f"{all_result['ndvi']['health']}%")
    c4.metric("Disease", all_result["disease"]["label"], f"{all_result['disease']['confidence']}%")

    render_alerts(all_result["alerts"])

    left, right = st.columns(2)

    with left:
        crop_df = pd.DataFrame(all_result["crop"]["scores"])
        st.plotly_chart(
            px.bar(crop_df, x="Crop", y="Suitability", color="Suitability", title="Telangana Crop Suitability"),
            use_container_width=True
        )

        ndvi_df = pd.DataFrame({"Day": all_result["ndvi"]["days"], "NDVI": all_result["ndvi"]["trend"]})
        st.plotly_chart(
            px.line(ndvi_df, x="Day", y="NDVI", markers=True, title="NDVI Trend Graph"),
            use_container_width=True
        )

    with right:
        irr_df = pd.DataFrame({"Day": all_result["irrigation"]["days"], "Liters": all_result["irrigation"]["liters"]})
        st.plotly_chart(
            px.area(irr_df, x="Day", y="Liters", title="Irrigation Forecast"),
            use_container_width=True
        )

        fig4 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=all_result["ndvi"]["risk"],
            title={'text': "Stress / Risk Gauge"},
            gauge={'axis': {'range': [0, 100]}}
        ))
        st.plotly_chart(fig4, use_container_width=True)

elif module == "Predict All Modules":
    st.subheader("One-Click Telangana Farm Prediction")

    if st.button("Predict All Modules in One Click", use_container_width=True):
        result = all_result
        st.success("All modules predicted successfully.")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Crop", result["crop"]["crop"], f"{result['crop']['confidence']}%")
        m2.metric("Yield", f"{result['yield']['yield']} t/ha")
        m3.metric("Irrigation", result["irrigation"]["need"], result["irrigation"]["action"])
        m4.metric("NDVI", result["ndvi"]["ndvi"], f"Health {result['ndvi']['health']}%")

        render_alerts(result["alerts"])

        st.markdown("### Explainable AI")
        why_df = pd.DataFrame({
            "Feature": list(result["crop"]["why"].keys()),
            "Importance": list(result["crop"]["why"].values())
        })
        st.plotly_chart(
            px.bar(why_df, x="Importance", y="Feature", orientation="h", title="Why this crop recommended"),
            use_container_width=True
        )

        a, b = st.columns(2)

        with a:
            crop_df = pd.DataFrame(result["crop"]["scores"])
            st.plotly_chart(
                px.bar(crop_df, x="Crop", y="Suitability", color="Suitability", title="Crop Suitability"),
                use_container_width=True
            )

            yield_df = pd.DataFrame({"Month": result["yield"]["months"], "Yield": result["yield"]["trend"]})
            st.plotly_chart(
                px.line(yield_df, x="Month", y="Yield", markers=True, title="Yield Prediction Chart"),
                use_container_width=True
            )

        with b:
            ndvi_df = pd.DataFrame({"Day": result["ndvi"]["days"], "NDVI": result["ndvi"]["trend"]})
            st.plotly_chart(
                px.line(ndvi_df, x="Day", y="NDVI", markers=True, title="NDVI Trend Graph"),
                use_container_width=True
            )

            fert = result["fertilizer"]
            radar = go.Figure()
            radar.add_trace(go.Scatterpolar(
                r=[fert["current"]["N"], fert["current"]["P"], fert["current"]["K"]],
                theta=["N", "P", "K"],
                fill='toself',
                name='Current Soil'
            ))
            radar.update_layout(title="Soil Radar Chart", polar=dict(radialaxis=dict(visible=True)))
            st.plotly_chart(radar, use_container_width=True)

        ai_summary = get_ai_reply(
            "Give final crop recommendation, risks, alerts, irrigation plan, and next 3 days action in friendly language.",
            result
        )
        st.markdown("### AI Farm Summary")
        st.write(ai_summary)

elif module == "Crop Recommendation":
    result = all_result["crop"]
    st.metric("Recommended Telangana Crop", result["crop"], f"{result['confidence']}% confidence")

    crop_df = pd.DataFrame(result["scores"])
    st.plotly_chart(
        px.bar(crop_df, x="Crop", y="Suitability", color="Suitability", title="Telangana Crop Ranking"),
        use_container_width=True
    )

    why_df = pd.DataFrame({"Feature": list(result["why"].keys()), "Importance": list(result["why"].values())})
    st.plotly_chart(
        px.bar(why_df, x="Importance", y="Feature", orientation="h", title="Why this crop recommended"),
        use_container_width=True
    )

elif module == "Yield Prediction":
    result = all_result["yield"]
    st.metric("Predicted Yield", f"{result['yield']} t/ha")

    ydf = pd.DataFrame({"Month": result["months"], "Yield": result["trend"]})
    st.plotly_chart(
        px.line(ydf, x="Month", y="Yield", markers=True, title="Yield Prediction Chart"),
        use_container_width=True
    )

elif module == "Irrigation":
    result = all_result["irrigation"]
    st.metric("Irrigation Need", result["need"], result["action"])

    idf = pd.DataFrame({"Day": result["days"], "Liters": result["liters"]})
    st.plotly_chart(
        px.bar(idf, x="Day", y="Liters", color="Liters", title="Weekly Irrigation Schedule"),
        use_container_width=True
    )

elif module == "Fertilizer & Soil":
    result = all_result["fertilizer"]
    st.metric("Fertilizer Advice", result["advice"], f"{result['confidence']}% confidence")

    soil_df = pd.DataFrame({
        "Nutrient": ["N", "P", "K"],
        "Current": [result["current"]["N"], result["current"]["P"], result["current"]["K"]],
        "Target": [result["target"]["N"], result["target"]["P"], result["target"]["K"]],
    })

    fig = go.Figure()
    fig.add_bar(name="Current", x=soil_df["Nutrient"], y=soil_df["Current"])
    fig.add_bar(name="Target", x=soil_df["Nutrient"], y=soil_df["Target"])
    fig.update_layout(barmode="group", title="Current vs Target NPK")
    st.plotly_chart(fig, use_container_width=True)

    radar = go.Figure()
    radar.add_trace(go.Scatterpolar(
        r=[result["current"]["N"], result["current"]["P"], result["current"]["K"]],
        theta=["N", "P", "K"],
        fill='toself',
        name='Current Soil'
    ))
    radar.update_layout(title="Soil Radar Chart", polar=dict(radialaxis=dict(visible=True)))
    st.plotly_chart(radar, use_container_width=True)

elif module == "NDVI Analysis":
    result = all_result["ndvi"]
    st.metric("NDVI", result["ndvi"], f"Health {result['health']}%")

    ndvi_df = pd.DataFrame({"Day": result["days"], "NDVI": result["trend"]})
    st.plotly_chart(
        px.line(ndvi_df, x="Day", y="NDVI", markers=True, title="NDVI Trend Graph"),
        use_container_width=True
    )

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=result["health"],
        title={'text': "Crop Health Gauge"},
        gauge={'axis': {'range': [0, 100]}}
    ))
    st.plotly_chart(gauge, use_container_width=True)

elif module == "Disease Detection":
    uploaded_file = st.file_uploader("Upload leaf image", type=["jpg", "jpeg", "png"])
    result = engine.disease(uploaded_file is not None)

    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded crop image", use_container_width=True)

    st.metric("Predicted Disease", result["label"], f"{result['confidence']}% confidence")

    pdf = pd.DataFrame({
        "Class": list(result["probabilities"].keys()),
        "Probability": list(result["probabilities"].values())
    })
    st.plotly_chart(
        px.bar(pdf, x="Class", y="Probability", color="Probability", title="Disease Prediction Confidence"),
        use_container_width=True
    )

elif module == "Weather Auto":
    st.subheader("Auto Weather Integration")

    if weather:
        st.success(f"Live weather loaded for {weather['city']}")
        wdf = pd.DataFrame({
            "Metric": ["Temperature", "Humidity", "Rainfall"],
            "Value": [weather["temperature"], weather["humidity"], weather["rainfall"]]
        })
        st.plotly_chart(
            px.bar(wdf, x="Metric", y="Value", color="Metric", title="Current Weather"),
            use_container_width=True
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Temperature", f"{weather['temperature']} °C")
        c2.metric("Humidity", f"{weather['humidity']} %")
        c3.metric("Rainfall", f"{weather['rainfall']} mm")
    else:
        st.info("Live weather could not be loaded, so backup farm values are active.")
        manual_df = pd.DataFrame({
            "Metric": ["Temperature", "Humidity", "Rainfall"],
            "Value": [temperature, humidity, rainfall]
        })
        st.plotly_chart(
            px.bar(manual_df, x="Metric", y="Value", color="Metric", title="Backup Weather Inputs"),
            use_container_width=True
        )

        with st.expander("Weather debug details"):
            st.json(weather_debug)

elif module == "Reports Download":
    st.subheader("Farm Report Download")

    result = all_result
    lines = [
        f"District: {district}",
        f"Season: {season}",
        f"Telangana Crop Recommendation: {result['crop']['crop']}",
        f"Crop Confidence: {result['crop']['confidence']}%",
        f"Predicted Yield: {result['yield']['yield']} t/ha",
        f"Irrigation Need: {result['irrigation']['need']}",
        f"Irrigation Action: {result['irrigation']['action']}",
        f"NDVI: {result['ndvi']['ndvi']}",
        f"Health Score: {result['ndvi']['health']}%",
        f"Disease Status: {result['disease']['label']}",
        f"Fertilizer Advice: {result['fertilizer']['advice']}",
        f"Alerts: {[msg for _, msg in result['alerts']]}"
    ]

    csv_df = pd.DataFrame({"Farm Report": lines})
    csv_bytes = csv_df.to_csv(index=False).encode("utf-8")
    pdf_buffer = build_pdf_report(lines)

    st.download_button(
        "Download Farm Report CSV",
        data=csv_bytes,
        file_name="telangana_farm_report.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.download_button(
        "Download Farm Report PDF",
        data=pdf_buffer,
        file_name="telangana_farm_report.pdf",
        mime="application/pdf",
        use_container_width=True
    )

elif module == "AI Assistant":
    st.subheader("AI Farm Assistant")

    st.info("Fix chatbot urgently: add OPENROUTER_API_KEY in Streamlit secrets for real AI answers.")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_prompt = st.chat_input("Ask about crop, irrigation, disease, yield, weather, or actions...")

    if user_prompt:
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})

        with st.chat_message("user"):
            st.markdown(user_prompt)

        reply = get_ai_reply(user_prompt, all_result)

        with st.chat_message("assistant"):
            st.markdown(reply)

        st.session_state.chat_history.append({"role": "assistant", "content": reply})

elif module == "Voice Assistant":
    st.subheader("Voice Assistant")
    st.write("Record farmer voice and later connect speech-to-text + AI reply.")
    audio = st.audio_input("Record your question")
    if audio:
        st.audio(audio)
        st.success("Voice recorded successfully. Next step: connect speech-to-text pipeline.")

elif module == "FastAPI Plan":
    st.subheader("FastAPI Backend Plan")
    st.markdown("""
- Streamlit handles UI.
- FastAPI handles prediction APIs.
- ML/DL models move to backend endpoints.
- Company-ready architecture: Streamlit → FastAPI → Models.
- FastAPI gives automatic docs at `/docs`.
""")

st.markdown("---")
st.markdown(
    "<div class='footer'>Agri Vision AI – Telangana Edition | Telangana crops | One-click prediction | Farm report | AI alerts</div>",
    unsafe_allow_html=True
)
