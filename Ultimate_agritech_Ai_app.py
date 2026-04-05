import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AgriVision AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CSS
# =========================
st.markdown("""
<style>
.main {background: linear-gradient(135deg, #020617 0%, #071226 45%, #0b1120 100%);}
.block-container {padding-top: 1rem; padding-bottom: 1rem;}
[data-testid="stAppViewContainer"] {background: linear-gradient(135deg, #020617 0%, #071226 45%, #0b1120 100%);}
[data-testid="stHeader"] {background: rgba(0,0,0,0);}
[data-testid="stSidebar"] {background: linear-gradient(180deg, #14532d 0%, #166534 50%, #14532d 100%);}
[data-testid="stSidebar"] * {color: white !important;}
.hero-dashboard {
    background: linear-gradient(90deg, #166534, #22c55e, #16a34a);
    color: white;
    padding: 1.8rem 1.5rem;
    border-radius: 24px;
    box-shadow: 0 12px 32px rgba(34, 197, 94, 0.18);
    margin-bottom: 1.5rem;
}
.metric-card {
    background: linear-gradient(135deg, #111827 0%, #172033 100%);
    border-radius: 20px;
    padding: 1.4rem;
    text-align: center;
    border: 1px solid rgba(148, 163, 184, 0.18);
    box-shadow: 0 10px 28px rgba(0,0,0,0.35);
    color: #e5e7eb;
}
.result-card, .health-card, .chart-card, .info-card {
    background: linear-gradient(135deg, #0f172a 0%, #162235 100%);
    border-radius: 18px;
    padding: 18px;
    border: 1px solid rgba(148,163,184,0.16);
    box-shadow: 0 10px 28px rgba(0,0,0,0.35);
    color: #e5e7eb;
    margin-bottom: 12px;
}
.farm-title {font-size: 2.35rem; font-weight: 800; margin: 0;}
.subtitle {margin: 0.4rem 0 0 0; font-size: 1.05rem; opacity: 0.96;}
.small-muted {color: #94a3b8; font-size: 0.9rem;}
.badge-good {
    background: #dcfce7;
    color: #166534;
    padding: 0.35rem 0.75rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 700;
}
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #111827 0%, #172033 100%);
    border: 1px solid rgba(148, 163, 184, 0.15);
    padding: 16px;
    border-radius: 18px;
    box-shadow: 0 10px 28px rgba(0,0,0,0.28);
}
div[data-testid="stMetricLabel"] {color: #94a3b8 !important;}
div[data-testid="stMetricValue"] {color: #f8fafc !important;}
div[data-testid="stMetricDelta"] {color: #22c55e !important;}
h1, h2, h3, h4, h5, h6, p, label, div, span {color: #f8fafc;}
</style>
""", unsafe_allow_html=True)

# =========================
# SESSION STATE
# =========================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "latest_farm_data" not in st.session_state:
    st.session_state.latest_farm_data = None
if "district" not in st.session_state:
    st.session_state.district = "Hyderabad"
if "season" not in st.session_state:
    st.session_state.season = "Kharif"

# =========================
# PLOT HELPERS
# =========================
def apply_dark_plotly(fig, title=None):
    fig.update_layout(
        template=None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0f172a",
        font=dict(color="#e5e7eb"),
        title_font=dict(color="#f8fafc", size=20),
        legend=dict(font=dict(color="#e5e7eb")),
        margin=dict(l=20, r=20, t=60, b=20)
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.15)", tickfont=dict(color="#cbd5e1"))
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.15)", tickfont=dict(color="#cbd5e1"))
    if title:
        fig.update_layout(title=title)
    return fig

def probability_bar_chart(prob_df, title="Class Confidence"):
    fig = px.bar(
        prob_df,
        x="Class",
        y="Confidence",
        color="Confidence",
        color_continuous_scale="Viridis",
        title=title
    )
    fig.update_traces(texttemplate="%{y:.2f}%", textposition="outside")
    fig = apply_dark_plotly(fig)
    fig.update_layout(coloraxis_showscale=False)
    return fig

def probability_donut_chart(prob_df, title="Prediction Share"):
    fig = px.pie(
        prob_df.head(5),
        names="Class",
        values="Confidence",
        hole=0.55,
        title=title
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb")
    )
    return fig

def gauge_chart(value, title, color="#22c55e", max_value=100):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title},
        gauge={
            "axis": {"range": [0, max_value]},
            "bar": {"color": color},
            "bgcolor": "#0f172a",
            "borderwidth": 1
        }
    ))
    fig.update_layout(
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb")
    )
    return fig

def radar_chart(categories, values, title):
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill="toself",
        line=dict(color="#22c55e", width=3),
        name=title
    ))
    fig.update_layout(
        title=title,
        paper_bgcolor="rgba(0,0,0,0)",
        polar=dict(
            bgcolor="#0f172a",
            radialaxis=dict(visible=True, gridcolor="rgba(148,163,184,0.15)", tickfont=dict(color="#cbd5e1")),
            angularaxis=dict(gridcolor="rgba(148,163,184,0.15)", tickfont=dict(color="#cbd5e1"))
        ),
        font=dict(color="#e5e7eb")
    )
    return fig

# =========================
# CNN MODEL SERVICE
# =========================
class CNNModelService:
    def __init__(self, model_dir="models"):
        self.model_dir = Path(model_dir)
        self.models = {}
        self.classes = {}
        self.errors = {}
        self.target_size = (224, 224)

        self.default_nutrient_classes = {
            0: "Healthy",
            1: "Nitrogen Deficiency",
            2: "Phosphorus Deficiency",
            3: "Potassium Deficiency",
            4: "Magnesium Deficiency",
            5: "Zinc Deficiency",
            6: "Iron Deficiency",
            7: "Sulphur Deficiency"
        }

        self._load_all()

    def _normalize_classes(self, loaded_classes, model_key=None):
        if loaded_classes is None:
            if model_key == "nutrient_deficiency":
                return self.default_nutrient_classes
            return []

        if isinstance(loaded_classes, dict):
            keys = list(loaded_classes.keys())
            values = list(loaded_classes.values())

            if all(isinstance(k, str) for k in keys) and all(isinstance(v, (int, np.integer)) for v in values):
                return {int(v): str(k) for k, v in loaded_classes.items()}

            if all(isinstance(k, (int, np.integer)) for k in keys):
                return {int(k): str(v) for k, v in loaded_classes.items()}

            return loaded_classes

        if isinstance(loaded_classes, (list, tuple, np.ndarray)):
            return list(loaded_classes)

        return loaded_classes

    def _load_one(self, model_key, model_file, classes_file=None, fallback_classes=None):
        model_path = self.model_dir / model_file

        if not model_path.exists():
            self.errors[model_key] = f"Missing model file: {model_file}"
            return

        if not TF_AVAILABLE:
            self.errors[model_key] = "TensorFlow not installed"
            return

        try:
            self.models[model_key] = tf.keras.models.load_model(model_path)

            loaded_classes = None
            if classes_file:
                classes_path = self.model_dir / classes_file
                if classes_path.exists():
                    loaded_classes = joblib.load(classes_path)
                else:
                    loaded_classes = fallback_classes
            else:
                loaded_classes = fallback_classes

            self.classes[model_key] = self._normalize_classes(loaded_classes, model_key=model_key)

        except Exception as e:
            self.errors[model_key] = str(e)

    def _load_all(self):
        self._load_one("tomato_disease", "tomato_disease_model.h5", "tomato_classes.joblib")
        self._load_one("rice_disease", "rice_disease_model.h5", "rice_classes.joblib")
        self._load_one("rice_pest", "rice_pest_model.h5", "rice_pest_classes.joblib")
        self._load_one("leaf_detection", "leaf_model.h5", "leaf_classes.joblib")
        self._load_one("nutrient_deficiency", "nutrient_deficiency_cnn_model.keras", "nutrient_deficiency_classes.joblib", fallback_classes=self.default_nutrient_classes)

    def has_model(self, model_key):
        return model_key in self.models

    def get_error(self, model_key):
        return self.errors.get(model_key, "Model not loaded")

    def preprocess_image(self, pil_image):
        image = pil_image.convert("RGB").resize(self.target_size)
        arr = np.array(image, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)
        return image, arr

    def _get_class_names(self, model_key):
        class_map = self.classes.get(model_key, [])
        if isinstance(class_map, dict):
            max_idx = max(class_map.keys()) if class_map else -1
            return [class_map.get(i, f"Class {i}") for i in range(max_idx + 1)]
        if isinstance(class_map, (list, tuple, np.ndarray)):
            return list(class_map)
        return []

    def predict(self, pil_image, model_key):
        if model_key not in self.models:
            raise ValueError(self.get_error(model_key))

        _, arr = self.preprocess_image(pil_image)
        pred = self.models[model_key].predict(arr, verbose=0)[0]
        pred = np.array(pred).flatten()

        class_names = self._get_class_names(model_key)
        if not class_names:
            class_names = [f"Class {i}" for i in range(len(pred))]

        top_idx = int(np.argmax(pred))
        top_class = class_names[top_idx] if top_idx < len(class_names) else f"Class {top_idx}"
        top_conf = float(pred[top_idx]) * 100

        prob_df = pd.DataFrame({
            "Class": class_names[:len(pred)],
            "Confidence": np.round(pred[:len(class_names)] * 100, 2)
        }).sort_values("Confidence", ascending=False)

        return {
            "class_index": top_idx,
            "class_name": top_class,
            "confidence": round(top_conf, 2),
            "probabilities": prob_df
        }

cnn_service = CNNModelService("models")

# =========================
# ML MODELS
# =========================
@st.cache_resource
def train_ml_models():
    crop_names = ["Rice", "Cotton", "Maize", "Wheat", "Sugarcane", "Soybean", "Groundnut", "Chilli", "Turmeric", "Red Gram"]
    crop_profiles = {
        "Rice": [80, 40, 40, 28, 80, 6.5, 180],
        "Cotton": [70, 35, 50, 30, 60, 7.0, 90],
        "Maize": [85, 45, 40, 27, 65, 6.7, 110],
        "Wheat": [65, 35, 35, 22, 55, 6.8, 60],
        "Sugarcane": [90, 50, 55, 29, 70, 7.2, 160],
        "Soybean": [60, 40, 35, 26, 65, 6.5, 95],
        "Groundnut": [55, 35, 45, 28, 55, 6.6, 70],
        "Chilli": [75, 40, 45, 27, 58, 6.8, 65],
        "Turmeric": [70, 45, 50, 26, 75, 6.4, 140],
        "Red Gram": [45, 30, 35, 29, 55, 7.0, 75],
    }

    X_crop, y_crop = [], []
    for crop in crop_names:
        base = crop_profiles[crop]
        for _ in range(35):
            n = np.clip(np.random.normal(base[0], 8), 10, 120)
            p = np.clip(np.random.normal(base[1], 6), 10, 80)
            k = np.clip(np.random.normal(base[2], 6), 10, 80)
            temp = np.clip(np.random.normal(base[3], 3), 15, 40)
            humidity = np.clip(np.random.normal(base[4], 8), 30, 95)
            ph = np.clip(np.random.normal(base[5], 0.4), 4.5, 8.5)
            rainfall = np.clip(np.random.normal(base[6], 25), 0, 300)
            X_crop.append([n, p, k, temp, humidity, ph, rainfall])
            y_crop.append(crop)

    crop_encoder = LabelEncoder()
    y_crop_enc = crop_encoder.fit_transform(y_crop)
    crop_model = RandomForestClassifier(n_estimators=220, max_depth=14, random_state=42)
    crop_model.fit(X_crop, y_crop_enc)

    X_yield, y_yield = [], []
    crop_factor_map = {crop: i + 1 for i, crop in enumerate(crop_names)}
    for crop in crop_names:
        crop_factor = crop_factor_map[crop]
        base = crop_profiles[crop]
        for _ in range(40):
            ph = np.clip(np.random.normal(base[5], 0.5), 4.5, 8.5)
            temp = np.clip(np.random.normal(base[3], 3), 15, 40)
            rainfall = np.clip(np.random.normal(base[6], 30), 0, 300)
            fertilizer = np.clip(np.random.normal(base[0], 10), 10, 140)
            humidity = np.clip(np.random.normal(base[4], 8), 30, 95)
            soil_moisture = np.clip(np.random.normal(42, 12), 10, 80)
            y = 1.2 + crop_factor * 0.12 + temp * 0.05 + rainfall * 0.01 + fertilizer * 0.015 + humidity * 0.02 + soil_moisture * 0.03 + np.random.normal(0, 0.3)
            X_yield.append([ph, temp, rainfall, fertilizer, humidity, soil_moisture, crop_factor])
            y_yield.append(max(0.8, y))

    yield_model = RandomForestRegressor(n_estimators=260, max_depth=16, random_state=42)
    yield_model.fit(X_yield, y_yield)

    X_irr, y_irr = [], []
    for _ in range(400):
        soil_m = np.random.uniform(10, 80)
        temp = np.random.uniform(15, 40)
        hum = np.random.uniform(30, 95)
        ph = np.random.uniform(4.5, 8.5)
        rain = np.random.uniform(0, 250)
        if rain > 120 or soil_m > 60:
            label = "Low"
        elif soil_m < 28 and temp > 30:
            label = "High"
        else:
            label = "Moderate"
        X_irr.append([soil_m, temp, hum, ph, rain])
        y_irr.append(label)

    irr_encoder = LabelEncoder()
    y_irr_enc = irr_encoder.fit_transform(y_irr)
    irrigation_model = RandomForestClassifier(n_estimators=180, max_depth=10, random_state=42)
    irrigation_model.fit(X_irr, y_irr_enc)

    return {
        "crop_model": crop_model,
        "crop_encoder": crop_encoder,
        "yield_model": yield_model,
        "irrigation_model": irrigation_model,
        "irrigation_encoder": irr_encoder,
        "crop_factor_map": crop_factor_map,
        "all_crops": crop_names
    }

ml_bundle = train_ml_models()

class AgriAPIService:
    def __init__(self, bundle):
        self.bundle = bundle

    def recommend_crop(self, features):
        pred = self.bundle["crop_model"].predict([features])[0]
        crop = self.bundle["crop_encoder"].inverse_transform([pred])[0]
        probs = self.bundle["crop_model"].predict_proba([features])[0]
        labels = list(self.bundle["crop_encoder"].classes_)
        prob_df = pd.DataFrame({
            "Class": labels,
            "Confidence": np.round(probs * 100, 2)
        }).sort_values("Confidence", ascending=False)
        conf = round(float(np.max(probs)) * 100, 2)
        return crop, conf, prob_df

    def predict_yield(self, ph, temp, rainfall, fertilizer, humidity, soil_moisture, crop_name):
        crop_factor = self.bundle["crop_factor_map"].get(crop_name, 1)
        pred = self.bundle["yield_model"].predict([[ph, temp, rainfall, fertilizer, humidity, soil_moisture, crop_factor]])[0]
        return round(float(pred), 2)

    def predict_irrigation(self, soil_moisture, temp, humidity, ph, rainfall):
        pred = self.bundle["irrigation_model"].predict([[soil_moisture, temp, humidity, ph, rainfall]])[0]
        probs = self.bundle["irrigation_model"].predict_proba([[soil_moisture, temp, humidity, ph, rainfall]])[0]
        label = self.bundle["irrigation_encoder"].inverse_transform([pred])[0]
        class_names = list(self.bundle["irrigation_encoder"].classes_)
        prob_df = pd.DataFrame({"Class": class_names, "Confidence": np.round(probs * 100, 2)}).sort_values("Confidence", ascending=False)
        if label == "Low":
            action = "Skip irrigation cycle"
        elif label == "High":
            action = "Deep irrigation needed"
        else:
            action = "Light irrigation"
        return label, action, prob_df

api_service = AgriAPIService(ml_bundle)

def calculate_ndvi(red, nir):
    if (nir + red) == 0:
        return 0.0
    return round((nir - red) / (nir + red), 4)

def predict_health_score(ndvi, moisture, temp):
    score = (ndvi * 50) + (moisture * 0.3) + ((35 - abs(temp - 28)) * 0.2)
    return min(100, max(0, round(score, 1)))

def fertilizer_advice(n, p, k):
    if n < 50:
        return "Apply nitrogen fertilizer in split doses", 87.0
    if p < 40:
        return "Apply phosphorus fertilizer with basal dose", 86.0
    if k < 40:
        return "Apply potash for stress tolerance", 85.0
    return "Use balanced NPK fertilizer with organic manure", 89.0

def process_image(uploaded_file):
    return Image.open(uploaded_file).convert("RGB")

def render_header():
    st.markdown(
        f"""
        <div class="hero-dashboard">
            <div style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:1rem;'>
                <div>
                    <h1 class="farm-title">AgriVision AI</h1>
                    <p class="subtitle">Farm Health AI</p>
                </div>
                <div style='text-align:right;'>
                    <div><span class="badge-good">Telangana Edition</span></div>
                    <div style='margin-top:0.5rem;font-size:0.95rem;'>📍 {st.session_state.district} | 🌾 {st.session_state.season}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_top_dashboard():
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("<div class='metric-card'><div class='small-muted'>Overall Health</div><h2 style='color:#22c55e;'>87.4%</h2><div class='small-muted'>+3.2% this week</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='metric-card'><div class='small-muted'>Active Alerts</div><h2 style='color:#f87171;'>4</h2><div class='small-muted'>2 critical zones</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='metric-card'><div class='small-muted'>Yield Forecast</div><h2 style='color:#86efac;'>4.2 t/ha</h2><div class='small-muted'>On target</div></div>", unsafe_allow_html=True)
    with c4:
        st.markdown("<div class='metric-card'><div class='small-muted'>Soil Moisture</div><h2 style='color:#60a5fa;'>62%</h2><div class='small-muted'>Optimal</div></div>", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🌾 AgriVision AI")
page = st.sidebar.radio(
    "Choose Module",
    [
        "Dashboard",
        "Smart Advisor",
        "Multi-Crop Analysis",
        "Crop Recommendation",
        "Yield Prediction",
        "Irrigation",
        "Fertilizer & Soil",
        "NDVI Analysis",
        "Disease Detection",
        "Pest Detection",
        "Leaf Detection",
        "Nutrient Deficiency",
        "AI Assistant"
    ]
)
st.sidebar.markdown("### Telangana Filters")
st.session_state.district = st.sidebar.selectbox("District", ["Hyderabad", "Ranga Reddy", "Medchal", "Sangareddy", "Mahabubnagar", "Warangal", "Karimnagar", "Nizamabad", "Khammam", "Nalgonda"])
st.session_state.season = st.sidebar.selectbox("Season", ["Kharif", "Rabi"])

render_header()
render_top_dashboard()

if page == "Dashboard":
    st.markdown("## Dashboard")

    line_df = pd.DataFrame({
        "Day": [f"Day {i}" for i in range(1, 16)],
        "Health Index": [72, 74, 73, 76, 77, 79, 80, 81, 83, 82, 84, 85, 87, 88, 89],
        "Moisture": [54, 55, 56, 58, 57, 59, 60, 62, 61, 63, 64, 65, 66, 67, 68]
    })
    pie_df = pd.DataFrame({"Zone": ["Healthy", "Moderate", "Risk", "Critical"], "Value": [52, 28, 14, 6]})
    bar_df = pd.DataFrame({"District": ["Hyderabad", "Warangal", "Nizamabad", "Khammam"], "Yield": [4.2, 4.8, 3.9, 4.5]})
    heat_df = pd.DataFrame(
        np.array([[82, 76, 68], [88, 71, 64], [79, 73, 59], [91, 80, 70]]),
        columns=["Health", "Moisture", "Yield"],
        index=["Zone A", "Zone B", "Zone C", "Zone D"]
    )

    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(line_df, x="Day", y=["Health Index", "Moisture"], title="Health and Moisture Trend")
        fig = apply_dark_plotly(fig)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.pie(pie_df, names="Zone", values="Value", hole=0.55, title="Risk Zone Distribution")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e5e7eb"))
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig = px.bar(bar_df, x="District", y="Yield", color="Yield", title="District Yield Comparison", color_continuous_scale="Viridis")
        fig = apply_dark_plotly(fig)
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        fig = px.imshow(heat_df, text_auto=True, color_continuous_scale="YlGn", title="Zone Heatmap")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e5e7eb"))
        st.plotly_chart(fig, use_container_width=True)

elif page == "Smart Advisor":
    st.markdown("## Smart Advisor")
    c1, c2, c3 = st.columns(3)
    with c1:
        n = st.number_input("Nitrogen (N)", value=88.0)
        p = st.number_input("Phosphorus (P)", value=40.0)
        k = st.number_input("Potassium (K)", value=41.0)
        ph = st.number_input("Soil pH", value=6.7)
    with c2:
        temp = st.number_input("Temperature (°C)", value=29.0)
        humidity = st.number_input("Humidity (%)", value=78.0)
        rainfall = st.number_input("Rainfall (mm)", value=180.0)
    with c3:
        moisture = st.number_input("Soil Moisture (%)", value=42.0)
        red = st.number_input("Red Band", min_value=0.0, max_value=1.0, value=0.30)
        nir = st.number_input("NIR Band", min_value=0.0, max_value=1.0, value=0.72)

    if st.button("Run Analysis", type="primary", use_container_width=True):
        crop, crop_conf, crop_prob_df = api_service.recommend_crop([n, p, k, temp, humidity, ph, rainfall])
        irrigation, action, irr_prob_df = api_service.predict_irrigation(moisture, temp, humidity, ph, rainfall)
        yield_pred = api_service.predict_yield(ph, temp, rainfall, n, humidity, moisture, crop)
        ndvi = calculate_ndvi(red, nir)
        health = predict_health_score(ndvi, moisture, temp)
        fert, fert_conf = fertilizer_advice(n, p, k)

        st.session_state.latest_farm_data = {
            "district": st.session_state.district,
            "season": st.session_state.season,
            "crop": crop,
            "irrigation": irrigation,
            "action": action,
            "yield": yield_pred,
            "ndvi": ndvi,
            "health": health,
            "fertilizer": fert
        }

        a, b, c = st.columns(3)
        a.metric("Recommended Crop", crop, f"{crop_conf}% confidence")
        b.metric("Predicted Yield", f"{yield_pred} t/ha", "ML output")
        c.metric("Health Score", f"{health}%", f"NDVI {ndvi}")

        x1, x2 = st.columns(2)
        with x1:
            st.plotly_chart(probability_bar_chart(crop_prob_df.head(5), "Top Crop Recommendation Confidence"), use_container_width=True)
        with x2:
            st.plotly_chart(probability_bar_chart(irr_prob_df, "Irrigation Prediction Confidence"), use_container_width=True)

        x3, x4 = st.columns(2)
        with x3:
            st.plotly_chart(radar_chart(["N", "P", "K", "Moisture", "Humidity"], [n, p, k, moisture, humidity], "Farm Parameter Radar"), use_container_width=True)
        with x4:
            st.plotly_chart(gauge_chart(health, "Farm Health Gauge"), use_container_width=True)

        trend_df = pd.DataFrame({"Day": [f"Day {i}" for i in range(1, 8)], "NDVI": [max(0.1, round(ndvi - 0.05 + i * 0.01, 3)) for i in range(1, 8)]})
        fig = px.area(trend_df, x="Day", y="NDVI", title="NDVI Forecast Trend")
        fig = apply_dark_plotly(fig)
        st.plotly_chart(fig, use_container_width=True)

elif page == "Multi-Crop Analysis":
    st.markdown("## Multi-Crop Analysis")
    crop_df = pd.DataFrame({
        "Crop": ml_bundle["all_crops"],
        "Expected Yield": [round(api_service.predict_yield(6.7, 29, 120, 80, 70, 42, crop), 2) for crop in ml_bundle["all_crops"]]
    }).sort_values("Expected Yield", ascending=False)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(crop_df, x="Crop", y="Expected Yield", color="Expected Yield", title="Multi-Crop Yield Comparison", color_continuous_scale="Viridis")
        fig = apply_dark_plotly(fig)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.treemap(crop_df, path=["Crop"], values="Expected Yield", color="Expected Yield", color_continuous_scale="YlGn")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e5e7eb"), title="Crop Yield Treemap")
        st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(crop_df, x="Crop", y="Expected Yield", size="Expected Yield", color="Expected Yield", title="Crop Yield Bubble View", color_continuous_scale="Turbo")
    fig = apply_dark_plotly(fig)
    st.plotly_chart(fig, use_container_width=True)

elif page == "Crop Recommendation":
    st.markdown("## Crop Recommendation")
    vals = [st.number_input(f"Feature {i+1}", value=float(80 + i)) for i in range(7)]
    if st.button("Recommend Crop", use_container_width=True):
        crop, conf, prob_df = api_service.recommend_crop(vals)
        st.success(f"Recommended crop: {crop} ({conf}% confidence)")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(probability_bar_chart(prob_df.head(7), "Crop Recommendation Probability"), use_container_width=True)
        with c2:
            st.plotly_chart(probability_donut_chart(prob_df, "Top Crop Share"), use_container_width=True)

elif page == "Yield Prediction":
    st.markdown("## Yield Prediction")
    ph = st.number_input("Soil pH", value=6.7, key="yield_ph")
    temp = st.number_input("Temperature", value=29.0, key="yield_temp")
    rainfall = st.number_input("Rainfall", value=120.0, key="yield_rain")
    fertilizer = st.number_input("Fertilizer", value=85.0, key="yield_fert")
    humidity = st.number_input("Humidity", value=70.0, key="yield_hum")
    soil_moisture = st.number_input("Soil Moisture", value=42.0, key="yield_sm")
    crop_name = st.selectbox("Crop", ml_bundle["all_crops"])
    if st.button("Predict Yield", use_container_width=True):
        y = api_service.predict_yield(ph, temp, rainfall, fertilizer, humidity, soil_moisture, crop_name)
        st.metric("Expected Yield", f"{y} t/ha", "ML prediction")

        compare_df = pd.DataFrame({
            "Factor": ["pH", "Temp", "Rainfall", "Fertilizer", "Humidity", "Soil Moisture"],
            "Value": [ph, temp, rainfall, fertilizer, humidity, soil_moisture]
        })
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(compare_df, x="Factor", y="Value", color="Value", title="Yield Input Factors", color_continuous_scale="Viridis")
            fig = apply_dark_plotly(fig)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.plotly_chart(gauge_chart(y, "Yield Gauge", color="#86efac", max_value=12), use_container_width=True)

elif page == "Irrigation":
    st.markdown("## Irrigation")
    soil_moisture = st.number_input("Soil Moisture", value=42.0, key="irr_sm")
    temp = st.number_input("Temperature", value=29.0, key="irr_temp")
    humidity = st.number_input("Humidity", value=70.0, key="irr_hum")
    ph = st.number_input("pH", value=6.8, key="irr_ph")
    rainfall = st.number_input("Rainfall", value=100.0, key="irr_rain")
    if st.button("Get Irrigation Plan", use_container_width=True):
        result, action, irr_prob_df = api_service.predict_irrigation(soil_moisture, temp, humidity, ph, rainfall)
        st.info(f"Need: {result} | Action: {action}")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(probability_bar_chart(irr_prob_df, "Irrigation Category Confidence"), use_container_width=True)
        with c2:
            scatter_df = pd.DataFrame({"Rainfall": [rainfall], "Soil Moisture": [soil_moisture], "Temperature": [temp]})
            fig = px.scatter(scatter_df, x="Rainfall", y="Soil Moisture", size="Temperature", color="Temperature", title="Water Stress Scatter")
            fig = apply_dark_plotly(fig)
            st.plotly_chart(fig, use_container_width=True)

elif page == "Fertilizer & Soil":
    st.markdown("## Fertilizer & Soil")
    n = st.number_input("Nitrogen", value=45.0)
    p = st.number_input("Phosphorus", value=38.0)
    k = st.number_input("Potassium", value=42.0)
    if st.button("Analyze Soil", use_container_width=True):
        fert, conf = fertilizer_advice(n, p, k)
        st.success(f"Recommendation: {fert} ({conf}% confidence)")
        soil_df = pd.DataFrame({"Nutrient": ["N", "P", "K"], "Value": [n, p, k]})
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(soil_df, x="Nutrient", y="Value", color="Nutrient", title="Soil Nutrient Levels",
                         color_discrete_map={"N": "#22c55e", "P": "#38bdf8", "K": "#f59e0b"})
            fig = apply_dark_plotly(fig)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.plotly_chart(radar_chart(["Nitrogen", "Phosphorus", "Potassium"], [n, p, k], "Soil Radar"), use_container_width=True)

        heat_df = pd.DataFrame(np.array([[n, p, k]]), columns=["N", "P", "K"], index=["Current"])
        fig = px.imshow(heat_df, text_auto=True, color_continuous_scale="YlOrBr", title="Soil Heatmap")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e5e7eb"))
        st.plotly_chart(fig, use_container_width=True)

elif page == "NDVI Analysis":
    st.markdown("## NDVI Analysis")
    red = st.slider("Red Band", 0.0, 1.0, 0.30)
    nir = st.slider("NIR Band", 0.0, 1.0, 0.72)
    temp = st.slider("Temperature", 15.0, 45.0, 29.0)
    moisture = st.slider("Soil Moisture", 10.0, 80.0, 42.0)
    if st.button("Calculate NDVI", use_container_width=True):
        ndvi = calculate_ndvi(red, nir)
        health = predict_health_score(ndvi, moisture, temp)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(gauge_chart(health, "Health Score Gauge"), use_container_width=True)
        with c2:
            st.plotly_chart(gauge_chart(ndvi, "NDVI Gauge", color="#38bdf8", max_value=1), use_container_width=True)

        trend_df = pd.DataFrame({"Day": [f"Day {i}" for i in range(1, 8)], "NDVI": [max(0.1, round(ndvi - 0.04 + i * 0.01, 3)) for i in range(1, 8)]})
        fig = px.line(trend_df, x="Day", y="NDVI", markers=True, title="NDVI Trend")
        fig = apply_dark_plotly(fig)
        st.plotly_chart(fig, use_container_width=True)

        scatter_df = pd.DataFrame({"Band": ["Red", "NIR"], "Value": [red, nir]})
        fig = px.bar(scatter_df, x="Band", y="Value", color="Band", title="Red vs NIR Band")
        fig = apply_dark_plotly(fig)
        st.plotly_chart(fig, use_container_width=True)

elif page == "Disease Detection":
    st.markdown("## Disease Detection")
    crop_type = st.selectbox("Select Crop for Disease Detection", ["Tomato", "Rice"])
    uploaded_file = st.file_uploader("Upload crop leaf image", type=["jpg", "jpeg", "png"], key="disease_upload")
    if uploaded_file:
        image = process_image(uploaded_file)
        st.image(image, caption="Uploaded image", use_container_width=True)
        if st.button("Analyze Disease", use_container_width=True):
            try:
                model_key = "tomato_disease" if crop_type == "Tomato" else "rice_disease"
                result = cnn_service.predict(image, model_key)
                st.markdown(f"<div class='result-card'><h3>{crop_type} Disease Prediction</h3><p><strong>Predicted class:</strong> {result['class_name']}</p><p><strong>Confidence:</strong> {result['confidence']}%</p></div>", unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(probability_bar_chart(result["probabilities"].head(5), "Top Disease Class Confidence"), use_container_width=True)
                with c2:
                    st.plotly_chart(probability_donut_chart(result["probabilities"], "Disease Prediction Distribution"), use_container_width=True)
            except Exception as e:
                st.error(f"Prediction error: {e}")

elif page == "Pest Detection":
    st.markdown("## Pest Detection")
    uploaded_file = st.file_uploader("Upload rice pest image", type=["jpg", "jpeg", "png"], key="pest_upload")
    if uploaded_file:
        image = process_image(uploaded_file)
        st.image(image, caption="Uploaded pest image", use_container_width=True)
        if st.button("Analyze Pest", use_container_width=True):
            try:
                result = cnn_service.predict(image, "rice_pest")
                st.markdown(f"<div class='result-card'><h3>Rice Pest Prediction</h3><p><strong>Predicted class:</strong> {result['class_name']}</p><p><strong>Confidence:</strong> {result['confidence']}%</p></div>", unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(probability_bar_chart(result["probabilities"].head(5), "Rice Pest Top Confidence"), use_container_width=True)
                with c2:
                    st.plotly_chart(probability_donut_chart(result["probabilities"], "Pest Distribution"), use_container_width=True)
            except Exception as e:
                st.error(f"Prediction error: {e}")

elif page == "Leaf Detection":
    st.markdown("## Leaf Detection")
    uploaded_file = st.file_uploader("Upload leaf image", type=["jpg", "jpeg", "png"], key="leaf_upload")
    if uploaded_file:
        image = process_image(uploaded_file)
        st.image(image, caption="Uploaded leaf image", use_container_width=True)
        if st.button("Analyze Leaf", use_container_width=True):
            try:
                result = cnn_service.predict(image, "leaf_detection")
                st.markdown(f"<div class='result-card'><h3>Leaf Detection Prediction</h3><p><strong>Predicted class:</strong> {result['class_name']}</p><p><strong>Confidence:</strong> {result['confidence']}%</p></div>", unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(probability_bar_chart(result["probabilities"].head(5), "Leaf Detection Confidence"), use_container_width=True)
                with c2:
                    st.plotly_chart(probability_donut_chart(result["probabilities"], "Leaf Prediction Distribution"), use_container_width=True)
            except Exception as e:
                st.error(f"Prediction error: {e}")

elif page == "Nutrient Deficiency":
    st.markdown("## Nutrient Deficiency")
    uploaded_file = st.file_uploader("Upload nutrient deficiency leaf image", type=["jpg", "jpeg", "png"], key="nutrient_upload")
    if uploaded_file:
        image = process_image(uploaded_file)
        st.image(image, caption="Uploaded nutrient image", use_container_width=True)
        if st.button("Analyze Nutrient Deficiency", use_container_width=True):
            try:
                result = cnn_service.predict(image, "nutrient_deficiency")
                st.markdown(f"<div class='result-card'><h3>Nutrient Deficiency Prediction</h3><p><strong>Predicted class:</strong> {result['class_name']}</p><p><strong>Confidence:</strong> {result['confidence']}%</p></div>", unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(probability_bar_chart(result["probabilities"].head(5), "Nutrient Deficiency Confidence"), use_container_width=True)
                with c2:
                    st.plotly_chart(probability_donut_chart(result["probabilities"], "Nutrient Prediction Distribution"), use_container_width=True)
            except Exception as e:
                st.error(f"Prediction error: {e}")

elif page == "AI Assistant":
    st.markdown("## AI Assistant")
    st.info("Ask questions based on crop, NDVI, irrigation, fertilizer, disease, or yield.")
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask anything about your farm analysis..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        response = "AgriVision AI assistant is active. Use Smart Advisor and detection modules for data-backed guidance."
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.chat_history.append({"role": "assistant", "content": response})

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#94a3b8;padding:18px;font-size:14px;'>🌾 AgriVision AI | Advanced Visualization Edition | Telangana</div>",
    unsafe_allow_html=True
)
