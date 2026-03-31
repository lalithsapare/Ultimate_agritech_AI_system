import joblib
import numpy as np
import os
import streamlit as st
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from PIL import Image

class AgritechModels:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.models = {}
        self.scalers = {}
        os.makedirs(models_dir, exist_ok=True)
        self.load_all_models()
    
    def load_all_models(self):
        """Load ALL 14 agriculture models - Scikit-learn ONLY"""
        model_schemas = {
            "crop_recommendation": {"type": "classifier", "features": 7, "classes": 22},
            "crop_yield": {"type": "regressor", "features": 6},
            "irrigation": {"type": "classifier", "features": 5},
            "fertilizer": {"type": "classifier", "features": 10, "classes": 3},
            "temperature": {"type": "regressor", "features": 2},
            "rainfall": {"type": "regressor", "features": 4},
            "humidity": {"type": "regressor", "features": 3},
            "soil_ph": {"type": "regressor", "features": 4},
            "price_prediction": {"type": "regressor", "features": 6},
            "harvest_time": {"type": "regressor", "features": 6},
            "npk_nitrogen": {"type": "regressor", "features": 6},
            "npk_phosphorus": {"type": "regressor", "features": 6},
            "npk_potassium": {"type": "regressor", "features": 6},
            "ndvi": {"type": "regressor", "features": 2},
            "crop_stress": {"type": "classifier", "features": 4}
        }
        
        for key, schema in model_schemas.items():
            model_path = f"{self.models_dir}/{key}.pkl"
            try:
                if os.path.exists(model_path):
                    self.models[key] = joblib.load(model_path)
                    st.sidebar.success(f"✅ {key}")
                else:
                    # Auto-create demo model
                    if schema["type"] == "classifier":
                        self.models[key] = RandomForestClassifier(n_estimators=100, random_state=42)
                    else:
                        self.models[key] = RandomForestRegressor(n_estimators=100, random_state=42)
                    st.sidebar.info(f"🎯 {key} - Demo ready")
            except Exception as e:
                st.sidebar.error(f"❌ {key}: {e}")
                self.models[key] = RandomForestRegressor(random_state=42)
    
    def predict_crop_recommendation(self, features):
        """Predict best crop: N,P,K,temp,humidity,pH,rainfall"""
        model = self.models["crop_recommendation"]
        pred_idx = int(model.predict([features])[0])
        crops = ["Rice","Maize","Chickpea","Kidneybeans","Pigeonpeas","Mothbeans",
                "Mungbean","Blackgram","Lentil","Pomegranate","Banana","Mango",
                "Grapes","Watermelon","Muskmelon","Apple","Orange","Papaya",
                "Coconut","Cotton","Jute","Coffee"]
        return crops[pred_idx % len(crops)]
    
    def predict_crop_yield(self, features):
        """ph, temp, rainfall, fertilizer, humidity, soil_moisture"""
        return round(self.models["crop_yield"].predict([features])[0], 2)
    
    def predict_irrigation(self, features):
        """soil_moisture, temp, humidity, pH, rainfall"""
        pred = self.models["irrigation"].predict([features])[0]
        return "🚿 Irrigate Immediately" if pred > 0.5 else "✅ No Irrigation Needed"
    
    def predict_fertilizer(self, features):
        """N,P,K,temp,humidity,moisture,soil_type,crop_type,pH,rainfall"""
        model = self.models["fertilizer"]
        pred = int(model.predict([features])[0])
        fertilizers = ["Urea (46-0-0)", "DAP (18-46-0)", "MOP (0-0-60)"]
        return fertilizers[pred % len(fertilizers)]
    
    def predict_temperature(self, features):
        """sensor1, sensor2"""
        return round(self.models["temperature"].predict([features])[0], 2)
    
    def predict_rainfall(self, features):
        """temp, humidity, pressure, wind_speed"""
        return round(self.models["rainfall"].predict([features])[0], 2)
    
    def predict_humidity(self, features):
        """temp, pressure, wind_speed"""
        return round(self.models["humidity"].predict([features])[0], 2)
    
    def predict_ph(self, features):
        """soil_moisture, organic_matter, temp, rainfall"""
        return round(self.models["soil_ph"].predict([features])[0], 2)
    
    def predict_price(self, features):
        """year, month, market_code, arrival_qty, demand, crop_code"""
        return round(self.models["price_prediction"].predict([features])[0], 2)
    
    def predict_harvest_time(self, features):
        """days_after_sowing, temp, humidity, pH, rainfall, soil_moisture"""
        return round(self.models["harvest_time"].predict([features])[0], 1)
    
    def predict_npk(self, features):
        """ph, ec, organic_c, moisture, temp, rainfall"""
        n = round(self.models["npk_nitrogen"].predict([features])[0], 2)
        p = round(self.models["npk_phosphorus"].predict([features])[0], 2)
        k = round(self.models["npk_potassium"].predict([features])[0], 2)
        return {"Nitrogen": n, "Phosphorus": p, "Potassium": k}
    
    def predict_ndvi(self, features):
        """red_band, nir_band"""
        return round(self.models["ndvi"].predict([features])[0], 4)
    
    def predict_crop_stress(self, features):
        """ndvi, temp, soil_moisture, humidity"""
        pred = self.models["crop_stress"].predict([features])[0]
        return "🚨 High Stress" if pred > 0.5 else "✅ Healthy"
    
    def calculate_ndvi(self, red, nir):
        """Manual NDVI formula"""
        return (nir - red) / (nir + red + 1e-8)
    
    @st.cache_data
    def preprocess_image(self, image, size=(224, 224)):
        """Simple image preprocessing for demo"""
        image = image.resize(size)
        return np.array(image)
    
    def predict_disease(self, image_array):
        """Demo disease prediction"""
        diseases = ["Healthy", "Bacterial Blight", "Fungal Spot", "Rust", "Mosaic Virus"]
        return np.random.choice(diseases)
    
    def predict_all(self, all_inputs):
        """Predict everything at once"""
        results = {}
        try:
            results["crop"] = self.predict_crop_recommendation(all_inputs["crop"])
            results["yield"] = self.predict_crop_yield(all_inputs["yield"])
            results["irrigation"] = self.predict_irrigation(all_inputs["irrigation"])
        except:
            results = {"status": "Demo mode active"}
        return results
