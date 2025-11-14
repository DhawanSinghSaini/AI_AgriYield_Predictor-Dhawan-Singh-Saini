import streamlit as st
import joblib
import pandas as pd
import os

# --- Locate pipeline file automatically ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_FILENAME = os.path.join(BASE_DIR, "crop_yield_pipeline.pkl")

if not os.path.exists(PIPELINE_FILENAME):
    st.error(f"❌ Pipeline file not found at: {PIPELINE_FILENAME}")
    st.stop()

# Load pipeline (preprocessor + model)
pipeline = joblib.load(PIPELINE_FILENAME)

st.title("🌱 Crop Yield Prediction App")
st.write("Enter crop and environmental details to predict yield.")

# --- Dropdown values (sorted alphabetically) ---
crops = sorted([
    "Arecanut","Arhar/Tur","Bajra","Banana","Barley","Black pepper","Cardamom","Cashewnut",
    "Castor seed","Coconut","Coriander","Cotton(lint)","Cowpea(Lobia)","Dry chillies","Garlic",
    "Ginger","Gram","Groundnut","Guar seed","Horse-gram","Jowar","Jute","Khesari","Linseed",
    "Maize","Masoor","Mesta","Moong(Green Gram)","Moth","Niger seed","Oilseeds total",
    "Onion","Other Cereals","Other Kharif pulses","Other Rabi pulses","Other Summer Pulses",
    "other oilseeds","Peas & beans (Pulses)","Potato","Ragi","Rapeseed &Mustard","Rice",
    "Safflower","Sannhamp","Sesamum","Small millets","Soyabean","Sugarcane","Sunflower",
    "Sweet potato","Tapioca","Tobacco","Turmeric","Urad","Wheat"
])

seasons = sorted([
    "Autumn","Kharif","Rabi","Summer","Whole Year","Winter"
])

states = sorted([
    "Andhra Pradesh","Arunachal Pradesh","Assam","Bihar","Chhattisgarh","Delhi","Goa","Gujarat",
    "Haryana","Himachal Pradesh","Jammu and Kashmir","Jharkhand","Karnataka","Kerala",
    "Madhya Pradesh","Maharashtra","Manipur","Meghalaya","Mizoram","Nagaland","Odisha",
    "Puducherry","Punjab","Sikkim","State","Tamil Nadu","Telangana","Tripura","Uttar Pradesh",
    "Uttarakhand","West Bengal"
])

soils = sorted([
    "Alluvial","Black","Laterite","Loamy","Red"
])

# --- Numeric inputs ---
crop_year = st.number_input("Crop Year", min_value=1900, max_value=2100, value=2020)
area = st.number_input("Area (hectares)", min_value=0.0, value=1000.0)
production = st.number_input("Production (tons)", min_value=0.0, value=500.0)
rainfall = st.number_input("Annual Rainfall (mm)", min_value=0.0, value=800.0)
fertilizer = st.number_input("Fertilizer (kg/ha)", min_value=0.0, value=50.0)
pesticide = st.number_input("Pesticide (kg/ha)", min_value=0.0, value=10.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=70.0)
temperature = st.number_input("Average Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0)

# --- Dropdowns ---
crop = st.selectbox("Crop", crops)
season = st.selectbox("Season", seasons)
state = st.selectbox("State", states)
soil = st.selectbox("Soil Type", soils)

# --- Predict button ---
if st.button("Predict Yield"):
    raw_input = pd.DataFrame([{
        "Crop": crop,
        "Crop_Year": crop_year,
        "Season": season,
        "State": state,
        "Area": area,
        "Production": production,
        "Annual_Rainfall": rainfall,
        "Fertilizer": fertilizer,
        "Pesticide": pesticide,
        "HUMPIDITY": humidity,
        "SOIL TYPE": soil,
        "AVG_TEMPERATURE": temperature
    }])

    prediction = pipeline.predict(raw_input)[0]
    st.success(f"Predicted Yield: {prediction:.2f} kg/hectare")
