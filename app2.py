import streamlit as st
import pandas as pd
import joblib
import os

# -----------------------------
# Page config (must be first)
# -----------------------------
st.set_page_config(
    page_title="Solar Power Prediction",
    layout="centered"
)

# -----------------------------
# Load trained model (safe)
# -----------------------------
MODEL_PATH = "models/solar_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Make sure models/solar_model.pkl exists.")
    st.stop()

model = joblib.load(MODEL_PATH)

# -----------------------------
# Page title & description
# -----------------------------
st.title("☀️ Solar Power Prediction App")
st.write("This app predicts electricity generation of a solar power plant.")

# -----------------------------
# Input sections
# -----------------------------
st.header("Weather Conditions")

temperature = st.slider("Temperature at 2m (°C)", -10.0, 50.0, 25.0)
relative_humidity = st.slider("Relative Humidity at 2m (%)", 0.0, 100.0, 50.0)
pressure = st.slider("Mean Sea Level Pressure (hPa)", 950.0, 1050.0, 1013.0)
precipitation = st.slider("Total Precipitation (mm)", 0.0, 50.0, 0.0)
snowfall = st.slider("Snowfall Amount (mm)", 0.0, 50.0, 0.0)

st.header("Cloud Coverage")

total_cloud = st.slider("Total Cloud Cover (%)", 0.0, 100.0, 20.0)
high_cloud = st.slider("High Cloud Cover (%)", 0.0, 100.0, 10.0)
mid_cloud = st.slider("Medium Cloud Cover (%)", 0.0, 100.0, 15.0)
low_cloud = st.slider("Low Cloud Cover (%)", 0.0, 100.0, 20.0)

st.header("Radiation & Wind")

radiation = st.slider("Shortwave Radiation (W/m²)", 0.0, 1200.0, 600.0)
wind_speed_10 = st.slider("Wind Speed at 10m (m/s)", 0.0, 25.0, 5.0)
wind_dir_10 = st.slider("Wind Direction at 10m (°)", 0.0, 360.0, 180.0)

wind_speed_80 = st.slider("Wind Speed at 80m (m/s)", 0.0, 30.0, 7.0)
wind_dir_80 = st.slider("Wind Direction at 80m (°)", 0.0, 360.0, 180.0)

wind_speed_900 = st.slider("Wind Speed at 900mb (m/s)", 0.0, 40.0, 10.0)
wind_dir_900 = st.slider("Wind Direction at 900mb (°)", 0.0, 360.0, 180.0)

wind_gust = st.slider("Wind Gust at 10m (m/s)", 0.0, 40.0, 8.0)

st.header("Solar Geometry")

angle_of_incidence = st.slider("Angle of Incidence (°)", 0.0, 90.0, 30.0)
zenith = st.slider("Solar Zenith Angle (°)", 0.0, 90.0, 40.0)
azimuth = st.slider("Solar Azimuth Angle (°)", 0.0, 360.0, 180.0)

st.header("Lag Features")

power_lag_1 = st.slider("Power Lag 1 (kW)", 0.0, 1000.0, 300.0)
power_lag_2 = st.slider("Power Lag 2 (kW)", 0.0, 1000.0, 280.0)
power_lag_3 = st.slider("Power Lag 3 (kW)", 0.0, 1000.0, 260.0)

radiation_lag_1 = st.slider("Radiation Lag 1 (W/m²)", 0.0, 1200.0, 550.0)

# -----------------------------
# Build input dataframe
# -----------------------------
input_df = pd.DataFrame([{
    "temperature_2_m_above_gnd": temperature,
    "relative_humidity_2_m_above_gnd": relative_humidity,
    "mean_sea_level_pressure_MSL": pressure,
    "total_precipitation_sfc": precipitation,
    "snowfall_amount_sfc": snowfall,
    "total_cloud_cover_sfc": total_cloud,
    "high_cloud_cover_high_cld_lay": high_cloud,
    "medium_cloud_cover_mid_cld_lay": mid_cloud,
    "low_cloud_cover_low_cld_lay": low_cloud,
    "shortwave_radiation_backwards_sfc": radiation,
    "wind_speed_10_m_above_gnd": wind_speed_10,
    "wind_direction_10_m_above_gnd": wind_dir_10,
    "wind_speed_80_m_above_gnd": wind_speed_80,
    "wind_direction_80_m_above_gnd": wind_dir_80,
    "wind_speed_900_mb": wind_speed_900,
    "wind_direction_900_mb": wind_dir_900,
    "wind_gust_10_m_above_gnd": wind_gust,
    "angle_of_incidence": angle_of_incidence,
    "zenith": zenith,
    "azimuth": azimuth,
    "power_lag_1": power_lag_1,
    "power_lag_2": power_lag_2,
    "power_lag_3": power_lag_3,
    "radiation_lag_1": radiation_lag_1
}])

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Power Generation"):
    prediction = model.predict(input_df)
    st.success(f"Predicted Power Output: {prediction[0]:.2f} kW")
