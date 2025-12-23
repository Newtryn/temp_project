import streamlit as st
import pandas as pd
import joblib

# Sayfa başlığı
st.title("Solar Power Prediction App")

st.write("This app predicts electricity generation of a solar power plant.")

# Modeli yükle
model = joblib.load("models/solar_model.pkl")

st.header("Weather Inputs")

# Kullanıcıdan input al
irradiance = st.slider("Solar Irradiance (W/m2)", 0, 1200, 500)
temperature = st.slider("Temperature (°C)", -10, 50, 25)
wind_speed = st.slider("Wind Speed (m/s)", 0, 20, 5)
humidity = st.slider("Humidity (%)", 0, 100, 50)
cloud_cover = st.slider("Cloud Cover (%)", 0, 100, 20)

# Inputları dataframe yap
input_df = pd.DataFrame({
    "irradiance": [irradiance],
    "temperature": [temperature],
    "wind_speed": [wind_speed],
    "humidity": [humidity],
    "cloud_cover": [cloud_cover]
})

# Tahmin butonu
if st.button("Predict Power Generation"):
    prediction = model.predict(input_df)
    st.success(f"Predicted Power Output: {prediction[0]:.2f} kW")
