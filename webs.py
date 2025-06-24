import streamlit as st
import numpy as np
import joblib

# Load model dan scaler
model = joblib.load("best_random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Prediksi Harga Rumah di King County")

st.write("Masukkan 4 fitur berikut:")

# Input dari user
sqft_living = st.number_input("Luas Ruang Tinggal (sqft)", min_value=100, max_value=10000, value=1500)
grade = st.slider("Kualitas Bangunan (Grade)", 1, 13, 7)
lat = st.number_input("Latitude", value=47.5)
long = st.number_input("Longitude", value=-122.2)

# Prediksi
if st.button("Prediksi Harga"):
    input_data = np.array([[sqft_living, lat, grade, long]])
    input_scaled = scaler.transform(input_data)
    pred_price = model.predict(input_scaled)[0]

    st.success(f"Perkiraan Harga Rumah: **Rp {pred_price:,.0f}**")
