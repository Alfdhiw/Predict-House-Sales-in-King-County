import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load model dan scaler
model = joblib.load("regresi_linear_model.pkl")

# Fitur input
st.title("Prediksi Harga Rumah")

st.write("Masukkan informasi rumah di bawah ini:")

sqft_living = st.number_input("Luas rumah (sqft_living)", min_value=100, max_value=10000, value=2000)
bedrooms = st.number_input("Jumlah kamar tidur", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("Jumlah kamar mandi", min_value=1.0, max_value=8.0, value=2.0)
floors = st.number_input("Jumlah lantai", min_value=1.0, max_value=4.0, value=1.0)
waterfront = st.selectbox("Menghadap ke air (waterfront)", [0, 1])
view = st.slider("Nilai pemandangan (view)", 0, 4, 0)
condition = st.slider("Kondisi rumah", 1, 5, 3)
grade = st.slider("Grade bangunan", 1, 13, 7)
sqft_above = st.number_input("Luas bangunan atas tanah (sqft_above)", min_value=100, max_value=10000, value=1800)
sqft_basement = st.number_input("Luas basement (sqft_basement)", min_value=0, max_value=5000, value=200)
yr_built = st.number_input("Tahun dibangun", min_value=1900, max_value=2025, value=2000)
yr_renovated = st.number_input("Tahun renovasi (0 jika belum pernah)", min_value=0, max_value=2025, value=0)
lat = st.number_input("Latitude", value=47.5)
long = st.number_input("Longitude", value=-122.2)
sqft_living15 = st.number_input("Luas rata-rata rumah tetangga (sqft_living15)", min_value=100, max_value=10000, value=2000)
sqft_lot = st.number_input("Luas tanah (sqft_lot)", min_value=500, max_value=100000, value=5000)
sqft_lot15 = st.number_input("Luas tanah tetangga (sqft_lot15)", min_value=500, max_value=100000, value=5000)
zipcode = st.number_input("Kode pos (zipcode)", min_value=100, max_value=100000, value=98178)

# Gabungkan input ke array
features = np.array([[bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view,
                      condition, grade, sqft_above, sqft_basement, yr_built, yr_renovated,
                      lat, long, sqft_living15, sqft_lot15, zipcode]])

# Scaling input seperti pada model
scaler = joblib.load("scaler.pkl")
# Anda sebaiknya menggunakan scaler yang sama dari training, namun jika tidak tersedia:
# Untuk keperluan demo ini, akan diskalakan ulang berdasar input (bisa tidak akurat)
st.write("Shape input fitur:", features.shape)
st.write("Scaler expects:", scaler.n_features_in_)
features_scaled = scaler.transform(features)

if st.button("Prediksi Harga"):
    predicted_price = model.predict(features_scaled)[0]
    st.success(f"Perkiraan harga rumah: ${predicted_price:,.2f}")
