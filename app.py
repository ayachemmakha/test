import streamlit as st
import numpy as np
import joblib

# Configuration de la page
st.set_page_config(page_title="Estimation immobilière", page_icon="🏡", layout="centered")

# Titre
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🏡 Estimation de la valeur d'une maison</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Prédisez le prix d'une maison en Californie à partir des caractéristiques du quartier.</p>", unsafe_allow_html=True)

# Charger modèle et scaler
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

# Séparateurs visuels
st.markdown("---")

# Interface utilisateur avec colonnes
col1, col2 = st.columns(2)

with col1:
    MedInc = st.slider("💵 Revenu médian (x10k $)", 0.5, 15.0, 3.0)
    HouseAge = st.slider("🏠 Âge médian des maisons", 1.0, 52.0, 20.0)
    AveRooms = st.slider("🛋️ Nombre moyen de pièces", 1.0, 15.0, 5.0)
    AveBedrms = st.slider("🛏️ Nombre moyen de chambres", 0.5, 5.0, 1.0)

with col2:
    Population = st.slider("👥 Population", 3.0, 35682.0, 1000.0)
    AveOccup = st.slider("👨‍👩‍👧‍👦 Occupants par maison", 0.5, 20.0, 3.0)
    Latitude = st.slider("🌎 Latitude", 32.5, 42.0, 35.0)
    Longitude = st.slider("🌍 Longitude", -124.5, -114.0, -120.0)

# Bouton de prédiction
if st.button("🔍 Estimer la valeur"):
    input_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    st.markdown("---")
    st.markdown(f"""
    <div style="background-color: #e8f5e9; padding: 20px; border-radius: 10px; border: 1px solid #4CAF50;">
        <h3 style="color: #2e7d32;">💰 Valeur estimée :</h3>
        <p style="font-size: 24px;"><strong>{prediction * 100_000:,.2f} $</strong></p>
    </div>
    """, unsafe_allow_html=True)
