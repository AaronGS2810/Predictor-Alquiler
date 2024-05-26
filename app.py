
import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA

# Cargar los modelos y los objetos de preprocesamiento ajustados
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')
poly = joblib.load('poly.pkl')
rf_model = joblib.load('rf_model.pkl')
lr_model = joblib.load('lr_model.pkl')

# Definir las variables
variables = [
    'host_total_listings_count',
    'NumeroBanhos',
    'bedrooms',
    'accommodates',
    'beds',
    'calculated_host_listings_count',
    'availability_60',
    'neighbourhood_cleansed'
]

# Título de la aplicación
st.title('Predictor de Precio de Alquiler')

# Crear entradas para cada variable
inputs = {}
for var in variables:
    inputs[var] = st.number_input(f'Ingrese el valor para {var}', step=1.0)

# Botón para realizar la predicción
if st.button('Predecir Precio'):
    try:
        # Obtener los valores ingresados
        values = [inputs[var] for var in variables]

        # Estandarizar los valores ingresados
        values_scaled = scaler.transform([values])

        # Aplicar PCA
        values_pca = pca.transform(values_scaled)

        # Crear características polinómicas
        values_poly = poly.transform(values_pca)

        # Realizar la predicción
        pred_rf = rf_model.predict(values_pca)
        pred_quad = lr_model.predict(values_poly)
        prediction = (pred_rf + pred_quad) / 2

        # Invertir la transformación logarítmica
        price_prediction = np.exp(prediction)[0]

        # Mostrar el resultado
        st.success(f'El precio estimado de alquiler es: {price_prediction:.2f} USD')
    
    except Exception as e:
        st.error(f'Error: {str(e)}')
