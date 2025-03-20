import streamlit as st
import joblib
import numpy as np
import pandas as pd
from PIL import Image 
from sklearn.preprocessing import StandardScaler

# Cargar modelo y escalador
scaler = joblib.load("scaler.bin")
model = joblib.load("knn_model.joblib")

# Cargar imagen
image = Image.open("heart.jpg")

# Configurar la página
st.title("Predicción de Enfermedad del Corazón")
st.subheader("Realizado por Cesar Solano")
st.image(image, caption="Cuidado del corazón",  use_container_width =True)

# Instrucciones
st.write("""
### Instrucciones de Uso
Ingrese su edad y nivel de colesterol para predecir si tiene riesgo de sufrir una enfermedad del corazón.
Los valores deben estar dentro del rango permitido.
""")

# Entrada del usuario
edad = st.slider("Edad", 20, 77, 40)
colesterol = st.slider("Colesterol", 100, 600, 200)

# Crear DataFrame
input_data = pd.DataFrame([[edad, colesterol]], columns=["edad", "colesterol"])

# Preprocesamiento
input_scaled = scaler.transform(input_data)

# Predicción
if st.button("Predecir"):
    prediccion = model.predict(input_scaled)[0]
    if prediccion == 0:
        st.success("😊 No tiene probabilidades de sufrir del corazón.")
    else:
        st.error("😢 Cuídese, podría estar en riesgo.")

# Pie de página
st.markdown("---")
st.markdown("**Unab 2025®**")