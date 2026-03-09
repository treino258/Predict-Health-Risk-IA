# app.py - Interface com Streamlit para previsão de risco cardíaco
from pathlib import Path
import streamlit as st
import pandas as pd
import joblib
import os

# Carregar modelo treinado
BASE_DIR = Path(__file__).resolve().parent.parent  # src/app → src/
MODEL_PATH = BASE_DIR / "models" / "modelo_cardiaco.pkl"
print(f"Tentando carregar o modelo em: {MODEL_PATH}")

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Arquivo não encontrado! O Python procurou em: {MODEL_PATH}")
    st.info("Verifique se a pasta 'src' está fora da pasta 'Documents'.")
    st.stop()

pipeline = joblib.load(MODEL_PATH)

st.set_page_config(page_title="Previsão de Risco Cardíaco", layout="centered")
st.title("Previsão de Risco Cardíaco com IA")
st.write("Preencha os dados clínicos do paciente abaixo para prever o risco de doença cardíaca.")

# Campos de entrada
age = st.number_input("Idade", min_value=1, max_value=120)
sex = st.selectbox("Sexo", ["F", "M"])
chest_pain = st.selectbox("Tipo de Dor no , ASY (assintomático), ATA(dor torácica típica), NAP (Dor que nao é angina), TA (dor torácica atípica)", ["ASY", "ATA", "NAP", "TA"])
restingbp = st.number_input("Pressão Arterial em Repouso", min_value=0)
cholesterol = st.number_input("Colesterol", min_value=0)
fastingbs = st.selectbox("Glicemia em Jejum > 120?", ["No", "Yes"])
resting_ecg = st.selectbox("Resultado do ECG (eletrocardiograma) em Repouso", ["Normal", "ST", "LVH"])
maxhr = st.number_input("Frequência Cardíaca Máxima", min_value=0)
exercise_angina = st.selectbox("Angina Induzida por Exercício", ["N", "Y"])
oldpeak = st.number_input("Depressão ST (Oldpeak)", min_value=0.0, step=0.1)
st_slope = st.selectbox("Inclinação do Segmento ST", ["Down", "Flat", "Up"])

# Pré-processamento manual dos dados (codificação igual ao get_dummies com drop_first=True)
input_df = pd.DataFrame([{
    "Age": age,
    "Sex": sex,
    "ChestPainType": chest_pain,
    "RestingBP": restingbp,
    "Cholesterol": cholesterol,
    "FastingBS": 1 if fastingbs == "Yes" else 0,
    "RestingECG": resting_ecg,
    "MaxHR": maxhr,
    "ExerciseAngina": exercise_angina,
    "Oldpeak": oldpeak,
    "ST_Slope": st_slope
}])

# Converter para DataFrame com as colunas que o modelo espera

# Botão de previsão
if st.button("🔍 Analisar Risco"):
    prediction = pipeline.predict(input_df)[0]
    st.subheader("Resultado da Análise")
    if prediction == 1:
        st.error("🔴 Alto risco de doença cardíaca!")
    else:
        st.success("🟢 Baixo risco de doença cardíaca.")
