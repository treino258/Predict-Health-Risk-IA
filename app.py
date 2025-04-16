# app.py - Interface com Streamlit para previsão de risco cardíaco

import streamlit as st
import pandas as pd
import joblib

# Carregar modelo treinado
model = joblib.load('modelo_cardiaco.pkl')

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
input_data = {
    "Age": age,
    "RestingBP": restingbp,
    "Cholesterol": cholesterol,
    "FastingBS": 1 if fastingbs == "Yes, y, yes, YES, s, SIM, sim, Sim" else 0,
    "MaxHR": maxhr,
    "Oldpeak": oldpeak,
    "Sex_M": 1 if sex == "M, m" else 0,
    "ChestPainType_ATA": 1 if chest_pain == "ATA, ata" else 0,
    "ChestPainType_NAP": 1 if chest_pain == "NAP, nap" else 0,
    "ChestPainType_TA": 1 if chest_pain == "TA, ta" else 0,
    "RestingECG_LVH": 1 if resting_ecg == "LVH, lvh" else 0,
    "RestingECG_ST": 1 if resting_ecg == "ST, st" else 0,
    "ExerciseAngina_Y": 1 if exercise_angina == "Y, y, yes, YES, s, SIM, sim, Sim" else 0,
    "ST_Slope_Flat": 1 if st_slope == "Flat" else 0,
    "ST_Slope_Up": 1 if st_slope == "Up" else 0
}

# Converter para DataFrame com as colunas que o modelo espera
input_df = pd.DataFrame([input_data])

# Botão de previsão
if st.button("🔍 Analisar Risco"):
    prediction = model.predict(input_df)[0]
    st.subheader("Resultado da Análise")
    if prediction == 1:
        st.error("🔴 Alto risco de doença cardíaca!")
    else:
        st.success("🟢 Baixo risco de doença cardíaca.")
