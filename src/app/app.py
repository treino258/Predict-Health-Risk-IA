# app.py - Interface com Streamlit para previsão de risco cardíaco
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")

from llm.chat_tab import render_chat
import streamlit as st
import pandas as pd
import joblib
import os

# Carregar modelo treinado
BASE_DIR = Path(__file__).resolve().parent.parent  # src/app → src/
MODEL_PATH = BASE_DIR / ".." / "models" / "modelo_cardiaco.pkl"

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Arquivo não encontrado! O Python procurou em: {MODEL_PATH}")
    st.stop()

artefato = joblib.load(MODEL_PATH)

model = artefato["model"]
threshold = artefato["threshold"]

st.set_page_config(page_title="Previsão de Risco Cardíaco", layout="centered")

# ── TABS ──────────────────────────────────────────────────────────────────────
tab_previsao, tab_chat = st.tabs(["🫀 Previsão de Risco", "🤖 Assistente IA"])

# ── ABA 1: PREVISÃO ───────────────────────────────────────────────────────────
with tab_previsao:
    st.title("Previsão de Risco Cardíaco com IA")
    st.write("Preencha os dados clínicos do paciente abaixo para prever o risco de doença cardíaca.")

    age = st.number_input("Idade", min_value=1, max_value=120)
    sex = st.selectbox("Sexo", ["F", "M"])
    chest_pain = st.selectbox(
        "Tipo de Dor no Peito — ASY (assintomático), ATA (torácica típica), NAP (não-anginosa), TA (atípica)",
        ["ASY", "ATA", "NAP", "TA"]
    )
    restingbp = st.number_input("Pressão Arterial em Repouso", min_value=0)
    cholesterol = st.number_input("Colesterol", min_value=0)
    fastingbs = st.selectbox("Glicemia em Jejum > 120?", ["No", "Yes"])
    resting_ecg = st.selectbox("Resultado do ECG em Repouso", ["Normal", "ST", "LVH"])
    maxhr = st.number_input("Frequência Cardíaca Máxima", min_value=0)
    exercise_angina = st.selectbox("Angina Induzida por Exercício", ["N", "Y"])
    oldpeak = st.number_input("Depressão ST (Oldpeak)", min_value=0.0, step=0.1)
    st_slope = st.selectbox("Inclinação do Segmento ST", ["Down", "Flat", "Up"])

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

    if st.button("🔍 Analisar Risco"):
        proba = model.predict_proba(input_df)[:, 1][0]
        prediction = int(proba >= threshold)
        st.subheader("Resultado da Análise")

        st.write(f"**Risco estimado:** {proba:.2%}")


        # classificação mais rica
        if proba < 0.2:
            st.info("🟢 Risco baixo")
        elif proba < 0.5:
            st.warning("🟡 Risco moderado")
        else:
            st.error("🔴 Risco alto")
# ── ABA 2: CHAT ───────────────────────────────────────────────────────────────
with tab_chat:
    render_chat()