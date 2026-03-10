import joblib
import pandas as pd
import numpy as np
import os

# Caminho do modelo treinado
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'src', 'models', 'modelo_cardiaco.pkl')

COLUNAS_NUMERICAS = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]
COLUNAS_CATEGORICAS = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]


def _load_model():
    """Carrega o modelo treinado."""
    return joblib.load(MODEL_PATH)


def get_feature_importance() -> str:
    """
    Retorna as variáveis clínicas mais importantes para a predição de risco cardíaco,
    ordenadas por importância. Use esta ferramenta quando o usuário perguntar sobre
    quais fatores mais influenciam o risco cardíaco.
    """
    try:
        pipeline = _load_model()
        rf_model = pipeline.named_steps['modelo']
        preprocessor = pipeline.named_steps['preprocessamento']

        feature_names = COLUNAS_NUMERICAS + list(
            preprocessor.named_transformers_['cat']
            .get_feature_names_out(COLUNAS_CATEGORICAS)
        )

        importances = rf_model.feature_importances_
        feature_df = pd.DataFrame({
            'variavel': feature_names,
            'importancia': importances
        }).sort_values('importancia', ascending=False).head(8)

        result = "Top 8 variáveis mais importantes para o risco cardíaco:\n\n"
        for _, row in feature_df.iterrows():
            bar = "█" * int(row['importancia'] * 50)
            result += f"• {row['variavel']}: {row['importancia']:.3f} {bar}\n"

        return result
    except Exception as e:
        return f"Erro ao carregar importância das features: {str(e)}"


def predict_risk(
    age: int,
    sex: str,
    chest_pain_type: str,
    resting_bp: int,
    cholesterol: int,
    fasting_bs: int,
    resting_ecg: str,
    max_hr: int,
    exercise_angina: str,
    oldpeak: float,
    st_slope: str
) -> str:
    """
    Faz uma predição de risco cardíaco para um paciente com base em seus dados clínicos.
    Use esta ferramenta quando o usuário fornecer dados de um paciente e pedir uma avaliação.

    Parâmetros:
    - age: idade em anos (ex: 55)
    - sex: sexo biológico — 'M' para masculino, 'F' para feminino
    - chest_pain_type: tipo de dor no peito — 'ATA' (angina típica), 'NAP' (dor não-anginosa),
                       'ASY' (assintomático), 'TA' (angina atípica)
    - resting_bp: pressão arterial em repouso em mmHg (ex: 130)
    - cholesterol: colesterol sérico em mg/dL (ex: 250)
    - fasting_bs: glicemia em jejum > 120 mg/dL — 1 para sim, 0 para não
    - resting_ecg: ECG em repouso — 'Normal', 'ST', 'LVH'
    - max_hr: frequência cardíaca máxima atingida (ex: 150)
    - exercise_angina: angina induzida por exercício — 'Y' para sim, 'N' para não
    - oldpeak: depressão do segmento ST (ex: 1.5)
    - st_slope: inclinação do segmento ST — 'Up', 'Flat', 'Down'
    """
    try:
        pipeline = _load_model()

        patient_data = pd.DataFrame([{
            "Age": age,
            "Sex": sex,
            "ChestPainType": chest_pain_type,
            "RestingBP": resting_bp,
            "Cholesterol": cholesterol,
            "FastingBS": fasting_bs,
            "RestingECG": resting_ecg,
            "MaxHR": max_hr,
            "ExerciseAngina": exercise_angina,
            "Oldpeak": oldpeak,
            "ST_Slope": st_slope
        }])

        prediction = pipeline.predict(patient_data)[0]
        probability = pipeline.predict_proba(patient_data)[0]

        prob_high = probability[1] * 100
        prob_low = probability[0] * 100

        result = f"**Resultado da Predição:**\n\n"
        result += f"• Classificação: {'⚠️ ALTO RISCO' if prediction == 1 else '✅ BAIXO RISCO'}\n"
        result += f"• Probabilidade de doença cardíaca: {prob_high:.1f}%\n"
        result += f"• Probabilidade de saudável: {prob_low:.1f}%\n\n"
        result += f"Dados analisados: Paciente de {age} anos, sexo {'masculino' if sex == 'M' else 'feminino'}, "
        result += f"dor no peito tipo {chest_pain_type}, colesterol {cholesterol} mg/dL, "
        result += f"frequência cardíaca máxima {max_hr} bpm.\n\n"
        result += "⚠️ Este resultado é um apoio à triagem. Sempre consulte um médico para diagnóstico definitivo."

        return result
    except Exception as e:
        return f"Erro ao realizar predição: {str(e)}"


def get_model_metrics() -> str:
    """
    Retorna as métricas de desempenho do modelo de Machine Learning.
    Use esta ferramenta quando o usuário perguntar sobre a precisão, acurácia,
    confiabilidade ou qualidade do modelo.
    """
    metrics = """
Métricas do modelo RandomForestClassifier (otimizado via RandomizedSearchCV, validação cruzada 5-fold):

• Recall (Alto Risco):     91% — dos pacientes com doença cardíaca real, o modelo identifica 91%
• F1-Score (Alto Risco):   90% — equilíbrio entre precisão e recall
• Precision (Alto Risco):  89% — dos classificados como alto risco, 89% realmente têm a doença
• Acurácia geral:          88.6%
• Recall médio (CV):       ~82% — estimativa robusta com validação cruzada

Métrica principal de otimização: Recall
Motivo: Em triagem cardíaca, um falso negativo (doente classificado como saudável) 
tem consequências potencialmente fatais. Minimizar falsos negativos é prioridade.

Dataset: 918 pacientes, 11 variáveis clínicas (Kaggle — Heart Failure Prediction Dataset)
"""
    return metrics