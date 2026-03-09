import pytest
import pandas as pd
import numpy as np
import joblib
import os


MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'src', 'models', 'modelo_cardiaco.pkl')


@pytest.fixture(scope="module")
def modelo():
    """Carrega o modelo uma vez para todos os testes do módulo."""
    assert os.path.exists(MODEL_PATH), f"Modelo não encontrado: {MODEL_PATH}"
    return joblib.load(MODEL_PATH)


@pytest.fixture
def paciente_alto_risco():
    """Perfil clínico de alto risco cardíaco."""
    return pd.DataFrame([{
        "Age": 65,
        "Sex": "M",
        "ChestPainType": "ASY",       # Assintomático — maior risco
        "RestingBP": 160,
        "Cholesterol": 300,
        "FastingBS": 1,                # Glicemia em jejum > 120 mg/dl
        "RestingECG": "LVH",
        "MaxHR": 100,
        "ExerciseAngina": "Y",
        "Oldpeak": 2.5,
        "ST_Slope": "Flat",
    }])


@pytest.fixture
def paciente_baixo_risco():
    """Perfil clínico de baixo risco cardíaco."""
    return pd.DataFrame([{
        "Age": 35,
        "Sex": "F",
        "ChestPainType": "ATA",       # Angina atípica — menor risco
        "RestingBP": 120,
        "Cholesterol": 180,
        "FastingBS": 0,
        "RestingECG": "Normal",
        "MaxHR": 170,
        "ExerciseAngina": "N",
        "Oldpeak": 0.0,
        "ST_Slope": "Up",
    }])


def test_modelo_carrega(modelo):
    """O modelo deve carregar sem erros."""
    assert modelo is not None


def test_predicao_retorna_valor_binario(modelo, paciente_alto_risco):
    """A predição deve retornar 0 ou 1."""
    resultado = modelo.predict(paciente_alto_risco)
    assert resultado[0] in [0, 1], f"Predição inesperada: {resultado[0]}"


def test_predicao_alto_risco(modelo, paciente_alto_risco):
    """Perfil de alto risco deve ser classificado como doença cardíaca (1)."""
    resultado = modelo.predict(paciente_alto_risco)
    assert resultado[0] == 1, (
        "Paciente de alto risco foi classificado como baixo risco. "
        "Verifique o modelo ou o perfil de teste."
    )


def test_predicao_baixo_risco(modelo, paciente_baixo_risco):
    """Perfil de baixo risco deve ser classificado como saudável (0)."""
    resultado = modelo.predict(paciente_baixo_risco)
    assert resultado[0] == 0, (
        "Paciente de baixo risco foi classificado como alto risco. "
        "Verifique o modelo ou o perfil de teste."
    )


def test_probabilidade_alto_risco_maior_que_50(modelo, paciente_alto_risco):
    """A probabilidade de doença cardíaca deve ser > 50% para perfil de alto risco."""
    proba = modelo.predict_proba(paciente_alto_risco)
    prob_doenca = proba[0][1]
    assert prob_doenca > 0.5, f"Probabilidade de doença: {prob_doenca:.2f} — esperado > 0.50"


def test_modelo_aceita_multiplos_pacientes(modelo, paciente_alto_risco, paciente_baixo_risco):
    """O modelo deve processar um batch de pacientes sem erros."""
    batch = pd.concat([paciente_alto_risco, paciente_baixo_risco], ignore_index=True)
    resultados = modelo.predict(batch)
    assert len(resultados) == 2, f"Esperado 2 predições, obtido {len(resultados)}"