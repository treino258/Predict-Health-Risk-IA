import os
import pandas as pd
import pytest

CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'src', 'DataSet', 'heart.csv')

COLUNAS_ESPERADAS = [
    "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol",
    "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina",
    "Oldpeak", "ST_Slope", "HeartDisease"
]


def test_arquivo_existe():
    """O arquivo heart.csv deve existir no caminho esperado."""
    assert os.path.exists(CSV_PATH), f"Arquivo não encontrado: {CSV_PATH}"


def test_dataframe_nao_vazio():
    """O dataset não pode estar vazio."""
    df = pd.read_csv(CSV_PATH)
    assert len(df) > 0, "O dataset está vazio."


def test_colunas_esperadas_presentes():
    """Todas as colunas necessárias para o modelo devem estar presentes."""
    df = pd.read_csv(CSV_PATH)
    for coluna in COLUNAS_ESPERADAS:
        assert coluna in df.columns, f"Coluna ausente: {coluna}"


def test_target_binario():
    """A coluna HeartDisease deve conter apenas 0 e 1."""
    df = pd.read_csv(CSV_PATH)
    valores_unicos = set(df['HeartDisease'].unique())
    assert valores_unicos == {0, 1}, f"Valores inesperados em HeartDisease: {valores_unicos}"


def test_tamanho_minimo():
    """O dataset deve ter pelo menos 500 registros para treino confiável."""
    df = pd.read_csv(CSV_PATH)
    assert len(df) >= 500, f"Dataset muito pequeno: {len(df)} registros."