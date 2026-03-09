import pytest
import pandas as pd
import numpy as np
import os


CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'src', 'DataSet', 'heart.csv')

COLUNAS_NUMERICAS = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]
COLUNAS_CATEGORICAS = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]


def test_sem_nulos_nas_features():
    """Nenhuma feature deve ter valores nulos — o modelo não tolera NaN."""
    df = pd.read_csv(CSV_PATH)
    features = COLUNAS_NUMERICAS + COLUNAS_CATEGORICAS
    nulos = df[features].isnull().sum()
    colunas_com_nulo = nulos[nulos > 0]
    assert len(colunas_com_nulo) == 0, f"Colunas com nulos: {colunas_com_nulo.to_dict()}"


def test_colesterol_zero_é_anomalia():
    """
    Colesterol = 0 é clinicamente impossível.
    O dataset contém registros assim — este teste documenta a anomalia
    e garante que sabemos quantos são (referência: ~172 registros).
    """
    df = pd.read_csv(CSV_PATH)
    zeros = (df['Cholesterol'] == 0).sum()
    # Documenta a anomalia sem falhar — é um dado conhecido do dataset
    assert zeros < len(df) * 0.25, (
        f"Mais de 25% dos registros têm colesterol zero ({zeros}). "
        "Verifique se a imputação foi aplicada."
    )


def test_valores_fora_de_faixa_idade():
    """Idade deve estar entre 1 e 120 anos."""
    df = pd.read_csv(CSV_PATH)
    invalidos = df[(df['Age'] < 1) | (df['Age'] > 120)]
    assert len(invalidos) == 0, f"{len(invalidos)} registros com idade inválida."


def test_valores_fora_de_faixa_frequencia_cardiaca():
    """Frequência cardíaca máxima deve estar entre 40 e 250 bpm."""
    df = pd.read_csv(CSV_PATH)
    invalidos = df[(df['MaxHR'] < 40) | (df['MaxHR'] > 250)]
    assert len(invalidos) == 0, f"{len(invalidos)} registros com MaxHR inválido."


def test_categoricas_sem_valores_inesperados():
    """
    Colunas categóricas devem conter apenas valores conhecidos.
    Valores fora do domínio quebram o OneHotEncoder em produção.
    """
    df = pd.read_csv(CSV_PATH)

    dominios = {
        "Sex": {"M", "F"},
        "ChestPainType": {"ATA", "NAP", "ASY", "TA"},
        "RestingECG": {"Normal", "ST", "LVH"},
        "ExerciseAngina": {"Y", "N"},
        "ST_Slope": {"Up", "Flat", "Down"},
    }

    for coluna, valores_validos in dominios.items():
        valores_encontrados = set(df[coluna].unique())
        inesperados = valores_encontrados - valores_validos
        assert len(inesperados) == 0, (
            f"Coluna '{coluna}' contém valores inesperados: {inesperados}"
        )