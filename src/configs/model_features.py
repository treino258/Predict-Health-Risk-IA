from sklearn.pipeline import Pipeline

# ── Constantes ────────────────────────────────────────────────────────────────
# Centralizar aqui evita erros de digitação espalhados pelo código

TARGET = "HeartDisease"

NUMERICAL_COLS = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]
CATEGORICAL_COLS = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]


TRANSFORMER_NUM = "num"
TRANSFORMER_CAT = "cat"

STEP_MODEL = "model"

STEP_PREPROCESSOR = "preprocessor"

def get_feature_names(pipeline: Pipeline) -> list[str]:
    """Retorna os nomes das features após o preprocessamento."""
    cat_encoder = (
        pipeline
        .named_steps[STEP_PREPROCESSOR]
        .named_transformers_[TRANSFORMER_CAT]
    )
    return NUMERICAL_COLS + list(cat_encoder.get_feature_names_out(CATEGORICAL_COLS))