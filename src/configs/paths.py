from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

PATHS = {
    "data": BASE_DIR  / "DataSet" / "heart_cleaned.csv",
    "models": BASE_DIR  / "models",
    "images": BASE_DIR  / "img",
    "model_file": BASE_DIR  / "models" / "modelo_cardiaco.pkl",
    "config": BASE_DIR  / "configs" / "train.yaml",
}

