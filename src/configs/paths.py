from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

PATHS = {
    "data": BASE_DIR  / "DataSet" / "heart_cleaned.csv",
    "models": BASE_DIR  / "models",
    "images": BASE_DIR  / "img",
    "model_file": BASE_DIR  / "models" / "modelo_cardiaco.pkl",
    "config": BASE_DIR  / "configs" / "train.yaml",

}

DATA = BASE_DIR  / "DataSet" / "heart_cleaned.csv",
MODELS = BASE_DIR  / "models",
IMGS = BASE_DIR  / "img",
MODEL_PATH = BASE_DIR  / "models" / "modelo_cardiaco.pkl",
CONFIG = BASE_DIR  / "configs" / "train.yaml",