from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SRC_DIR = BASE_DIR / "src"

PATHS = {
    "data": SRC_DIR  / "DataSet" / "heart_cleaned.csv",
    "models": SRC_DIR / "models",
    "images": BASE_DIR  / "img",
    "model_file": "models" / "modelo_cardiaco.pkl",
    "config": SRC_DIR  / "configs" / "train.yaml",
}
