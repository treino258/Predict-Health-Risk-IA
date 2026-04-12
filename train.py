# train.py
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score
)

# ── 1. DADOS ──────────────────────────────────────────────────────────────────
csv_path = os.path.join(os.path.dirname(__file__), "src", "DataSet", "heart.csv")
df = pd.read_csv(csv_path)

X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

colunas_categoricas = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
colunas_numericas   = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]

# ── 2. SPLIT EM TRÊS PARTES — teste é sagrado, não é tocado até o final ───────
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
    # 0.176 de 85% ≈ 15% do total
)

print(f"Treino: {len(X_train)} | Validação: {len(X_val)} | Teste: {len(X_test)}")

# ── 3. PIPELINE ───────────────────────────────────────────────────────────────
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), colunas_numericas),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), colunas_categoricas)
])

pipeline = Pipeline(steps=[
    ("preprocessamento", preprocessor),
    ("modelo", RandomForestClassifier(random_state=42, class_weight="balanced"))
])

# ── 4. HYPERPARAMETER SEARCH (só no treino) ───────────────────────────────────
param_grid = {
    "modelo__n_estimators": [100, 200, 300],
    "modelo__max_depth": [None, 10, 20],
    "modelo__min_samples_split": [2, 5, 10]
}

search = RandomizedSearchCV(
    pipeline, param_grid, n_iter=10, cv=5,
    random_state=42, scoring="recall"
)
search.fit(X_train, y_train)
best_pipeline = search.best_estimator_
print("Melhores parâmetros:", search.best_params_)

# ── 5. CALIBRAÇÃO ─────────────────────────────────────────────────────────────
calibrated = CalibratedClassifierCV(best_pipeline, method="isotonic", cv=5)
calibrated.fit(X_train, y_train)

# ── 6. THRESHOLD — escolhido na VALIDAÇÃO, não no teste ──────────────────────
def find_best_threshold(model, X, y_true, target_sensitivity=0.95):
    y_proba = model.predict_proba(X)[:, 1]
    thresholds = np.linspace(0.05, 0.95, 181)
    candidates = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0

        if sensitivity >= target_sensitivity:
            candidates.append({"threshold": t, "specificity": specificity})

    if not candidates:
        print("Aviso: nenhum threshold atingiu o sensitivity alvo. Usando 0.5.")
        return 0.5

    return max(candidates, key=lambda d: d["specificity"])["threshold"]

best_t = find_best_threshold(calibrated, X_val, y_val)  # <-- X_val, não X_test
print(f"Threshold escolhido na validação: {best_t:.3f}")

# ── 7. AVALIAÇÃO FINAL — X_test tocado UMA ÚNICA VEZ, aqui ───────────────────
y_proba_test = calibrated.predict_proba(X_test)[:, 1]
y_pred_test  = (y_proba_test >= best_t).astype(int)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()
print("\n── Resultado Final (Test Set) ──")
print(f"  Recall (Sensitivity): {tp / (tp + fn):.4f}")
print(f"  Specificity:          {tn / (tn + fp):.4f}")
print(f"  Precision:            {precision_score(y_test, y_pred_test):.4f}")
print(f"  F1:                   {f1_score(y_test, y_pred_test):.4f}")
print(f"\n{classification_report(y_test, y_pred_test)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_test)}")

# ── 8. SALVA O MODELO ─────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
joblib.dump({"model": calibrated, "threshold": best_t}, "models/modelo_cardiaco.pkl")
print("\nModelo salvo em models/modelo_cardiaco.pkl")