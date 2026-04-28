

# ── Imports ───────────────────────────────────────────────────────────────────

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ── Constantes ────────────────────────────────────────────────────────────────
# Centralizar aqui evita erros de digitação espalhados pelo código

TARGET = "HeartDisease"

NUMERICAL_COLS = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]
CATEGORICAL_COLS = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]

# Nomes dos steps do pipeline — mudar aqui muda em todo lugar
STEP_PREPROCESSOR = "preprocessor"
STEP_MODEL = "model"

TRANSFORMER_NUM = "num"
TRANSFORMER_CAT = "cat"

PATHS = {
    "data": Path("DataSet/heart_cleaned.csv"),
    "models": Path("models"),
    "images": Path("img"),
    "model_file": Path("models/modelo_cardiaco.pkl"),
}


# ── Configuração ──────────────────────────────────────────────────────────────

@dataclass
class Config:
    test_size: float = 0.2
    val_size: float = 0.2
    n_iter_search: int = 20
    cv_folds: int = 5
    random_state: int = 42
    target_sensitivity: float = 0.95


# ── Carregamento e validação dos dados ───────────────────────────────────────

def load_data(path: Path) -> pd.DataFrame:
    """Carrega o CSV e valida que as colunas esperadas existem."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {path}")

    df = pd.read_csv(path)

    expected = set(NUMERICAL_COLS + CATEGORICAL_COLS + [TARGET])
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no dataset: {missing}")

    return df


def check_coverage(df: pd.DataFrame) -> None:
    """Verifica que todas as colunas de X estão cobertas pelo pipeline."""
    all_cols = set(df.drop(columns=TARGET).columns)
    covered = set(NUMERICAL_COLS + CATEGORICAL_COLS)
    ignored = all_cols - covered
    if ignored:
        print(f"⚠️  Colunas ignoradas pelo pipeline: {ignored}")
    else:
        print("✅ Todas as colunas estão cobertas pelo pipeline.")


# ── Split ─────────────────────────────────────────────────────────────────────

def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    config: Config,
) -> tuple:
    """
    Divide em treino, validação e teste de forma estratificada.

    Retorna: X_train, X_val, X_test, y_train, y_val, y_test
    """
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y,
        test_size=config.test_size,
        stratify=y,
        random_state=config.random_state,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=config.val_size,
        stratify=y_train_full,
        random_state=config.random_state,
    )

    total = len(X_train) + len(X_val) + len(X_test)
    print(f"Split → treino: {len(X_train)} ({len(X_train)/total:.0%})"
          f" | val: {len(X_val)} ({len(X_val)/total:.0%})"
          f" | teste: {len(X_test)} ({len(X_test)/total:.0%})")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ── Pipeline ──────────────────────────────────────────────────────────────────

def build_pipeline() -> Pipeline:
    """
    Constrói o pipeline com preprocessamento numérico e categórico.

    Usando constantes NUMERICAL_COLS e CATEGORICAL_COLS para garantir
    que o pipeline cobre exatamente o que foi definido no topo do arquivo.
    """
    preprocessor = ColumnTransformer([
        (TRANSFORMER_NUM, StandardScaler(), NUMERICAL_COLS),
        (TRANSFORMER_CAT, OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
    ])

    model = RandomForestClassifier(
        random_state=42,
        class_weight="balanced",
    )

    return Pipeline([
        (STEP_PREPROCESSOR, preprocessor),
        (STEP_MODEL, model),
    ])


def tune_model(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, config: Config) -> Pipeline:
    """
    Busca os melhores hiperparâmetros via RandomizedSearchCV.

    Retorna o melhor estimador já treinado.
    """
    param_dist = {
        f"{STEP_MODEL}__n_estimators": [100, 200, 300],
        f"{STEP_MODEL}__max_depth": [5, 10, 20],       # removido None — evita overfitting em datasets pequenos
        f"{STEP_MODEL}__min_samples_split": [2, 5, 10],
        f"{STEP_MODEL}__min_samples_leaf": [1, 2, 4],  # parâmetro extra de regularização
    }

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=config.n_iter_search,
        scoring="average_precision",
        cv=config.cv_folds,
        n_jobs=-1,              # usa todos os cores disponíveis
        random_state=config.random_state,
        verbose=1,
    )

    search.fit(X_train, y_train)

    best = search.best_estimator_
    print(f"Melhores params: { {k: v for k, v in search.best_params_.items()} }")
    print(f"CV PR AUC (treino): {search.best_score_:.4f}")

    return best


# ── Calibração e seleção do modelo ───────────────────────────────────────────

def calibrate(pipeline: Pipeline, X_val: pd.DataFrame, y_val: pd.Series) -> CalibratedClassifierCV:
    """Calibra as probabilidades do pipeline usando o conjunto de validação."""
    calibrated = CalibratedClassifierCV(pipeline, method="sigmoid", cv=None)
    calibrated.fit(X_val, y_val)
    return calibrated


def choose_best_model(
    original: Pipeline,
    calibrated: CalibratedClassifierCV,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> object:
    """
    Compara original vs calibrado e retorna o melhor por PR AUC.

    Avaliado em validação — nunca em teste.
    """
    def metrics(model):
        proba = model.predict_proba(X_val)[:, 1]
        return {
            "pr_auc": average_precision_score(y_val, proba),
            "roc_auc": roc_auc_score(y_val, proba),
            "brier": brier_score_loss(y_val, proba),
        }

    m_orig = metrics(original)
    m_cal = metrics(calibrated)

    print("\n=== Comparação Original vs Calibrado ===")
    print(f"{'Modelo':<12} {'PR AUC':>8} {'ROC AUC':>8} {'Brier':>8}")
    print(f"{'Original':<12} {m_orig['pr_auc']:>8.4f} {m_orig['roc_auc']:>8.4f} {m_orig['brier']:>8.4f}")
    print(f"{'Calibrado':<12} {m_cal['pr_auc']:>8.4f} {m_cal['roc_auc']:>8.4f} {m_cal['brier']:>8.4f}")

    if m_cal["pr_auc"] >= m_orig["pr_auc"]:
        print("\n✔️  Escolha: CALIBRADO")
        return calibrated
    else:
        print("\n✔️  Escolha: ORIGINAL")
        return original


# ── Extração do pipeline interno ─────────────────────────────────────────────

def extract_pipeline(model: object) -> Pipeline:
    """
    Extrai o pipeline sklearn independente de ser calibrado ou não.

    Centralizado aqui — nunca repete esse if/else no notebook.
    """
    if isinstance(model, CalibratedClassifierCV):
        return model.calibrated_classifiers_[0].estimator
    if isinstance(model, Pipeline):
        return model
    raise TypeError(f"Tipo de modelo não reconhecido: {type(model)}")


def get_feature_names(pipeline: Pipeline) -> list[str]:
    """Retorna os nomes das features após o preprocessamento."""
    cat_encoder = (
        pipeline
        .named_steps[STEP_PREPROCESSOR]
        .named_transformers_[TRANSFORMER_CAT]
    )
    return NUMERICAL_COLS + list(cat_encoder.get_feature_names_out(CATEGORICAL_COLS))


# ── Threshold ─────────────────────────────────────────────────────────────────

def find_best_threshold(
    model: object,
    X: pd.DataFrame,
    y_true: pd.Series,
    target_sensitivity: float = 0.95,
) -> tuple[float, np.ndarray, list, list]:
    """
    Encontra o threshold que maximiza especificidade com sensibilidade >= target.

    Retorna: best_threshold, thresholds, recalls, specificities
    """
    y_proba = model.predict_proba(X)[:, 1]
    thresholds = np.linspace(0.05, 0.95, 181)
    recalls, specificities, candidates = [], [], []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        recalls.append(sensitivity)
        specificities.append(specificity)
        if sensitivity >= target_sensitivity:
            candidates.append({"threshold": t, "specificity": specificity})

    if not candidates:
        raise ValueError(
            f"Nenhum threshold atingiu sensibilidade >= {target_sensitivity}. "
            "Considere reduzir target_sensitivity."
        )

    best_t = max(candidates, key=lambda d: d["specificity"])["threshold"]
    print(f"Threshold escolhido: {best_t:.2f} (sensibilidade alvo: {target_sensitivity})")

    return best_t, thresholds, recalls, specificities


# ── Avaliação final ───────────────────────────────────────────────────────────

def evaluate_on_test(
    model: object,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
) -> dict:
    """
    Avalia o modelo no conjunto de teste.

    IMPORTANTE: só chamar esta função uma vez, no final.
    Chamar múltiplas vezes durante desenvolvimento vaza informação do teste.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    results = {
        "recall":      tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "precision":   precision_score(y_test, y_pred),
        "f1":          f1_score(y_test, y_pred),
        "pr_auc":      average_precision_score(y_test, y_proba),
        "roc_auc":     roc_auc_score(y_test, y_proba),
        "brier":       brier_score_loss(y_test, y_proba),
    }

    print("\n── Resultado Final (Test Set) ──────────────────")
    for k, v in results.items():
        print(f"  {k:<14}: {v:.4f}")
    print(f"\n{classification_report(y_test, y_pred)}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    return results


# ── Visualizações ─────────────────────────────────────────────────────────────

def plot_threshold_curve(
    thresholds: np.ndarray,
    recalls: list,
    specificities: list,
    best_t: float,
    save_dir: Path,
) -> None:
    idx = int(np.abs(thresholds - best_t).argmin())

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(thresholds, recalls, label="Recall (Sensitivity)")
    ax.plot(thresholds, specificities, label="Specificity")
    ax.scatter([thresholds[idx]], [recalls[idx]], zorder=5)
    ax.scatter([thresholds[idx]], [specificities[idx]], zorder=5)
    ax.axvline(best_t, linestyle="--", alpha=0.4, label=f"t={best_t:.2f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Recall vs Specificity vs Threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_dir / "threshold_curve.png", dpi=150)
    plt.close(fig)
    print(f"Salvo: {save_dir / 'threshold_curve.png'}")


def plot_feature_importance(
    model: object,
    feature_names: list[str],
    save_dir: Path,
) -> None:
    pipeline = extract_pipeline(model)
    rf = pipeline.named_steps[STEP_MODEL]

    if not hasattr(rf, "feature_importances_"):
        print("Modelo não suporta feature_importances_ — pulando.")
        return

    pd.Series(rf.feature_importances_, index=feature_names)\
        .sort_values()\
        .plot(kind="barh", figsize=(10, 6))

    plt.title("Importância das Features (MDI)")
    plt.tight_layout()
    plt.savefig(save_dir / "feature_importance.png", dpi=150)
    plt.close()
    print(f"Salvo: {save_dir / 'feature_importance.png'}")


def plot_shap(
    model: object,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    feature_names: list[str],
    save_dir: Path,
    sample_idx: int = 0,
) -> None:
    pipeline = extract_pipeline(model)
    rf = pipeline.named_steps[STEP_MODEL]
    preprocessor = pipeline.named_steps[STEP_PREPROCESSOR]

    X_train_t = preprocessor.transform(X_train)
    X_val_t   = preprocessor.transform(X_val)

    explainer   = shap.Explainer(rf, X_train_t, feature_names=feature_names)
    shap_values = explainer(X_val_t, check_additivity=False)

    # Global — importância média
    plt.figure()
    shap.plots.bar(shap_values[:, :, 1], show=False)
    plt.title("SHAP — Importância Global")
    plt.tight_layout()
    plt.savefig(save_dir / "shap_global.png", dpi=150)
    plt.close()

    # Summary — distribuição + direção
    plt.figure()
    shap.summary_plot(shap_values[:, :, 1], X_val_t, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(save_dir / "shap_summary.png", dpi=150)
    plt.close()

    # Individual
    plt.figure()
    shap.plots.waterfall(shap_values[sample_idx, :, 1], show=False)
    plt.tight_layout()
    plt.savefig(save_dir / "shap_individual.png", dpi=150)
    plt.close()

    print(f"SHAP salvo em: {save_dir}")


# ── Persistência ──────────────────────────────────────────────────────────────

def save_model(model: object, threshold: float, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "threshold": threshold}, path)
    print(f"Modelo salvo em: {path}")


def load_model(path: Path) -> tuple:
    if not path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {path}")
    artifact = joblib.load(path)
    return artifact["model"], artifact["threshold"]


# ── Execução principal ────────────────────────────────────────────────────────

def main():
    config = Config()

    # Diretórios de saída
    PATHS["models"].mkdir(parents=True, exist_ok=True)
    PATHS["images"].mkdir(parents=True, exist_ok=True)

    # 1. Dados
    print("\n[1/7] Carregando dados...")
    df = load_data(PATHS["data"])
    check_coverage(df)
    print(f"Shape: {df.shape}")
    print("Balanceamento:\n", df[TARGET].value_counts())

    X = df.drop(columns=TARGET)
    y = df[TARGET]

    # 2. Split
    print("\n[2/7] Dividindo dados...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, config)

    # 3. Baseline
    print("\n[3/7] Baseline...")
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    baseline = average_precision_score(y_val, dummy.predict_proba(X_val)[:, 1])
    print(f"Baseline PR AUC: {baseline:.4f}")

    # 4. Tuning
    print("\n[4/7] Tuning do modelo...")
    best_pipeline = tune_model(build_pipeline(), X_train, y_train, config)

    # 5. Calibração e seleção
    print("\n[5/7] Calibração...")
    calibrated = calibrate(best_pipeline, X_val, y_val)
    best_model = choose_best_model(best_pipeline, calibrated, X_val, y_val)

    # 6. Threshold
    print("\n[6/7] Buscando melhor threshold...")
    best_t, thresholds, recalls, specificities = find_best_threshold(
        best_model, X_val, y_val, config.target_sensitivity
    )
    plot_threshold_curve(thresholds, recalls, specificities, best_t, PATHS["images"])

    # 7. Avaliação final (só aqui tocamos no teste)
    print("\n[7/7] Avaliação final no teste...")
    evaluate_on_test(best_model, X_test, y_test, best_t)

    # Visualizações
    pipeline     = extract_pipeline(best_model)
    feature_names = get_feature_names(pipeline)

    plot_feature_importance(best_model, feature_names, PATHS["images"])
    plot_shap(best_model, X_train, X_val, feature_names, PATHS["images"])

    # Salvar
    save_model(best_model, best_t, PATHS["model_file"])


if __name__ == "__main__":
    main()
