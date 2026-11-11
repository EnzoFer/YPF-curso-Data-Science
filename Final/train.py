"""Entrena y evalúa el pipeline de clasificación de daño observado."""
import json
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score, confusion_matrix, classification_report)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

from model import (FEATURES, MODEL_PATH, NUMERIC, NOMINAL, ORDINAL,
                   engineer_features, make_pipeline)

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "nyc_tree_census_sample.csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tuning", choices=["grid", "optuna"], default="grid",
                        help="GridSearchCV (default) u Optuna TPE (búsqueda compacta)")
    parser.add_argument("--trials", type=int, default=6, help="Pruebas Optuna, entre 1 y 20")
    args = parser.parse_args()
    if not DATA.exists():
        raise FileNotFoundError("Falta el dataset. Ejecutá `python download_dataset.py` primero.")
    raw = pd.read_csv(DATA, low_memory=False)
    prepared = engineer_features(raw)
    prepared = prepared[prepared["tree_dbh"].between(1, 60) & prepared["target"].notna()].copy()
    if prepared["target"].nunique() != 2:
        raise ValueError("Se necesitan las dos clases sidewalk=Damage y NoDamage.")

    # Top de especies se aprende sólo con la partición de entrenamiento.
    X_raw = prepared[NUMERIC + ORDINAL + NOMINAL].copy()
    y = prepared["target"].astype(int)
    # La muestra aleatoria estratificada reproduce el alcance del notebook (hasta 100k).
    if len(X_raw) > 100_000:
        X_raw, _, y, _ = train_test_split(
            X_raw, y, train_size=100_000, random_state=42, stratify=y
        )
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )
    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Búsqueda pequeña y reproducible, alineada con el GridSearch del notebook.
    if args.tuning == "optuna":
        if not 1 <= args.trials <= 20:
            parser.error("--trials debe estar entre 1 y 20")
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        optuna_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        def objective(trial):
            params = {
                "clf__n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
                "clf__max_depth": trial.suggest_int("max_depth", 8, 20),
                "clf__min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 15),
            }
            return float(cross_val_score(make_pipeline(**{
                "n_estimators": params["clf__n_estimators"],
                "max_depth": params["clf__max_depth"],
                "min_samples_leaf": params["clf__min_samples_leaf"],
            }), X_train, y_train, cv=optuna_cv, scoring="f1", n_jobs=1).mean())

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=args.trials)
        best_params = {f"clf__{key}": value for key, value in study.best_params.items()}
        best = make_pipeline(**study.best_params).fit(X_train, y_train)
        cv_f1 = float(study.best_value)
    else:
        from sklearn.model_selection import GridSearchCV
        search = GridSearchCV(
            make_pipeline(),
            {"clf__n_estimators": [100, 200], "clf__max_depth": [8, 12, 16],
             "clf__min_samples_leaf": [1, 10]},
            cv=cv, scoring="f1", refit=True, n_jobs=2, return_train_score=True,
        )
        search.fit(X_train, y_train)
        best, best_params, cv_f1 = search.best_estimator_, search.best_params_, float(search.best_score_)
    pred = best.predict(X_test)
    prob = best.predict_proba(X_test)[:, 1]
    dummy_pred = dummy.predict(X_test)
    metrics = {
        "records": int(len(prepared)), "modeling_sample": int(len(X_raw)),
        "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
        "positive_rate_test": float(y_test.mean()),
        "tuning": args.tuning, "best_params": best_params, "cv_f1_damage": cv_f1,
        "test": {"f1_damage": float(f1_score(y_test, pred)),
                 "recall_damage": float(recall_score(y_test, pred)),
                 "precision_damage": float(precision_score(y_test, pred, zero_division=0)),
                 "accuracy": float(accuracy_score(y_test, pred)),
                 "roc_auc": float(roc_auc_score(y_test, prob)),
                 "confusion_matrix": confusion_matrix(y_test, pred).tolist()},
        "dummy_test": {"f1_damage": float(f1_score(y_test, dummy_pred, zero_division=0)),
                       "recall_damage": float(recall_score(y_test, dummy_pred, zero_division=0)),
                       "accuracy": float(accuracy_score(y_test, dummy_pred))},
        "classification_report": classification_report(y_test, pred, output_dict=True,
                                                         zero_division=0),
        "warning": "Clasifica sidewalk observado en el censo 2015; no predice daño futuro.",
    }
    (ROOT / "artifacts").mkdir(parents=True, exist_ok=True)
    joblib.dump(best, MODEL_PATH)
    (ROOT / "artifacts" / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print("Artefacto guardado:", MODEL_PATH)


if __name__ == "__main__":
    main()
