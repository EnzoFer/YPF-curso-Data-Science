"""Preprocesamiento común entre entrenamiento y servicio de inferencia."""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "artifacts" / "sidewalk_damage_pipeline.joblib"
NUMERIC = ["tree_dbh", "root_conflict_score", "trunk_conflict_score", "branch_conflict_score"]
ORDINAL = ["steward", "health"]
NOMINAL = ["borough", "guards", "curb_loc", "spc_latin"]
FEATURES = NUMERIC + ORDINAL + NOMINAL
STEWARD_CATEGORIES = ["None", "1or2", "3or4", "4orMore"]
HEALTH_CATEGORIES = ["Poor", "Fair", "Good"]


def engineer_features(frame: pd.DataFrame, top_species: list[str] | None = None) -> pd.DataFrame:
    """Convierte observaciones originales a las variables usadas en el notebook."""
    data = frame.copy()
    # NYC Open Data nombra este campo boroname; algunas copias del curso lo renombran borough.
    if "borough" not in data and "boroname" in data:
        data["borough"] = data["boroname"]
    for col in ["steward", "guards"]:
        if col in data:
            data[col] = data[col].fillna("None").replace(r"^\s*$", "None", regex=True)
    data["health"] = data["health"].fillna("Good").replace(r"^\s*$", "Good", regex=True)
    data["curb_loc"] = data["curb_loc"].fillna("OnCurb").replace(r"^\s*$", "OnCurb", regex=True)
    for col in ["borough", "spc_latin"]:
        data[col] = data[col].fillna("Unknown").replace(r"^\s*$", "Unknown", regex=True)
    groups = {
        "root_conflict_score": ["root_stone", "root_grate", "root_other"],
        "trunk_conflict_score": ["trunk_wire", "trnk_light", "trnk_other"],
        "branch_conflict_score": ["brch_light", "brch_shoe", "brch_other"],
    }
    for output, cols in groups.items():
        data[output] = data[cols].eq("Yes").sum(axis=1)
    if top_species is None:
        top_species = data["spc_latin"].value_counts().head(20).index.tolist()
    data["spc_clean"] = data["spc_latin"].where(data["spc_latin"].isin(top_species), "Other")
    data["tree_dbh"] = pd.to_numeric(data["tree_dbh"], errors="coerce")
    if "sidewalk" in data:
        data["target"] = data["sidewalk"].map({"Damage": 1, "NoDamage": 0})
    return data


def make_pipeline(n_estimators: int = 200, max_depth: int = 16,
                  min_samples_leaf: int = 10) -> Pipeline:
    preprocessor = ColumnTransformer(transformers=[
        ("num", "passthrough", NUMERIC),
        ("ord", OrdinalEncoder(
            categories=[STEWARD_CATEGORIES, HEALTH_CATEGORIES],
            handle_unknown="use_encoded_value", unknown_value=-1,
        ), ORDINAL),
        ("nom", OneHotEncoder(handle_unknown="ignore", min_frequency=10), NOMINAL),
    ])
    classifier = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_leaf=min_samples_leaf, class_weight="balanced",
        # El paralelismo externo evita sobreasignación en GridSearchCV.
        n_jobs=1, random_state=42,
    )
    return Pipeline([("prep", preprocessor), ("clf", classifier)])


def load_model(path: Path = MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError("Modelo no entrenado. Ejecutá `python train.py`.")
    return joblib.load(path)
