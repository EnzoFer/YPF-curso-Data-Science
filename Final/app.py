"""API educativa: clasifica daño observado en una vereda junto a un árbol."""
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:  # Permite `uvicorn Final.app:app` desde la raíz o `uvicorn app:app` desde Final/.
    from .model import FEATURES, engineer_features, load_model
except ImportError:
    from model import FEATURES, engineer_features, load_model


class TreeObservation(BaseModel):
    tree_dbh: float = Field(gt=0, le=60, description="Diámetro del tronco, pulgadas")
    borough: str = Field(min_length=1, max_length=40)
    curb_loc: str = Field(min_length=1, max_length=40)
    health: str = Field(min_length=1, max_length=20)
    spc_latin: str = Field(min_length=1, max_length=120)
    steward: str = "None"
    guards: str = "None"
    root_stone: str = "No"
    root_grate: str = "No"
    root_other: str = "No"
    trunk_wire: str = "No"
    trnk_light: str = "No"
    trnk_other: str = "No"
    brch_light: str = "No"
    brch_shoe: str = "No"
    brch_other: str = "No"


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.model = load_model()
    except FileNotFoundError:
        app.state.model = None
    yield


app = FastAPI(
    title="VeredaSegura — API educativa",
    description=("Clasifica el estado de vereda observado en el censo NYC 2015. "
                 "No predice evolución futura ni reemplaza una inspección municipal."),
    version="1.0.0", lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": app.state.model is not None}


@app.post("/predict")
def predict(observation: TreeObservation):
    if app.state.model is None:
        raise HTTPException(status_code=503, detail="Modelo ausente. Ejecutá python train.py.")
    raw = pd.DataFrame([observation.model_dump()])
    # OneHotEncoder aprendió categorías y niveles infrecuentes dentro del pipeline;
    # una especie nueva se procesa con handle_unknown='ignore'.
    prepared = engineer_features(raw)
    X = prepared[FEATURES]
    probability = float(app.state.model.predict_proba(X)[0, 1])
    label = int(probability >= 0.5)
    return {
        "observed_damage_class": "Damage" if label else "NoDamage",
        "probability_damage": round(probability, 4),
        "threshold": 0.5,
        "warning": "Clasificación educativa de daño observado (censo NYC 2015); no predice daño futuro ni determina intervenciones.",
    }
