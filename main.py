from __future__ import annotations

from typing import Literal, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import (
    CLASSIFIER_PATH,
    REGRESSOR_PATH,
)
from ml_pipeline import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    load_classification_bundle,
    load_regression_bundle,
    save_models,
    train_models,
)


app = FastAPI(title="XGBoost Property Service", version="1.0.0")


class TrainResponse(BaseModel):
    reg_mae: float = Field(..., description="Mean Absolute Error del modelo de regresión")


class PredictRequest(BaseModel):
    # Numéricas
    latitude: float
    longitude: float
    cluster_zona: int = 0
    m2_construidos: float = 0
    m2_terreno: float = 0
    dormitorios: int = 0
    banos: int = 0
    estacionamientos: int = 0
    antiguedad: int = 0
    precio_publicacion: float = 0
    precio_m2: float = 0
    tiempo_en_mercado: int = 0
    numero_reducciones: int = 0
    diferencia_vs_promedio_zona: float = 0
    ratio_activas_vendidas_zona: float = 0
    mes_publicacion: int = 0
    anio_publicacion: int = 0

    # Categóricas
    tipo_propiedad: Optional[str] = "Casa"
    subtipo_original: Optional[str] = "Casa"
    categoria_propiedad: Optional[str] = "Casa"
    estado_propiedad: Optional[str] = "Sin especificar"
    ciudad: Optional[str] = "Santa Cruz de la Sierra"
    status: Optional[str] = "Activa"
    transaction_type: Optional[str] = "Venta"


class RegressionPrediction(BaseModel):
    predicted_price: float
    expected_abs_error: float
    interval_approx: dict


class ClassificationPrediction(BaseModel):
    price_segment: Literal["bajo", "medio", "alto"]
    probabilities: dict


@app.post("/train", response_model=TrainResponse)
def train_endpoint():
    """
    Entrena ambos modelos (regresión y clasificación) leyendo datos desde Postgres.
    """
    models = train_models()
    save_models(models)
    return TrainResponse(reg_mae=models.reg_mae)


def _load_bundles_or_500():
    try:
        reg_bundle = load_regression_bundle()
        clf_bundle = load_classification_bundle()
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="Modelos no encontrados. Ejecuta primero POST /train o corre ml_pipeline.py.",
        )
    return reg_bundle, clf_bundle


def _prepare_feature_row(payload: PredictRequest, encoder, numeric_features, categorical_features):
    data = payload.dict()
    X_num = np.array([[float(data.get(col, 0)) for col in numeric_features]])
    X_cat_raw = [[str(data.get(col, "Desconocido")) for col in categorical_features]]
    X_cat = encoder.transform(X_cat_raw)
    return np.hstack([X_num, X_cat])


@app.post("/predict/regression", response_model=RegressionPrediction)
def predict_regression(payload: PredictRequest):
    """
    Predice el precio de venta de una propiedad.
    """
    reg_bundle, _ = _load_bundles_or_500()
    regressor = reg_bundle["regressor"]
    encoder = reg_bundle["encoder"]
    numeric_features = reg_bundle.get("numeric_features", NUMERIC_FEATURES)
    categorical_features = reg_bundle.get("categorical_features", CATEGORICAL_FEATURES)
    reg_mae = float(reg_bundle.get("reg_mae", 0.0))

    X_row = _prepare_feature_row(payload, encoder, numeric_features, categorical_features)
    pred = float(regressor.predict(X_row)[0])

    # Intervalo aproximado basándonos en el MAE global
    if reg_mae > 0:
        lower = max(pred - reg_mae, 0.0)
        upper = pred + reg_mae
    else:
        lower = upper = pred

    return RegressionPrediction(
        predicted_price=pred,
        expected_abs_error=reg_mae,
        interval_approx={"lower": lower, "upper": upper},
    )


@app.post("/predict/classification", response_model=ClassificationPrediction)
def predict_classification(payload: PredictRequest):
    """
    Clasifica la propiedad en segmento de precio (bajo/medio/alto).
    """
    _, clf_bundle = _load_bundles_or_500()
    classifier = clf_bundle["classifier"]
    encoder = clf_bundle["encoder"]
    numeric_features = clf_bundle.get("numeric_features", NUMERIC_FEATURES)
    categorical_features = clf_bundle.get("categorical_features", CATEGORICAL_FEATURES)

    X_row = _prepare_feature_row(payload, encoder, numeric_features, categorical_features)
    probs = classifier.predict_proba(X_row)[0]

    idx_to_label = {0: "bajo", 1: "medio", 2: "alto"}
    label = idx_to_label[int(np.argmax(probs))]
    prob_dict = {idx_to_label[i]: float(p) for i, p in enumerate(probs)}

    return ClassificationPrediction(price_segment=label, probabilities=prob_dict)


@app.get("/health")
def health():
    return {"status": "ok"}

