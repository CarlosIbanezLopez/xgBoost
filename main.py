from __future__ import annotations

import json
import logging
import os
from typing import Literal, Optional

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from db import fetch_comparable_listings, fetch_nearest_zone_cluster
from config import (
    CLASSIFIER_PATH,
    REGRESSOR_PATH,
)
from ml_pipeline import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    load_classification_bundle,
    load_regression_bundle,
    load_regression_no_pub_bundle,
    save_models,
    train_models,
)


app = FastAPI(title="XGBoost Property Service", version="1.0.0")

logger = logging.getLogger(__name__)


class TrainResponse(BaseModel):
    reg_mae: float = Field(..., description="Mean Absolute Error del modelo de regresión")


class PredictRequest(BaseModel):
    # Numéricas
    latitude: float
    longitude: float
    cluster_zona: Optional[int] = None
    m2_construidos: float = 0
    m2_terreno: float = 0
    dormitorios: int = 0
    banos: int = 0
    estacionamientos: int = 0
    antiguedad: int = 0
    # Si viene None o 0, se usará el modelo alternativo sin precio_publicacion
    precio_publicacion: Optional[float] = None
    precio_m2: Optional[float] = None
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
    ciudad: Optional[str] = None
    pais: Optional[str] = None
    status: Optional[str] = "Activa"
    transaction_type: Optional[str] = "Venta"


class RegressionPrediction(BaseModel):
    predicted_price: float
    expected_abs_error: float
    expected_pct_error: float
    interval_approx: dict
    comparables: list[dict] = Field(default_factory=list)


class ClassificationPrediction(BaseModel):
    price_segment: Literal["bajo", "medio", "alto"]
    probabilities: dict


# --- Seguridad básica para /train ---
API_KEY_TRAIN = os.getenv("API_KEY_TRAIN")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_train_key(api_key: str = Depends(api_key_header)):
    if not API_KEY_TRAIN:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Train API not configured",
        )
    if api_key != API_KEY_TRAIN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )


@app.post("/train", response_model=TrainResponse, dependencies=[Depends(require_train_key)])
def train_endpoint():
    """
    Entrena ambos modelos (regresión y clasificación) leyendo datos desde Postgres.
    """
    models = train_models()
    save_models(models)
    return TrainResponse(reg_mae=models.reg_mae)


_regression_bundle = None
_regression_no_pub_bundle = None
_classification_bundle = None


def _load_bundles_or_500():
    global _regression_bundle, _regression_no_pub_bundle, _classification_bundle
    try:
        if _regression_bundle is None or _regression_no_pub_bundle is None or _classification_bundle is None:
            _regression_bundle = load_regression_bundle()
            _regression_no_pub_bundle = load_regression_no_pub_bundle()
            _classification_bundle = load_classification_bundle()
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="Modelos no encontrados. Ejecuta primero POST /train o corre ml_pipeline.py.",
        )
    return _regression_bundle, _regression_no_pub_bundle, _classification_bundle


def _prepare_feature_row(payload: PredictRequest, encoder, numeric_features, categorical_features):
    data = payload.dict()

    # Calcular cluster_zona/ciudad/pais desde coordenadas (si no vienen)
    if data.get("cluster_zona") in (None, 0) or not data.get("ciudad") or not data.get("pais"):
        nearest = fetch_nearest_zone_cluster(
            latitude=float(data["latitude"]),
            longitude=float(data["longitude"]),
        )
        if nearest:
            if data.get("cluster_zona") in (None, 0):
                data["cluster_zona"] = int(nearest["cluster_id"])
            if not data.get("ciudad"):
                data["ciudad"] = str(nearest["ciudad"])
            if not data.get("pais"):
                data["pais"] = str(nearest.get("pais") or "")

    # Calcular precio_m2 si no viene o viene en 0
    if data.get("precio_m2") in (None, 0):
        area = data.get("m2_construidos") or data.get("m2_terreno") or 0
        if area and area > 0 and (data.get("precio_publicacion") not in (None, 0)):
            data["precio_m2"] = float(data["precio_publicacion"]) / float(area)
        else:
            data["precio_m2"] = 0.0

    # Mostrar en consola lo que entra al pipeline (enriquecido)
    enriched_json = json.dumps(data, ensure_ascii=False)
    print(f"[xgboost] input_enriched_json={enriched_json}", flush=True)
    logger.info("XGBoost input (enriched): %s", enriched_json)

    def _num_value(col: str) -> float:
        v = data.get(col, 0)
        if v is None:
            v = 0
        return float(v)

    X_num = np.array([[ _num_value(col) for col in numeric_features ]])
    X_cat_raw = [[str(data.get(col, "Desconocido")) for col in categorical_features]]
    X_cat = encoder.transform(X_cat_raw)
    X_row = np.hstack([X_num, X_cat])

    # Mostrar en consola el vector numérico final que entra al modelo
    x_vec = X_row.tolist()
    print(f"[xgboost] input_vector_numeric={x_vec}", flush=True)
    logger.info("XGBoost vector: %s", x_vec)
    return X_row


@app.post("/predict/regression", response_model=RegressionPrediction)
def predict_regression(payload: PredictRequest):
    """
    Predice el precio de venta de una propiedad.
    """
    reg_bundle, reg_no_pub_bundle, _ = _load_bundles_or_500()

    use_no_pub = payload.precio_publicacion in (None, 0)
    active_bundle = reg_no_pub_bundle if use_no_pub else reg_bundle

    regressor = active_bundle["regressor"]
    encoder = active_bundle["encoder"]
    numeric_features = active_bundle.get("numeric_features", NUMERIC_FEATURES)
    categorical_features = active_bundle.get("categorical_features", CATEGORICAL_FEATURES)
    reg_mae = float(active_bundle.get("reg_mae", 0.0))
    price_bins = np.array(active_bundle.get("price_bins"))
    mean_abs_pct_error_by_bin = np.array(active_bundle.get("mean_abs_pct_error_by_bin"))

    X_row = _prepare_feature_row(payload, encoder, numeric_features, categorical_features)
    pred = float(regressor.predict(X_row)[0])

    # Error absoluto esperado global (para compatibilidad)
    expected_abs_error = reg_mae

    # Error porcentual esperado según el rango de precio predicho
    if price_bins is not None and mean_abs_pct_error_by_bin is not None:
        # encontrar el bin para este precio predicho
        bin_idx = int(np.digitize(pred, price_bins, right=True) - 1)
        bin_idx = max(0, min(bin_idx, len(mean_abs_pct_error_by_bin) - 1))
        expected_pct_error = float(mean_abs_pct_error_by_bin[bin_idx])
    else:
        # fallback: usar MAE global sobre el propio precio
        expected_pct_error = float(expected_abs_error / pred) if pred != 0 else 0.0

    # Intervalo aproximado basado en el error porcentual
    lower = max(pred * (1.0 - expected_pct_error), 0.0)
    upper = pred * (1.0 + expected_pct_error)

    comparables = fetch_comparable_listings(
        latitude=payload.latitude,
        longitude=payload.longitude,
        ciudad=payload.ciudad,
        pais=payload.pais,
        tipo_propiedad=payload.tipo_propiedad,
        m2_construidos=payload.m2_construidos,
        m2_terreno=payload.m2_terreno,
        limit=20,
    )

    return RegressionPrediction(
        predicted_price=pred,
        expected_abs_error=expected_abs_error,
        expected_pct_error=expected_pct_error,
        interval_approx={"lower": lower, "upper": upper},
        comparables=comparables,
    )


@app.post("/predict/classification", response_model=ClassificationPrediction)
def predict_classification(payload: PredictRequest):
    """
    Clasifica la propiedad en segmento de precio (bajo/medio/alto).
    """
    _, _, clf_bundle = _load_bundles_or_500()
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

