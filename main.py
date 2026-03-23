from __future__ import annotations

import json
import logging
import os
from typing import Dict, Literal, Optional

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from config import VALID_COMBINATIONS, get_model_indices
from db import fetch_comparable_listings, fetch_nearest_zone_cluster
from ml_pipeline import (
    CATEGORICAL_FEATURES,
    load_bundle,
    save_all_bundles,
    train_all_models,
)


app = FastAPI(title="XGBoost Property Service — Multi-Model", version="2.0.0")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class TrainResponse(BaseModel):
    results: list[dict] = Field(..., description="MAE y métricas por combinación entrenada")


class PredictRequest(BaseModel):
    # --- Routing obligatorio ---
    tipo_transaccion: Literal["Venta", "Alquiler"] = "Venta"
    segmento: Literal["Residencial", "Comercial"] = "Residencial"

    # --- Numéricas ---
    latitude: float
    longitude: float
    cluster_zona: Optional[int] = None
    m2_construidos: float = 0
    m2_terreno: float = 0
    dormitorios: int = 0
    banos: int = 0
    estacionamientos: int = 0
    antiguedad: int = 0
    precio_publicacion: Optional[float] = None
    precio_alquiler_mes: Optional[float] = None   # nuevo
    precio_m2: Optional[float] = None
    tiempo_en_mercado: int = 0
    numero_reducciones: int = 0
    diferencia_vs_promedio_zona: float = 0
    ratio_activas_vendidas_zona: float = 0
    mes_publicacion: int = 0
    anio_publicacion: int = 0

    # --- Categóricas ---
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
    model_used: str = Field(..., description="Identificador del modelo usado")
    comparables: list[dict] = Field(default_factory=list)


class ClassificationPrediction(BaseModel):
    price_segment: Literal["bajo", "medio", "alto"]
    probabilities: dict
    model_used: str


# ---------------------------------------------------------------------------
# Seguridad para /train
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Caché de bundles en memoria
# ---------------------------------------------------------------------------

_bundle_cache: Dict[tuple, dict] = {}


def _get_bundle(tipo_transaccion: str, segmento: str) -> dict:
    """
    Carga y cachea el bundle para la combinación solicitada.
    """
    key = (tipo_transaccion, segmento)
    if key not in _bundle_cache:
        try:
            _bundle_cache[key] = load_bundle(tipo_transaccion, segmento)
        except FileNotFoundError:
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Modelos no encontrados para {tipo_transaccion}/{segmento}. "
                    "Ejecuta primero POST /train."
                ),
            )
    return _bundle_cache[key]


def _invalidate_cache():
    """Limpia el caché tras reentrenamiento."""
    _bundle_cache.clear()


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------

def _prepare_feature_row(
    payload: PredictRequest,
    bundle: dict,
    use_no_pub: bool,
) -> np.ndarray:
    data = payload.dict()

    # Enriquecer con cluster/ciudad/pais si faltan
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

    # Calcular precio_m2 si falta
    if data.get("precio_m2") in (None, 0):
        area = data.get("m2_construidos") or data.get("m2_terreno") or 0
        pub  = data.get("precio_publicacion") or 0
        if area > 0 and pub > 0:
            data["precio_m2"] = float(pub) / float(area)
        else:
            data["precio_m2"] = 0.0

    enriched_json = json.dumps(data, ensure_ascii=False, default=str)
    logger.info("XGBoost input (enriched): %s", enriched_json)
    print(f"[xgboost] input_enriched_json={enriched_json}", flush=True)

    numeric_features = (
        bundle["numeric_features_no_pub"] if use_no_pub
        else bundle["numeric_features_pub"]
    )
    encoder = bundle["encoder"]

    def _num(col: str) -> float:
        v = data.get(col, 0)
        return float(v) if v is not None else 0.0

    X_num = np.array([[_num(c) for c in numeric_features]])
    X_cat_raw = [[str(data.get(c, "Desconocido")) for c in CATEGORICAL_FEATURES]]
    X_cat = encoder.transform(X_cat_raw)
    X_row = np.hstack([X_num, X_cat])

    logger.info("XGBoost vector: %s", X_row.tolist())
    return X_row


def _has_pub_price(payload: PredictRequest) -> bool:
    """True si viene precio de publicación o alquiler válido."""
    if payload.tipo_transaccion == "Alquiler":
        return payload.precio_alquiler_mes not in (None, 0)
    return payload.precio_publicacion not in (None, 0)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/train", response_model=TrainResponse, dependencies=[Depends(require_train_key)])
def train_endpoint():
    """
    Entrena los 4 bundles (8 modelos) leyendo datos filtrados desde Postgres.
    """
    bundles = train_all_models()
    save_all_bundles(bundles)
    _invalidate_cache()

    results = [
        {
            "tipo_transaccion": b.tipo_transaccion,
            "segmento":         b.segmento,
            "mae_pub":          round(b.reg_mae_pub, 2),
            "mae_no_pub":       round(b.reg_mae_no_pub, 2),
            "clf_accuracy":     round(b.clf_accuracy, 4),
            "clf_f1_weighted":  round(b.clf_f1_weighted, 4),
        }
        for b in bundles.values()
    ]
    return TrainResponse(results=results)


@app.post("/predict/regression", response_model=RegressionPrediction)
def predict_regression(payload: PredictRequest):
    """
    Predice el precio de venta o alquiler de una propiedad.
    Selecciona automáticamente el modelo correcto según
    tipo_transaccion + segmento del request.
    """
    # Validar combinación
    try:
        idx_pub, idx_no_pub = get_model_indices(payload.tipo_transaccion, payload.segmento)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    bundle = _get_bundle(payload.tipo_transaccion, payload.segmento)
    use_no_pub = not _has_pub_price(payload)
    model_tag = (
        f"m{idx_no_pub}_{payload.tipo_transaccion}_{payload.segmento}_no_pub"
        if use_no_pub
        else f"m{idx_pub}_{payload.tipo_transaccion}_{payload.segmento}_pub"
    )

    X_row = _prepare_feature_row(payload, bundle, use_no_pub)

    regressor = bundle["regressor_no_pub"] if use_no_pub else bundle["regressor_pub"]
    pred = float(regressor.predict(X_row)[0])

    # Error esperado
    reg_mae = bundle["reg_mae_no_pub"] if use_no_pub else bundle["reg_mae_pub"]
    price_bins = bundle["price_bins_no_pub"] if use_no_pub else bundle["price_bins_pub"]
    pct_errors = bundle["mean_abs_pct_error_no_pub"] if use_no_pub else bundle["mean_abs_pct_error_pub"]

    bin_idx = int(np.digitize(pred, price_bins, right=True) - 1)
    bin_idx = max(0, min(bin_idx, len(pct_errors) - 1))
    expected_pct_error = float(pct_errors[bin_idx])

    lower = max(pred * (1.0 - expected_pct_error), 0.0)
    upper = pred * (1.0 + expected_pct_error)

    comparables = fetch_comparable_listings(
        latitude=payload.latitude,
        longitude=payload.longitude,
        ciudad=payload.ciudad,
        pais=payload.pais,
        tipo_propiedad=payload.tipo_propiedad,
        segmento=payload.segmento,
        tipo_transaccion=payload.tipo_transaccion,
        m2_construidos=payload.m2_construidos,
        m2_terreno=payload.m2_terreno,
        limit=20,
    )

    return RegressionPrediction(
        predicted_price=pred,
        expected_abs_error=float(reg_mae),
        expected_pct_error=expected_pct_error,
        interval_approx={"lower": lower, "upper": upper},
        model_used=model_tag,
        comparables=comparables,
    )


@app.post("/predict/classification", response_model=ClassificationPrediction)
def predict_classification(payload: PredictRequest):
    """
    Clasifica la propiedad en segmento de precio (bajo/medio/alto).
    """
    try:
        idx_pub, _ = get_model_indices(payload.tipo_transaccion, payload.segmento)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    bundle = _get_bundle(payload.tipo_transaccion, payload.segmento)
    model_tag = f"m{idx_pub}_{payload.tipo_transaccion}_{payload.segmento}_clf"

    # El classifier siempre usa las features con precio_publicacion
    X_row = _prepare_feature_row(payload, bundle, use_no_pub=False)

    classifier = bundle["classifier"]
    probs = classifier.predict_proba(X_row)[0]

    idx_to_label = {0: "bajo", 1: "medio", 2: "alto"}
    label = idx_to_label[int(np.argmax(probs))]
    prob_dict = {idx_to_label[i]: float(p) for i, p in enumerate(probs)}

    return ClassificationPrediction(
        price_segment=label,
        probabilities=prob_dict,
        model_used=model_tag,
    )


@app.get("/models", summary="Lista los modelos disponibles y sus combinaciones")
def list_models():
    return {
        "combinations": [
            {
                "tipo_transaccion": tt,
                "segmento": seg,
                "model_idx_pub": idxs[0],
                "model_idx_no_pub": idxs[1],
                "loaded": (tt, seg) in _bundle_cache,
            }
            for (tt, seg), idxs in VALID_COMBINATIONS.items()
        ]
    }


@app.get("/health")
def health():
    return {"status": "ok", "models_cached": len(_bundle_cache)}