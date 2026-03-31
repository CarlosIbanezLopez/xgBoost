from __future__ import annotations

import json
import logging
import os
from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from config import VALID_COMBINATIONS, get_model_indices
from db import (
    fetch_comparable_listings,
    fetch_location_market_stats,
    fetch_nearest_zone_cluster,
)
from ml_pipeline import (
    load_bundle,
    load_terrain_bundle,
    save_all_bundles,
    save_all_terrain_bundles,
    train_all_models,
    train_all_terrain_models,
)

app = FastAPI(title="XGBoost Property Service", version="2.1.0")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schemas de entrada
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    # Routing obligatorio
    tipo_transaccion: Literal["Venta", "Alquiler"] = Field(
        ..., description="Tipo de transacción: 'Venta' o 'Alquiler'"
    )
    segmento: Literal["Residencial", "Comercial"] = Field(
        ..., description="Segmento: 'Residencial' o 'Comercial'"
    )

    # Numéricas
    latitude: float
    longitude: float
    m2_construidos: float = 0
    m2_terreno: float = 0
    dormitorios: int = 0
    banos: int = 0
    estacionamientos: int = 0
    antiguedad: int = 0
    precio_publicacion: Optional[float] = None
    precio_alquiler_mes: Optional[float] = None
    precio_m2: Optional[float] = None
    tiempo_en_mercado: int = 0
    numero_reducciones: int = 0
    diferencia_vs_promedio_zona: float = 0
    ratio_activas_vendidas_zona: float = 0
    mes_publicacion: int = 0
    anio_publicacion: int = 0

    # Categóricas — se aceptan como string libre; el encoder maneja valores desconocidos
    tipo_propiedad: Optional[str] = Field(
        "Casa",
        description="Ej: 'Casa', 'Departamento', 'Local Comercial', 'Oficina', 'Terreno', 'Otro'"
    )
    estado_propiedad: Optional[str] = Field(
        "Sin especificar",
        description="Ej: 'Nuevo', 'En buen estado', 'En construcción', 'Para renovar'"
    )
    ciudad: Optional[str] = None
    pais: Optional[str] = None


# ---------------------------------------------------------------------------
# Schemas de salida
# ---------------------------------------------------------------------------

class TrainResponse(BaseModel):
    results: list[dict]


class RegressionPrediction(BaseModel):
    predicted_price: float
    expected_abs_error: float
    expected_pct_error: float
    interval_approx: dict
    model_used: str
    comparables: list[dict] = Field(default_factory=list)


class ClassificationPrediction(BaseModel):
    price_segment: Literal["bajo", "medio", "alto"]
    probabilities: dict
    model_used: str


class SaleProbabilityPrediction(BaseModel):
    """
    Probabilidad de que la propiedad se venda/alquile en cada horizonte temporal.
    Las probabilidades son acumuladas (≤ 30 días incluye las más rápidas, etc.)
    """
    prob_30_days: float = Field(..., description="P(venta en ≤ 30 días)")
    prob_60_days: float = Field(..., description="P(venta en ≤ 60 días)")
    prob_90_days: float = Field(..., description="P(venta en ≤ 90 días)")
    prob_over_90_days: float = Field(..., description="P(venta en > 90 días)")
    expected_speed: Literal["≤30 días", "31–60 días", "61–90 días", ">90 días"]
    model_used: str
    model_accuracy: float = Field(..., description="Accuracy del clasificador en test set")


# ---------------------------------------------------------------------------
# Seguridad para /train
# ---------------------------------------------------------------------------

API_KEY_TRAIN = os.getenv("API_KEY_TRAIN")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_train_key(api_key: str = Depends(api_key_header)):
    if not API_KEY_TRAIN:
        raise HTTPException(status_code=500, detail="Train API not configured")
    if api_key != API_KEY_TRAIN:
        raise HTTPException(status_code=401, detail="Invalid API key")


# ---------------------------------------------------------------------------
# Caché de bundles en memoria
# ---------------------------------------------------------------------------

_bundle_cache: Dict[tuple, dict] = {}
_terrain_bundle_cache: Dict[str, dict] = {}


def _get_bundle(tipo_transaccion: str, segmento: str) -> dict:
    key = (tipo_transaccion, segmento)
    if key not in _bundle_cache:
        try:
            _bundle_cache[key] = load_bundle(tipo_transaccion, segmento)
        except FileNotFoundError:
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Modelos no encontrados para {tipo_transaccion}/{segmento}. "
                    "Ejecuta POST /train primero."
                ),
            )
    return _bundle_cache[key]


def _get_terrain_bundle(tipo_transaccion: str) -> dict:
    key = tipo_transaccion
    if key not in _terrain_bundle_cache:
        _terrain_bundle_cache[key] = load_terrain_bundle(tipo_transaccion)
    return _terrain_bundle_cache[key]


def _invalidate_cache() -> None:
    _bundle_cache.clear()
    _terrain_bundle_cache.clear()


# ---------------------------------------------------------------------------
# Preparación de features
# ---------------------------------------------------------------------------

def _enrich(data: dict) -> dict:
    """Infiere ciudad/pais desde coordenadas si no vienen en el request."""
    if not data.get("ciudad") or not data.get("pais"):
        nearest = fetch_nearest_zone_cluster(
            latitude=float(data["latitude"]),
            longitude=float(data["longitude"]),
        )
        if nearest:
            if not data.get("ciudad"):
                data["ciudad"] = str(nearest["ciudad"])
            if not data.get("pais"):
                data["pais"] = str(nearest.get("pais") or "")
    return data


def _normalize_tipo_propiedad(tipo_propiedad: Optional[str]) -> str:
    return (tipo_propiedad or "").strip().lower()


def _is_terrain_request(tipo_propiedad: Optional[str]) -> bool:
    return _normalize_tipo_propiedad(tipo_propiedad) == "terreno"


def _has_price_basis(data: dict) -> bool:
    if float(data.get("precio_m2") or 0) > 0:
        return True
    if float(data.get("precio_publicacion") or 0) > 0:
        return True
    if data.get("tipo_transaccion") == "Alquiler":
        return float(data.get("precio_alquiler_mes") or 0) > 0
    return False


def _apply_market_fallbacks(data: dict, *, use_no_pub: bool) -> dict:
    """
    Cuando falta precio_publicacion, estima señales de mercado desde la zona.

    Esto evita que el modelo no_pub vea precio_m2=0 para cualquier coordenada
    y termine devolviendo respuestas demasiado parecidas entre zonas distintas.
    """
    needs_precio_m2 = data.get("precio_m2") in (None, 0)
    needs_ratio = data.get("ratio_activas_vendidas_zona") in (None, 0)

    if not use_no_pub and not needs_ratio:
        return data
    if not needs_precio_m2 and not needs_ratio:
        return data
    if not use_no_pub and _has_price_basis(data):
        return data

    market = fetch_location_market_stats(
        latitude=float(data["latitude"]),
        longitude=float(data["longitude"]),
        tipo_transaccion=str(data["tipo_transaccion"]),
        segmento=str(data["segmento"]),
        ciudad=data.get("ciudad"),
        pais=data.get("pais"),
    )
    if not market:
        return data

    if not data.get("ciudad") and market.get("ciudad"):
        data["ciudad"] = str(market["ciudad"])
    if not data.get("pais") and market.get("pais"):
        data["pais"] = str(market["pais"])

    if needs_precio_m2 and not _has_price_basis(data):
        precio_m2_mediana = float(market.get("precio_m2_mediana") or 0.0)
        if precio_m2_mediana > 0:
            data["precio_m2"] = precio_m2_mediana

    if needs_ratio:
        ratio = float(market.get("ratio_activas_vendidas_zona") or 0.0)
        if ratio > 0:
            data["ratio_activas_vendidas_zona"] = ratio

    return data


def _build_X(
    data: dict,
    numeric_features: list[str],
    encoder,
    categorical_features: Optional[list[str]] = None,
) -> np.ndarray:
    """Construye el vector de features listo para predict."""

    # Calcular precio_m2 si falta
    if data.get("precio_m2") in (None, 0):
        area = data.get("m2_construidos") or data.get("m2_terreno") or 0
        pub  = data.get("precio_publicacion") or 0
        data["precio_m2"] = float(pub) / float(area) if area > 0 and pub > 0 else 0.0

    def _num(col: str) -> float:
        v = data.get(col, 0)
        return float(v) if v is not None else 0.0

    X_num = np.array([[_num(c) for c in numeric_features]])
    
    categorical_features = categorical_features or []
    if encoder is None or not categorical_features:
        return X_num

    # Creamos un DataFrame para evitar el UserWarning de Scikit-Learn
    # "X does not have valid feature names, but OrdinalEncoder was fitted with feature names"
    X_cat_raw = pd.DataFrame(
        [[str(data.get(c) or "Desconocido") for c in categorical_features]],
        columns=categorical_features,
    )
    X_cat = encoder.transform(X_cat_raw)
    return np.hstack([X_num, X_cat])


def _has_pub_price(payload: PredictRequest) -> bool:
    if payload.precio_publicacion not in (None, 0):
        return True
    if payload.tipo_transaccion == "Alquiler":
        return payload.precio_alquiler_mes not in (None, 0)
    return False


def _has_terrain_pub_price(data: dict) -> bool:
    return float(data.get("precio_publicacion") or 0) > 0


def _inverse_target_transform(value: float, target_transform: Optional[str]) -> float:
    if target_transform == "log1p":
        return float(max(np.expm1(value), 0.0))
    return float(value)


def _expected_pct_error(pred: float, price_bins: np.ndarray, pct_errors: np.ndarray) -> float:
    if len(pct_errors) == 0:
        return 0.0
    bin_idx = max(0, min(
        int(np.digitize(pred, price_bins, right=True) - 1),
        len(pct_errors) - 1,
    ))
    return float(pct_errors[bin_idx])


def _log_input(data: dict, tag: str) -> None:
    j = json.dumps(data, ensure_ascii=False, default=str)
    logger.info("[%s] input: %s", tag, j)
    print(f"[xgboost][{tag}] {j}", flush=True)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/train", response_model=TrainResponse, dependencies=[Depends(require_train_key)])
def train_endpoint():
    """Entrena los bundles generales y el regressor especializado de Terreno."""
    bundles = train_all_models()
    save_all_bundles(bundles)
    terrain_bundles = train_all_terrain_models()
    save_all_terrain_bundles(terrain_bundles)
    _invalidate_cache()

    results = [
        {
            "tipo_transaccion": b.tipo_transaccion,
            "segmento":         b.segmento,
            "mae_pub":          round(b.reg_mae_pub, 2),
            "mae_no_pub":       round(b.reg_mae_no_pub, 2),
            "clf_precio_acc":   round(b.clf_accuracy, 4),
            "clf_precio_f1":    round(b.clf_f1_weighted, 4),
            "clf_venta_acc":    round(b.sale_prob_accuracy, 4),
        }
        for b in bundles.values()
    ]
    results.extend(
        {
            "tipo_transaccion": b.tipo_transaccion,
            "segmento": "Terreno",
            "mae_pub": round(b.reg_mae_pub, 2),
            "mae_no_pub": round(b.reg_mae_no_pub, 2),
            "sample_count": b.sample_count,
        }
        for b in terrain_bundles.values()
    )
    return TrainResponse(results=results)


@app.post("/train/terrain", response_model=TrainResponse, dependencies=[Depends(require_train_key)])
def train_terrain_endpoint():
    """Entrena solo los regressors especializados para Terreno."""
    bundles = train_all_terrain_models()
    save_all_terrain_bundles(bundles)
    _invalidate_cache()

    results = [
        {
            "tipo_transaccion": b.tipo_transaccion,
            "segmento": "Terreno",
            "mae_pub": round(b.reg_mae_pub, 2),
            "mae_no_pub": round(b.reg_mae_no_pub, 2),
            "sample_count": b.sample_count,
        }
        for b in bundles.values()
    ]
    return TrainResponse(results=results)


@app.post("/predict/regression", response_model=RegressionPrediction)
def predict_regression(payload: PredictRequest):
    """
    Predice el precio de venta o alquiler.
    Usa automáticamente el modelo con/sin precio_publicacion según lo que venga.
    """
    data = _enrich(payload.dict())
    terrain_bundle = None
    if _is_terrain_request(data.get("tipo_propiedad")):
        try:
            terrain_bundle = _get_terrain_bundle(payload.tipo_transaccion)
        except FileNotFoundError:
            terrain_bundle = None

    if terrain_bundle is not None:
        use_no_pub = not _has_terrain_pub_price(data)
        _log_input(data, "regression-terreno")

        numeric_features = (
            terrain_bundle["numeric_features_no_pub"] if use_no_pub
            else terrain_bundle["numeric_features_pub"]
        )
        X_row = _build_X(
            data,
            numeric_features,
            terrain_bundle["encoder"],
            terrain_bundle.get("categorical_features"),
        )

        regressor = (
            terrain_bundle["regressor_no_pub"] if use_no_pub
            else terrain_bundle["regressor_pub"]
        )
        pred = _inverse_target_transform(
            float(regressor.predict(X_row)[0]),
            terrain_bundle.get("target_transform"),
        )

        reg_mae = (
            terrain_bundle["reg_mae_no_pub"] if use_no_pub
            else terrain_bundle["reg_mae_pub"]
        )
        price_bins = (
            terrain_bundle["price_bins_no_pub"] if use_no_pub
            else terrain_bundle["price_bins_pub"]
        )
        pct_errors = (
            terrain_bundle["mean_abs_pct_error_no_pub"] if use_no_pub
            else terrain_bundle["mean_abs_pct_error_pub"]
        )
        expected_pct = _expected_pct_error(pred, price_bins, pct_errors)
        model_tag = (
            f"terrain_{payload.tipo_transaccion}_no_pub"
            if use_no_pub
            else f"terrain_{payload.tipo_transaccion}_pub"
        )

        comparables = fetch_comparable_listings(
            latitude=payload.latitude,
            longitude=payload.longitude,
            ciudad=data.get("ciudad"),
            pais=data.get("pais"),
            tipo_propiedad=data.get("tipo_propiedad"),
            segmento=None,
            tipo_transaccion=payload.tipo_transaccion,
            m2_construidos=0,
            m2_terreno=payload.m2_terreno,
            dormitorios=0,
            banos=0,
            precio_m2_referencia=0.0,
            limit=20,
        )

        return RegressionPrediction(
            predicted_price=pred,
            expected_abs_error=float(reg_mae),
            expected_pct_error=expected_pct,
            interval_approx={
                "lower": max(pred * (1.0 - expected_pct), 0.0),
                "upper": pred * (1.0 + expected_pct),
            },
            model_used=model_tag,
            comparables=comparables,
        )

    try:
        idx_pub, idx_no_pub = get_model_indices(payload.tipo_transaccion, payload.segmento)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    bundle = _get_bundle(payload.tipo_transaccion, payload.segmento)
    use_no_pub = not _has_pub_price(payload)

    data = _apply_market_fallbacks(data, use_no_pub=use_no_pub)
    _log_input(data, "regression")

    numeric_features = (
        bundle["numeric_features_no_pub"] if use_no_pub
        else bundle["numeric_features_pub"]
    )
    X_row = _build_X(
        data,
        numeric_features,
        bundle["encoder"],
        bundle.get("categorical_features"),
    )

    regressor = bundle["regressor_no_pub"] if use_no_pub else bundle["regressor_pub"]
    pred = float(regressor.predict(X_row)[0])

    reg_mae = bundle["reg_mae_no_pub"] if use_no_pub else bundle["reg_mae_pub"]
    price_bins = bundle["price_bins_no_pub"] if use_no_pub else bundle["price_bins_pub"]
    pct_errors = bundle["mean_abs_pct_error_no_pub"] if use_no_pub else bundle["mean_abs_pct_error_pub"]
    expected_pct = _expected_pct_error(pred, price_bins, pct_errors)
    model_tag = (
        f"m{idx_no_pub}_{payload.tipo_transaccion}_{payload.segmento}_no_pub"
        if use_no_pub
        else f"m{idx_pub}_{payload.tipo_transaccion}_{payload.segmento}_pub"
    )

    comparables = fetch_comparable_listings(
        latitude=payload.latitude,
        longitude=payload.longitude,
        ciudad=data.get("ciudad"),
        pais=data.get("pais"),
        tipo_propiedad=data.get("tipo_propiedad"),
        segmento=payload.segmento,
        tipo_transaccion=payload.tipo_transaccion,
        m2_construidos=payload.m2_construidos,
        m2_terreno=payload.m2_terreno,
        dormitorios=int(data.get("dormitorios") or 0),
        banos=int(data.get("banos") or 0),
        precio_m2_referencia=float(data.get("precio_m2") or 0.0),
        limit=20,
    )

    return RegressionPrediction(
        predicted_price=pred,
        expected_abs_error=float(reg_mae),
        expected_pct_error=expected_pct,
        interval_approx={
            "lower": max(pred * (1.0 - expected_pct), 0.0),
            "upper": pred * (1.0 + expected_pct),
        },
        model_used=model_tag,
        comparables=comparables,
    )


@app.post("/predict/classification", response_model=ClassificationPrediction)
def predict_classification(payload: PredictRequest):
    """Clasifica la propiedad en segmento de precio: bajo / medio / alto."""
    try:
        idx_pub, _ = get_model_indices(payload.tipo_transaccion, payload.segmento)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    bundle = _get_bundle(payload.tipo_transaccion, payload.segmento)
    data   = _enrich(payload.dict())
    _log_input(data, "classification")

    X_row = _build_X(
        data,
        bundle["numeric_features_pub"],
        bundle["encoder"],
        bundle.get("categorical_features"),
    )
    probs = bundle["classifier"].predict_proba(X_row)[0]

    idx_to_label = {0: "bajo", 1: "medio", 2: "alto"}
    label    = idx_to_label[int(np.argmax(probs))]
    prob_dict = {idx_to_label[i]: float(p) for i, p in enumerate(probs)}

    return ClassificationPrediction(
        price_segment=label,
        probabilities=prob_dict,
        model_used=f"m{idx_pub}_{payload.tipo_transaccion}_{payload.segmento}_clf_precio",
    )


@app.post("/predict/sale-probability", response_model=SaleProbabilityPrediction)
def predict_sale_probability(payload: PredictRequest):
    """
    Estima la probabilidad de que la propiedad se venda/alquile en
    ≤ 30, ≤ 60, ≤ 90 días o más de 90 días.

    Basado en un XGBClassifier entrenado sobre el tiempo_en_mercado histórico.
    Las probabilidades son acumuladas:
      - prob_30_days  = P(venta en ≤ 30 días)
      - prob_60_days  = P(venta en ≤ 60 días)  = P(≤30) + P(31–60)
      - prob_90_days  = P(venta en ≤ 90 días)  = P(≤60) + P(61–90)
      - prob_over_90  = P(venta en > 90 días)
    """
    try:
        idx_pub, _ = get_model_indices(payload.tipo_transaccion, payload.segmento)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    bundle = _get_bundle(payload.tipo_transaccion, payload.segmento)
    data   = _enrich(payload.dict())
    _log_input(data, "sale-probability")

    # Features del modelo de velocidad (sin tiempo_en_mercado)
    num_sp = bundle["numeric_features_sale_prob"]
    X_row = _build_X(
        data,
        num_sp,
        bundle["encoder"],
        bundle.get("categorical_features"),
    )

    probs = bundle["sale_prob_classifier"].predict_proba(X_row)[0]
    # probs[0]=≤30d  probs[1]=31-60d  probs[2]=61-90d  probs[3]=>90d

    p_30   = float(probs[0])
    p_60   = float(probs[0] + probs[1])
    p_90   = float(probs[0] + probs[1] + probs[2])
    p_over = float(probs[3])

    speed_labels = ["≤30 días", "31–60 días", "61–90 días", ">90 días"]
    expected_speed = speed_labels[int(np.argmax(probs))]

    return SaleProbabilityPrediction(
        prob_30_days=round(p_30, 4),
        prob_60_days=round(p_60, 4),
        prob_90_days=round(p_90, 4),
        prob_over_90_days=round(p_over, 4),
        expected_speed=expected_speed,
        model_used=f"m{idx_pub}_{payload.tipo_transaccion}_{payload.segmento}_clf_venta",
        model_accuracy=round(bundle.get("sale_prob_accuracy", 0.0), 4),
    )


@app.get("/models", summary="Combinaciones disponibles y estado de caché")
def list_models():
    return {
        "combinations": [
            {
                "tipo_transaccion": tt,
                "segmento":         seg,
                "model_idx_pub":    idxs[0],
                "model_idx_no_pub": idxs[1],
                "loaded_in_cache":  (tt, seg) in _bundle_cache,
            }
            for (tt, seg), idxs in VALID_COMBINATIONS.items()
        ],
        "terrain_regressors": [
            {
                "tipo_transaccion": tt,
                "loaded_in_cache": tt in _terrain_bundle_cache,
            }
            for tt in ("Venta", "Alquiler")
        ],
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_cached": len(_bundle_cache),
        "terrain_models_cached": len(_terrain_bundle_cache),
    }
