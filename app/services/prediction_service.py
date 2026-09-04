"""Prediction use cases and feature preparation."""

from __future__ import annotations

import json
import logging
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import HTTPException

from app.api.schemas import (
    ClassificationPrediction,
    PredictRequest,
    RegressionPrediction,
    SaleProbabilityPrediction,
)
from app.core.config import get_model_indices
from app.infrastructure.database import (
    fetch_comparable_listings,
    fetch_location_market_stats,
    fetch_nearest_zone_cluster,
)
from app.services.model_registry import get_bundle, get_terrain_bundle


logger = logging.getLogger(__name__)


def enrich(data: dict) -> dict:
    """Infer city and country from coordinates when they are absent."""
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


def normalize_tipo_propiedad(tipo_propiedad: Optional[str]) -> str:
    return (tipo_propiedad or "").strip().lower()


def is_terrain_request(tipo_propiedad: Optional[str]) -> bool:
    return normalize_tipo_propiedad(tipo_propiedad) == "terreno"


def has_price_basis(data: dict) -> bool:
    if float(data.get("precio_m2") or 0) > 0:
        return True
    if float(data.get("precio_publicacion") or 0) > 0:
        return True
    if data.get("tipo_transaccion") == "Alquiler":
        return float(data.get("precio_alquiler_mes") or 0) > 0
    return False


def apply_market_fallbacks(data: dict, *, use_no_pub: bool) -> dict:
    """Fill missing market signals using local comparable statistics."""
    needs_precio_m2 = data.get("precio_m2") in (None, 0)
    needs_ratio = data.get("ratio_activas_vendidas_zona") in (None, 0)

    if not use_no_pub and not needs_ratio:
        return data
    if not needs_precio_m2 and not needs_ratio:
        return data
    if not use_no_pub and has_price_basis(data):
        return data

    market = fetch_location_market_stats(
        latitude=float(data["latitude"]),
        longitude=float(data["longitude"]),
        tipo_transaccion=str(data["tipo_transaccion"]),
        segmento=str(data["segmento"]),
        tipo_propiedad=data.get("tipo_propiedad"),
        ciudad=data.get("ciudad"),
        pais=data.get("pais"),
    )
    if not market:
        return data

    if not data.get("ciudad") and market.get("ciudad"):
        data["ciudad"] = str(market["ciudad"])
    if not data.get("pais") and market.get("pais"):
        data["pais"] = str(market["pais"])

    if needs_precio_m2 and not has_price_basis(data):
        precio_m2_mediana = float(market.get("precio_m2_mediana") or 0.0)
        if precio_m2_mediana > 0:
            data["precio_m2"] = precio_m2_mediana

    if needs_ratio:
        ratio = float(market.get("ratio_activas_vendidas_zona") or 0.0)
        if ratio > 0:
            data["ratio_activas_vendidas_zona"] = ratio

    return data


def build_feature_matrix(
    data: dict,
    numeric_features: list[str],
    encoder,
    categorical_features: Optional[list[str]] = None,
) -> np.ndarray:
    """Build the model-ready feature vector for one property."""
    if "precio_m2" in numeric_features and data.get("precio_m2") in (None, 0):
        area = data.get("m2_construidos") or data.get("m2_terreno") or 0
        pub = data.get("precio_publicacion") or 0
        data["precio_m2"] = float(pub) / float(area) if area > 0 and pub > 0 else 0.0

    def numeric_value(column: str) -> float:
        value = data.get(column, 0)
        return float(value) if value is not None else 0.0

    numeric_matrix = np.array([[numeric_value(column) for column in numeric_features]])
    categorical_features = categorical_features or []
    if encoder is None or not categorical_features:
        return numeric_matrix

    categorical_frame = pd.DataFrame(
        [[str(data.get(column) or "Desconocido") for column in categorical_features]],
        columns=categorical_features,
    )
    categorical_matrix = encoder.transform(categorical_frame)
    return np.hstack([numeric_matrix, categorical_matrix])


def has_publication_price(payload: PredictRequest) -> bool:
    if payload.precio_publicacion not in (None, 0):
        return True
    if payload.tipo_transaccion == "Alquiler":
        return payload.precio_alquiler_mes not in (None, 0)
    return False


def has_terrain_publication_price(data: dict) -> bool:
    return float(data.get("precio_publicacion") or 0) > 0


def inverse_target_transform(value: float, target_transform: Optional[str]) -> float:
    if target_transform == "log1p":
        return float(max(np.expm1(value), 0.0))
    return float(value)


def expected_pct_error(
    prediction: float,
    price_bins: np.ndarray,
    pct_errors: np.ndarray,
) -> float:
    if len(pct_errors) == 0:
        return 0.0
    bin_index = max(
        0,
        min(
            int(np.digitize(prediction, price_bins, right=True) - 1),
            len(pct_errors) - 1,
        ),
    )
    return float(pct_errors[bin_index])


def log_input(data: dict, tag: str) -> None:
    serialized = json.dumps(data, ensure_ascii=False, default=str)
    logger.info("[%s] input: %s", tag, serialized)
    print(f"[xgboost][{tag}] {serialized}", flush=True)


def predict_regression_core(
    payload: PredictRequest,
    data: dict,
    *,
    log_request: bool = True,
) -> dict:
    terrain_bundle = None
    if is_terrain_request(data.get("tipo_propiedad")):
        try:
            terrain_bundle = get_terrain_bundle(payload.tipo_transaccion)
        except FileNotFoundError:
            terrain_bundle = None

    if terrain_bundle is not None:
        use_no_pub = not has_terrain_publication_price(data)
        if log_request:
            log_input(data, "regression-terreno")

        numeric_features = (
            terrain_bundle["numeric_features_no_pub"]
            if use_no_pub
            else terrain_bundle["numeric_features_pub"]
        )
        feature_matrix = build_feature_matrix(
            data,
            numeric_features,
            terrain_bundle["encoder"],
            terrain_bundle.get("categorical_features"),
        )
        regressor = (
            terrain_bundle["regressor_no_pub"]
            if use_no_pub
            else terrain_bundle["regressor_pub"]
        )
        prediction = inverse_target_transform(
            float(regressor.predict(feature_matrix)[0]),
            terrain_bundle.get("target_transform"),
        )
        reg_mae = (
            terrain_bundle["reg_mae_no_pub"]
            if use_no_pub
            else terrain_bundle["reg_mae_pub"]
        )
        price_bins = (
            terrain_bundle["price_bins_no_pub"]
            if use_no_pub
            else terrain_bundle["price_bins_pub"]
        )
        pct_errors = (
            terrain_bundle["mean_abs_pct_error_no_pub"]
            if use_no_pub
            else terrain_bundle["mean_abs_pct_error_pub"]
        )
        pct_error = expected_pct_error(prediction, price_bins, pct_errors)
        model_tag = (
            f"terrain_{payload.tipo_transaccion}_no_pub"
            if use_no_pub
            else f"terrain_{payload.tipo_transaccion}_pub"
        )
        return {
            "predicted_price": prediction,
            "expected_abs_error": float(reg_mae),
            "expected_pct_error": pct_error,
            "interval_approx": {
                "lower": max(prediction * (1.0 - pct_error), 0.0),
                "upper": prediction * (1.0 + pct_error),
            },
            "model_used": model_tag,
            "is_terrain": True,
            "data": data,
        }

    try:
        index_pub, index_no_pub = get_model_indices(
            payload.tipo_transaccion, payload.segmento
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    bundle = get_bundle(payload.tipo_transaccion, payload.segmento)
    use_no_pub = not has_publication_price(payload)
    data = apply_market_fallbacks(data, use_no_pub=use_no_pub)
    if log_request:
        log_input(data, "regression")

    numeric_features = (
        bundle["numeric_features_no_pub"]
        if use_no_pub
        else bundle["numeric_features_pub"]
    )
    feature_matrix = build_feature_matrix(
        data,
        numeric_features,
        bundle["encoder"],
        bundle.get("categorical_features"),
    )
    regressor = bundle["regressor_no_pub"] if use_no_pub else bundle["regressor_pub"]
    prediction = float(regressor.predict(feature_matrix)[0])
    reg_mae = bundle["reg_mae_no_pub"] if use_no_pub else bundle["reg_mae_pub"]
    price_bins = bundle["price_bins_no_pub"] if use_no_pub else bundle["price_bins_pub"]
    pct_errors = (
        bundle["mean_abs_pct_error_no_pub"]
        if use_no_pub
        else bundle["mean_abs_pct_error_pub"]
    )
    pct_error = expected_pct_error(prediction, price_bins, pct_errors)
    model_tag = (
        f"m{index_no_pub}_{payload.tipo_transaccion}_{payload.segmento}_no_pub"
        if use_no_pub
        else f"m{index_pub}_{payload.tipo_transaccion}_{payload.segmento}_pub"
    )
    return {
        "predicted_price": prediction,
        "expected_abs_error": float(reg_mae),
        "expected_pct_error": pct_error,
        "interval_approx": {
            "lower": max(prediction * (1.0 - pct_error), 0.0),
            "upper": prediction * (1.0 + pct_error),
        },
        "model_used": model_tag,
        "is_terrain": False,
        "data": data,
    }


def predict_regression(payload: PredictRequest) -> RegressionPrediction:
    prediction = predict_regression_core(payload, enrich(payload.model_dump()))
    data = prediction["data"]

    if prediction["is_terrain"]:
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
    else:
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
        predicted_price=prediction["predicted_price"],
        expected_abs_error=prediction["expected_abs_error"],
        expected_pct_error=prediction["expected_pct_error"],
        interval_approx=prediction["interval_approx"],
        model_used=prediction["model_used"],
        comparables=comparables,
    )


def predict_classification(payload: PredictRequest) -> ClassificationPrediction:
    try:
        index_pub, _ = get_model_indices(payload.tipo_transaccion, payload.segmento)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    bundle = get_bundle(payload.tipo_transaccion, payload.segmento)
    data = enrich(payload.model_dump())
    log_input(data, "classification")

    feature_matrix = build_feature_matrix(
        data,
        bundle.get("numeric_features_classifier_pub", bundle["numeric_features_pub"]),
        bundle["encoder"],
        bundle.get("categorical_features"),
    )
    probabilities = bundle["classifier"].predict_proba(feature_matrix)[0]
    index_to_label = {0: "bajo", 1: "medio", 2: "alto"}
    label = index_to_label[int(np.argmax(probabilities))]
    probability_map = {
        index_to_label[index]: float(probability)
        for index, probability in enumerate(probabilities)
    }
    return ClassificationPrediction(
        price_segment=label,
        probabilities=probability_map,
        model_used=f"m{index_pub}_{payload.tipo_transaccion}_{payload.segmento}_clf_precio",
    )


def predict_sale_probability(payload: PredictRequest) -> SaleProbabilityPrediction:
    try:
        index_pub, _ = get_model_indices(payload.tipo_transaccion, payload.segmento)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    bundle = get_bundle(payload.tipo_transaccion, payload.segmento)
    data = enrich(payload.model_dump())

    if float(data.get("precio_publicacion") or 0) <= 0:
        regression_prediction = predict_regression_core(
            payload,
            data.copy(),
            log_request=False,
        )
        data["precio_publicacion"] = regression_prediction["predicted_price"]
        logger.info(
            "[sale-probability] using regression suggested price %.2f from %s",
            data["precio_publicacion"],
            regression_prediction["model_used"],
        )

    log_input(data, "sale-probability")
    feature_matrix = build_feature_matrix(
        data,
        bundle["numeric_features_sale_prob"],
        bundle["encoder"],
        bundle.get("categorical_features"),
    )
    probabilities = bundle["sale_prob_classifier"].predict_proba(feature_matrix)[0]

    probability_30 = float(probabilities[0])
    probability_60 = float(probabilities[0] + probabilities[1])
    probability_90 = float(probabilities[0] + probabilities[1] + probabilities[2])
    probability_over = float(probabilities[3])
    speed_labels = ["≤30 días", "31–60 días", "61–90 días", ">90 días"]

    return SaleProbabilityPrediction(
        prob_30_days=round(probability_30, 4),
        prob_60_days=round(probability_60, 4),
        prob_90_days=round(probability_90, 4),
        prob_over_90_days=round(probability_over, 4),
        expected_speed=speed_labels[int(np.argmax(probabilities))],
        model_used=f"m{index_pub}_{payload.tipo_transaccion}_{payload.segmento}_clf_venta",
        model_accuracy=round(bundle.get("sale_prob_accuracy", 0.0), 4),
    )


# Compatibility aliases for code that imported helpers from the old main module.
_enrich = enrich
_normalize_tipo_propiedad = normalize_tipo_propiedad
_is_terrain_request = is_terrain_request
_has_price_basis = has_price_basis
_apply_market_fallbacks = apply_market_fallbacks
_build_X = build_feature_matrix
_has_pub_price = has_publication_price
_has_terrain_pub_price = has_terrain_publication_price
_inverse_target_transform = inverse_target_transform
_expected_pct_error = expected_pct_error
_log_input = log_input
def _predict_regression_core(
    payload: PredictRequest,
    data: dict,
    *,
    log_input: bool = True,
) -> dict:
    return predict_regression_core(payload, data, log_request=log_input)
