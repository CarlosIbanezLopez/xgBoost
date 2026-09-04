"""Backward-compatible ASGI entry point.

The implementation lives under ``app``; ``uvicorn main:app`` remains valid.
"""

from app.api.dependencies import API_KEY_TRAIN, api_key_header, require_train_key
from app.api.routes.predictions import (
    predict_classification,
    predict_regression,
    predict_sale_probability,
)
from app.api.routes.system import health, list_models
from app.api.routes.training import train_endpoint, train_terrain_endpoint
from app.api.schemas import (
    ClassificationPrediction,
    PredictRequest,
    RegressionPrediction,
    SaleProbabilityPrediction,
    TrainResponse,
)
from app.application import app, create_app
from app.core.config import VALID_COMBINATIONS, get_model_indices
from app.infrastructure.database import (
    fetch_comparable_listings,
    fetch_location_market_stats,
    fetch_nearest_zone_cluster,
)
from app.ml.pipeline import (
    load_bundle,
    load_terrain_bundle,
    save_all_bundles,
    save_all_terrain_bundles,
    train_all_models,
    train_all_terrain_models,
)
from app.services.model_registry import (
    _bundle_cache,
    _terrain_bundle_cache,
    get_bundle as _get_bundle,
    get_terrain_bundle as _get_terrain_bundle,
    invalidate_cache as _invalidate_cache,
)
from app.services.prediction_service import (
    _apply_market_fallbacks,
    _build_X,
    _enrich,
    _expected_pct_error,
    _has_price_basis,
    _has_pub_price,
    _has_terrain_pub_price,
    _inverse_target_transform,
    _is_terrain_request,
    _log_input,
    _normalize_tipo_propiedad,
    _predict_regression_core,
)


__all__ = [
    "app",
    "create_app",
    "PredictRequest",
    "TrainResponse",
    "RegressionPrediction",
    "ClassificationPrediction",
    "SaleProbabilityPrediction",
]
