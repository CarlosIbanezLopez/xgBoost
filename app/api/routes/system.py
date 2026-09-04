"""Operational and model-discovery endpoints."""

from fastapi import APIRouter

from app.core.config import VALID_COMBINATIONS
from app.services.model_registry import (
    _bundle_cache,
    _terrain_bundle_cache,
    cache_status,
)


router = APIRouter()


@router.get("/models", summary="Combinaciones disponibles y estado de caché")
def list_models():
    return {
        "combinations": [
            {
                "tipo_transaccion": tipo_transaccion,
                "segmento": segmento,
                "model_idx_pub": indices[0],
                "model_idx_no_pub": indices[1],
                "loaded_in_cache": (tipo_transaccion, segmento) in _bundle_cache,
            }
            for (tipo_transaccion, segmento), indices in VALID_COMBINATIONS.items()
        ],
        "terrain_regressors": [
            {
                "tipo_transaccion": tipo_transaccion,
                "loaded_in_cache": tipo_transaccion in _terrain_bundle_cache,
            }
            for tipo_transaccion in ("Venta", "Alquiler")
        ],
    }


@router.get("/health")
def health():
    return {"status": "ok", **cache_status()}
