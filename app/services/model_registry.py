"""In-memory access to persisted model bundles."""

from typing import Dict

from fastapi import HTTPException

from app.ml.pipeline import load_bundle, load_terrain_bundle


_bundle_cache: Dict[tuple, dict] = {}
_terrain_bundle_cache: Dict[str, dict] = {}


def get_bundle(tipo_transaccion: str, segmento: str) -> dict:
    key = (tipo_transaccion, segmento)
    if key not in _bundle_cache:
        try:
            _bundle_cache[key] = load_bundle(tipo_transaccion, segmento)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Modelos no encontrados para {tipo_transaccion}/{segmento}. "
                    "Ejecuta POST /train primero."
                ),
            ) from exc
    return _bundle_cache[key]


def get_terrain_bundle(tipo_transaccion: str) -> dict:
    if tipo_transaccion not in _terrain_bundle_cache:
        _terrain_bundle_cache[tipo_transaccion] = load_terrain_bundle(tipo_transaccion)
    return _terrain_bundle_cache[tipo_transaccion]


def invalidate_cache() -> None:
    _bundle_cache.clear()
    _terrain_bundle_cache.clear()


def cache_status() -> dict:
    return {
        "models_cached": len(_bundle_cache),
        "terrain_models_cached": len(_terrain_bundle_cache),
    }
