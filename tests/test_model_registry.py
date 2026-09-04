from unittest.mock import patch

import pytest
from fastapi import HTTPException

from app.services import model_registry


def setup_function():
    model_registry.invalidate_cache()


def test_general_bundle_is_loaded_once():
    bundle = {"model": "loaded"}

    with patch.object(model_registry, "load_bundle", return_value=bundle) as loader:
        first = model_registry.get_bundle("Venta", "Residencial")
        second = model_registry.get_bundle("Venta", "Residencial")

    assert first is bundle
    assert second is bundle
    loader.assert_called_once_with("Venta", "Residencial")


def test_missing_general_bundle_keeps_existing_http_error():
    with patch.object(model_registry, "load_bundle", side_effect=FileNotFoundError):
        with pytest.raises(HTTPException) as error:
            model_registry.get_bundle("Venta", "Residencial")

    assert error.value.status_code == 500
    assert "POST /train" in error.value.detail
