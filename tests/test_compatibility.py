from pathlib import Path

import app.core.config as application_config
import app.infrastructure.database as application_database
import app.ml.pipeline as application_pipeline
import config
import db
import main
import ml_pipeline
from app.application import app


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_root_modules_remain_compatible_aliases():
    assert config is application_config
    assert db is application_database
    assert ml_pipeline is application_pipeline
    assert main.app is app


def test_runtime_paths_remain_at_project_root():
    assert config.BASE_DIR == PROJECT_ROOT
    assert config.ENV_PATH == PROJECT_ROOT / ".env"
    assert config.MODEL_DIR == PROJECT_ROOT / "models"


def test_model_routing_contract_is_unchanged():
    assert config.get_model_indices("Venta", "Residencial") == (1, 2)
    assert config.get_model_indices("Venta", "Comercial") == (3, 4)
    assert config.get_model_indices("Alquiler", "Residencial") == (5, 6)
    assert config.get_model_indices("Alquiler", "Comercial") == (7, 8)
