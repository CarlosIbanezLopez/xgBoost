import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)


APP_ENV = os.getenv("APP_ENV", "development")


def _get_env_var(name: str, default: str | None = None, required_in_prod: bool = False) -> str:
    """
    Lee una variable de entorno con soporte para valores obligatorios en producción.
    """
    value = os.getenv(name, default)
    if required_in_prod and APP_ENV == "production" and not value:
        raise RuntimeError(f"Missing required env var {name} in production")
    if value is None:
        raise RuntimeError(f"Missing required env var {name}")
    return value


POSTGRES_DSN = _get_env_var(
    "POSTGRES_DSN",
    default="postgresql://user:password@localhost:5432/mi_base"
    if APP_ENV == "development"
    else None,
    required_in_prod=True,
)

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

REGRESSOR_PATH = MODEL_DIR / "xgb_regressor.joblib"
REGRESSOR_NO_PUB_PATH = MODEL_DIR / "xgb_regressor_no_pub.joblib"
CLASSIFIER_PATH = MODEL_DIR / "xgb_classifier.joblib"
ENCODER_PATH = MODEL_DIR / "encoder.joblib"

RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
