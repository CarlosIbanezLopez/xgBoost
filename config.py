import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)


POSTGRES_DSN = os.getenv(
    "POSTGRES_DSN",
    "postgresql://user:password@localhost:5432/mi_base",
)

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

REGRESSOR_PATH = MODEL_DIR / "xgb_regressor.joblib"
CLASSIFIER_PATH = MODEL_DIR / "xgb_classifier.joblib"

RANDOM_STATE = 42
