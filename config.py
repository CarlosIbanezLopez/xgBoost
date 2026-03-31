import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

APP_ENV = os.getenv("APP_ENV", "development")


def _get_env_var(name: str, default: str | None = None, required_in_prod: bool = False) -> str:
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

RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
TERRAIN_SPECIAL_MIN_ROWS = int(os.getenv("TERRAIN_SPECIAL_MIN_ROWS", "100"))

# ---------------------------------------------------------------------------
# Combinaciones válidas: (tipo_transaccion, segmento) → (idx_con_pub, idx_sin_pub)
#
#   Venta    + Residencial → modelos 1 (con pub) y 2 (sin pub)
#   Venta    + Comercial   → modelos 3 y 4
#   Alquiler + Residencial → modelos 5 y 6
#   Alquiler + Comercial   → modelos 7 y 8
# ---------------------------------------------------------------------------

VALID_COMBINATIONS: dict[tuple[str, str], tuple[int, int]] = {
    ("Venta",    "Residencial"): (1, 2),
    ("Venta",    "Comercial"):   (3, 4),
    ("Alquiler", "Residencial"): (5, 6),
    ("Alquiler", "Comercial"):   (7, 8),
}


def model_paths(model_idx: int) -> dict[str, Path]:
    """Paths de artefactos para un índice de modelo."""
    return {
        "regressor":  MODEL_DIR / f"xgb_regressor_m{model_idx}.joblib",
        "encoder":    MODEL_DIR / f"encoder_m{model_idx}.joblib",
        "classifier": MODEL_DIR / f"xgb_classifier_m{model_idx}.joblib",
    }


def terrain_model_paths(tipo_transaccion: str) -> dict[str, Path]:
    slug = f"terreno_{tipo_transaccion.strip().lower()}"
    return {
        "regressor_pub":    MODEL_DIR / f"xgb_regressor_{slug}_pub.joblib",
        "regressor_no_pub": MODEL_DIR / f"xgb_regressor_{slug}_no_pub.joblib",
        "encoder":          MODEL_DIR / f"encoder_{slug}.joblib",
    }


def get_model_indices(tipo_transaccion: str, segmento: str) -> tuple[int, int]:
    """Retorna (idx_con_pub, idx_sin_pub) o lanza ValueError si la combinación no existe."""
    key = (tipo_transaccion, segmento)
    if key not in VALID_COMBINATIONS:
        raise ValueError(
            f"Combinación no soportada: tipo_transaccion='{tipo_transaccion}', "
            f"segmento='{segmento}'. Válidas: {list(VALID_COMBINATIONS.keys())}"
        )
    return VALID_COMBINATIONS[key]
