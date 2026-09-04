"""Training endpoints."""

from fastapi import APIRouter, Depends

from app.api.dependencies import require_train_key
from app.api.schemas import TrainResponse
from app.services import training_service


router = APIRouter()


@router.post(
    "/train",
    response_model=TrainResponse,
    dependencies=[Depends(require_train_key)],
)
def train_endpoint() -> TrainResponse:
    """Entrena los bundles generales y el regressor especializado de Terreno."""
    return training_service.train_models()


@router.post(
    "/train/terrain",
    response_model=TrainResponse,
    dependencies=[Depends(require_train_key)],
)
def train_terrain_endpoint() -> TrainResponse:
    """Entrena solo los regressors especializados para Terreno."""
    return training_service.train_terrain_models()
