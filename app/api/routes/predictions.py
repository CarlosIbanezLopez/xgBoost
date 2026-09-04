"""Property prediction endpoints."""

from fastapi import APIRouter

from app.api.schemas import (
    ClassificationPrediction,
    PredictRequest,
    RegressionPrediction,
    SaleProbabilityPrediction,
)
from app.services import prediction_service


router = APIRouter(prefix="/predict")


@router.post("/regression", response_model=RegressionPrediction)
def predict_regression(payload: PredictRequest) -> RegressionPrediction:
    """
    Predice el precio de venta o alquiler.
    Usa automáticamente el modelo con/sin precio_publicacion según lo que venga.
    """
    return prediction_service.predict_regression(payload)


@router.post("/classification", response_model=ClassificationPrediction)
def predict_classification(payload: PredictRequest) -> ClassificationPrediction:
    """Clasifica la propiedad en segmento de precio: bajo / medio / alto."""
    return prediction_service.predict_classification(payload)


@router.post("/sale-probability", response_model=SaleProbabilityPrediction)
def predict_sale_probability(payload: PredictRequest) -> SaleProbabilityPrediction:
    """
    Estima la probabilidad de que la propiedad se venda/alquile en
    ≤ 30, ≤ 60, ≤ 90 días o más de 90 días.

    Basado en un XGBClassifier entrenado sobre el tiempo_en_mercado histórico.
    Si no llega precio_publicacion, primero estima un precio sugerido con
    el modelo de regresión y usa ese valor como señal de entrada.
    Las probabilidades son acumuladas:
      - prob_30_days  = P(venta en ≤ 30 días)
      - prob_60_days  = P(venta en ≤ 60 días)  = P(≤30) + P(31–60)
      - prob_90_days  = P(venta en ≤ 90 días)  = P(≤60) + P(61–90)
      - prob_over_90  = P(venta en > 90 días)
    """
    return prediction_service.predict_sale_probability(payload)
