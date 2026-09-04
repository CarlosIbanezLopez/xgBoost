"""Request and response contracts exposed by the HTTP API."""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    tipo_transaccion: Literal["Venta", "Alquiler"] = Field(
        ..., description="Tipo de transacción: 'Venta' o 'Alquiler'"
    )
    segmento: Literal["Residencial", "Comercial"] = Field(
        ..., description="Segmento: 'Residencial' o 'Comercial'"
    )

    latitude: float
    longitude: float
    m2_construidos: float = 0
    m2_terreno: float = 0
    dormitorios: int = 0
    banos: int = 0
    estacionamientos: int = 0
    antiguedad: int = 0
    precio_publicacion: Optional[float] = None
    precio_alquiler_mes: Optional[float] = None
    precio_m2: Optional[float] = None
    tiempo_en_mercado: int = 0
    numero_reducciones: int = 0
    diferencia_vs_promedio_zona: float = 0
    ratio_activas_vendidas_zona: float = 0
    mes_publicacion: int = 0
    anio_publicacion: int = 0

    tipo_propiedad: Optional[str] = Field(
        "Casa",
        description="Ej: 'Casa', 'Departamento', 'Local Comercial', 'Oficina', 'Terreno', 'Otro'",
    )
    estado_propiedad: Optional[str] = Field(
        "Sin especificar",
        description="Ej: 'Nuevo', 'En buen estado', 'En construcción', 'Para renovar'",
    )
    ciudad: Optional[str] = None
    pais: Optional[str] = None


class TrainResponse(BaseModel):
    results: list[dict]


class RegressionPrediction(BaseModel):
    predicted_price: float
    expected_abs_error: float
    expected_pct_error: float
    interval_approx: dict
    model_used: str
    comparables: list[dict] = Field(default_factory=list)


class ClassificationPrediction(BaseModel):
    price_segment: Literal["bajo", "medio", "alto"]
    probabilities: dict
    model_used: str


class SaleProbabilityPrediction(BaseModel):
    """
    Probabilidad de que la propiedad se venda/alquile en cada horizonte temporal.
    Las probabilidades son acumuladas (≤ 30 días incluye las más rápidas, etc.)
    """

    prob_30_days: float = Field(..., description="P(venta en ≤ 30 días)")
    prob_60_days: float = Field(..., description="P(venta en ≤ 60 días)")
    prob_90_days: float = Field(..., description="P(venta en ≤ 90 días)")
    prob_over_90_days: float = Field(..., description="P(venta en > 90 días)")
    expected_speed: Literal["≤30 días", "31–60 días", "61–90 días", ">90 días"]
    model_used: str
    model_accuracy: float = Field(..., description="Accuracy del clasificador en test set")
