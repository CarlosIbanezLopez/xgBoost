from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier, XGBRegressor

from config import (
    CLASSIFIER_PATH,
    RANDOM_STATE,
    REGRESSOR_PATH,
)
from db import fetch_property_analytics


NUMERIC_FEATURES: List[str] = [
    "latitude",
    "longitude",
    "cluster_zona",
    "m2_construidos",
    "m2_terreno",
    "dormitorios",
    "banos",
    "estacionamientos",
    "antiguedad",
    "precio_publicacion",
    "precio_m2",
    "tiempo_en_mercado",
    "numero_reducciones",
    "diferencia_vs_promedio_zona",
    "ratio_activas_vendidas_zona",
    "mes_publicacion",
    "anio_publicacion",
]

CATEGORICAL_FEATURES: List[str] = [
    "tipo_propiedad",
    "subtipo_original",
    "categoria_propiedad",
    "estado_propiedad",
    "ciudad",
    "status",
    "transaction_type",
]


@dataclass
class TrainedModels:
    regressor: XGBRegressor
    classifier: XGBClassifier
    encoder: OrdinalEncoder
    reg_mae: float


def _build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepara X para regresión y clasificación.

    - Target regresión: precio_venta.
    - Target clasificación: segmento de precio (bajo/medio/alto) según terciles.
    """
    df = df.copy()

    # Filtrar filas con precio de venta válido
    df = df[df["precio_venta"].notna()]

    # Rellenar numéricas con 0 (simple y robusto para empezar)
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(0)
        else:
            df[col] = 0

    # Rellenar categóricas con "Desconocido"
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna("Desconocido")
        else:
            df[col] = "Desconocido"

    # Clasificación: segmentar el precio en 3 clases (bajo/medio/alto)
    y_reg = df["precio_venta"].astype(float)
    q1, q2 = y_reg.quantile([0.33, 0.66])

    def price_to_class(price: float) -> int:
        if price <= q1:
            return 0  # bajo
        if price <= q2:
            return 1  # medio
        return 2  # alto

    y_clf = y_reg.apply(price_to_class)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    return X, y_reg, y_clf


def train_models(test_size: float = 0.2) -> TrainedModels:
    """
    Entrena XGBRegressor y XGBClassifier sobre los datos de Postgres.
    """
    df = fetch_property_analytics()
    X_raw, y_reg, y_clf = _build_features(df)

    # Separar numéricas y categóricas
    X_num = X_raw[NUMERIC_FEATURES].astype(float)
    X_cat = X_raw[CATEGORICAL_FEATURES].astype(str)

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_cat_enc = encoder.fit_transform(X_cat)

    X_full = np.hstack([X_num.values, X_cat_enc])

    X_train, X_test, y_train, y_test = train_test_split(
        X_full,
        y_reg,
        test_size=test_size,
        random_state=RANDOM_STATE,
    )

    regressor = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)
    reg_mae = float(mean_absolute_error(y_test, y_pred))

    # Clasificador usando las mismas features
    classifier = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    classifier.fit(X_full, y_clf)

    return TrainedModels(
        regressor=regressor,
        classifier=classifier,
        encoder=encoder,
        reg_mae=reg_mae,
    )


def save_models(models: TrainedModels) -> None:
    """
    Persiste modelos y encoder a disco.
    """
    joblib.dump(
        {
            "regressor": models.regressor,
            "classifier": models.classifier,
            "encoder": models.encoder,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "reg_mae": models.reg_mae,
        },
        REGRESSOR_PATH,
    )

    # Guardamos también solo el clasificador para cargar más rápido si se quiere
    joblib.dump(
        {
            "classifier": models.classifier,
            "encoder": models.encoder,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "reg_mae": models.reg_mae,
        },
        CLASSIFIER_PATH,
    )


def load_regression_bundle():
    return joblib.load(REGRESSOR_PATH)


def load_classification_bundle():
    return joblib.load(CLASSIFIER_PATH)


if __name__ == "__main__":
    models = train_models()
    save_models(models)
    print(f"Entrenamiento completado. MAE regresión: {models.reg_mae:.2f}")

