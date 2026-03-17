from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier, XGBRegressor

from config import (
    CLASSIFIER_PATH,
    ENCODER_PATH,
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
    "pais",
    "status",
    "transaction_type",
]


@dataclass
class TrainedModels:
    regressor: XGBRegressor
    classifier: XGBClassifier
    encoder: OrdinalEncoder
    reg_mae: float
    clf_accuracy: float
    clf_f1_weighted: float
    # métricas de error relativo por rango de precio
    price_bins: np.ndarray
    mean_abs_pct_error_by_bin: np.ndarray


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

    # Si precio_m2 no viene o es 0, lo calculamos como precio_publicacion / m2_construidos o m2_terreno
    if "precio_m2" in df.columns:
        mask_needs_calc = (df["precio_m2"] == 0) | df["precio_m2"].isna()
        area = df["m2_construidos"].where(df["m2_construidos"] > 0, df["m2_terreno"])
        area = area.where(area > 0, 1)  # evitar división por 0
        df.loc[mask_needs_calc, "precio_m2"] = (
            df.loc[mask_needs_calc, "precio_publicacion"] / area[mask_needs_calc]
        )

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

    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X_full,
        y_reg,
        y_clf,
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
    regressor.fit(X_train, y_reg_train)

    y_reg_pred = regressor.predict(X_test)
    reg_mae = float(mean_absolute_error(y_reg_test, y_reg_pred))

    # Error porcentual absoluto por muestra en test
    denom = np.clip(np.abs(y_reg_test.values), 1e-6, None)
    abs_pct_error = np.abs(y_reg_test.values - y_reg_pred) / denom

    # Bins de precio según el valor predicho
    # Usamos cuantiles para adaptarnos a la distribución de precios
    price_bins = np.quantile(y_reg_pred, [0.0, 0.25, 0.5, 0.75, 1.0])
    # Asegurar que los bins sean estrictamente crecientes para evitar problemas numéricos
    price_bins = np.unique(price_bins)
    if price_bins.shape[0] < 2:
        # fallback: un solo bin que cubre todo
        price_bins = np.array([y_reg_pred.min(), y_reg_pred.max()])

    bin_indices = np.digitize(y_reg_pred, price_bins, right=True) - 1
    bin_indices = np.clip(bin_indices, 0, price_bins.shape[0] - 2)

    mean_abs_pct_error_by_bin = []
    for b in range(price_bins.shape[0] - 1):
        mask = bin_indices == b
        if not np.any(mask):
            mean_abs_pct_error_by_bin.append(float(abs_pct_error.mean()))
        else:
            mean_abs_pct_error_by_bin.append(float(abs_pct_error[mask].mean()))
    mean_abs_pct_error_by_bin = np.array(mean_abs_pct_error_by_bin)

    # Clasificador usando las mismas features (sin data leakage)
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
    classifier.fit(X_train, y_clf_train)

    y_clf_pred = classifier.predict(X_test)
    clf_accuracy = float(accuracy_score(y_clf_test, y_clf_pred))
    clf_f1_weighted = float(f1_score(y_clf_test, y_clf_pred, average="weighted"))

    return TrainedModels(
        regressor=regressor,
        classifier=classifier,
        encoder=encoder,
        reg_mae=reg_mae,
        clf_accuracy=clf_accuracy,
        clf_f1_weighted=clf_f1_weighted,
        price_bins=price_bins,
        mean_abs_pct_error_by_bin=mean_abs_pct_error_by_bin,
    )


def save_models(models: TrainedModels) -> None:
    """
    Persiste modelos y encoder a disco.
    """
    # Guardar encoder una sola vez
    joblib.dump(
        {
            "encoder": models.encoder,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
        },
        ENCODER_PATH,
    )

    # Bundle de regresión
    joblib.dump(
        {
            "regressor": models.regressor,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "reg_mae": models.reg_mae,
            "clf_accuracy": models.clf_accuracy,
            "clf_f1_weighted": models.clf_f1_weighted,
            "price_bins": models.price_bins,
            "mean_abs_pct_error_by_bin": models.mean_abs_pct_error_by_bin,
        },
        REGRESSOR_PATH,
    )

    # Bundle de clasificación
    joblib.dump(
        {
            "classifier": models.classifier,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "reg_mae": models.reg_mae,
            "clf_accuracy": models.clf_accuracy,
            "clf_f1_weighted": models.clf_f1_weighted,
        },
        CLASSIFIER_PATH,
    )


def load_regression_bundle():
    bundle = joblib.load(REGRESSOR_PATH)
    encoder_bundle = joblib.load(ENCODER_PATH)
    bundle["encoder"] = encoder_bundle["encoder"]
    return bundle


def load_classification_bundle():
    bundle = joblib.load(CLASSIFIER_PATH)
    encoder_bundle = joblib.load(ENCODER_PATH)
    bundle["encoder"] = encoder_bundle["encoder"]
    return bundle


if __name__ == "__main__":
    models = train_models()
    save_models(models)
    print(f"Entrenamiento completado. MAE regresión: {models.reg_mae:.2f}")
    print(
        f"Métricas clasificador - accuracy: {models.clf_accuracy:.3f}, "
        f"F1 (weighted): {models.clf_f1_weighted:.3f}"
    )

