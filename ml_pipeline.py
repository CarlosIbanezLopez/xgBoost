from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier, XGBRegressor

from config import (
    RANDOM_STATE,
    VALID_COMBINATIONS,
    model_paths,
)
from db import fetch_property_analytics


# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------

NUMERIC_FEATURES_VENTA: List[str] = [
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

NUMERIC_FEATURES_VENTA_NO_PUB: List[str] = [
    c for c in NUMERIC_FEATURES_VENTA if c != "precio_publicacion"
]

# Para alquiler se usa precio_alquiler_mes en lugar de precio_publicacion
NUMERIC_FEATURES_ALQUILER: List[str] = [
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
    "precio_alquiler_mes",
    "precio_m2",
    "tiempo_en_mercado",
    "numero_reducciones",
    "diferencia_vs_promedio_zona",
    "ratio_activas_vendidas_zona",
    "mes_publicacion",
    "anio_publicacion",
]

NUMERIC_FEATURES_ALQUILER_NO_PUB: List[str] = [
    c for c in NUMERIC_FEATURES_ALQUILER
    if c not in ("precio_publicacion", "precio_alquiler_mes")
]

CATEGORICAL_FEATURES: List[str] = [
    "tipo_propiedad",
    "subtipo_original",
    "categoria_propiedad",
    "estado_propiedad",
    "segmento",
    "ciudad",
    "pais",
    "status",
    "transaction_type",
]

# Mapeo de combinacion → features numéricas (con pub, sin pub)
_NUMERIC_MAP: dict[tuple[str, str], tuple[list[str], list[str]]] = {
    ("Venta",    "Residencial"): (NUMERIC_FEATURES_VENTA,    NUMERIC_FEATURES_VENTA_NO_PUB),
    ("Venta",    "Comercial"):   (NUMERIC_FEATURES_VENTA,    NUMERIC_FEATURES_VENTA_NO_PUB),
    ("Alquiler", "Residencial"): (NUMERIC_FEATURES_ALQUILER, NUMERIC_FEATURES_ALQUILER_NO_PUB),
    ("Alquiler", "Comercial"):   (NUMERIC_FEATURES_ALQUILER, NUMERIC_FEATURES_ALQUILER_NO_PUB),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _target_col(tipo_transaccion: str) -> str:
    """Columna objetivo según tipo de transacción."""
    return "precio_alquiler_mes" if tipo_transaccion == "Alquiler" else "precio_venta"


def _build_features(
    df: pd.DataFrame,
    numeric_features: List[str],
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepara X numérico y categórico. El target debe venir en df['_target'].
    """
    df = df.copy()

    for col in numeric_features:
        if col in df.columns:
            df[col] = df[col].fillna(0)
        else:
            df[col] = 0.0

    # Calcular precio_m2 si falta
    if "precio_m2" in numeric_features and "precio_m2" in df.columns:
        mask = (df["precio_m2"] == 0) | df["precio_m2"].isna()
        area = df["m2_construidos"].where(df["m2_construidos"] > 0, df["m2_terreno"])
        area = area.where(area > 0, 1)
        pub = df.get("precio_publicacion", pd.Series(0, index=df.index)).fillna(0)
        df.loc[mask, "precio_m2"] = pub[mask] / area[mask]

    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna("Desconocido")
        else:
            df[col] = "Desconocido"

    X = df[numeric_features + CATEGORICAL_FEATURES]
    return X


def _compute_pct_error_bins(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calcula bins de precio y error porcentual medio por bin."""
    denom = np.clip(np.abs(y_true), 1e-6, None)
    abs_pct = np.abs(y_true - y_pred) / denom

    price_bins = np.quantile(y_pred, [0.0, 0.25, 0.5, 0.75, 1.0])
    price_bins = np.unique(price_bins)
    if price_bins.shape[0] < 2:
        price_bins = np.array([y_pred.min(), y_pred.max()])

    bin_idx = np.clip(np.digitize(y_pred, price_bins, right=True) - 1, 0, price_bins.shape[0] - 2)
    means = []
    for b in range(price_bins.shape[0] - 1):
        mask = bin_idx == b
        means.append(float(abs_pct[mask].mean()) if np.any(mask) else float(abs_pct.mean()))

    return price_bins, np.array(means)


# ---------------------------------------------------------------------------
# Bundle dataclass
# ---------------------------------------------------------------------------

@dataclass
class ModelBundle:
    """Un par (regressor_con_pub, regressor_sin_pub) + classifier + encoder."""
    tipo_transaccion: str
    segmento: str
    model_idx_pub: int       # índice impar  (con precio_publicacion)
    model_idx_no_pub: int    # índice par    (sin precio_publicacion)

    regressor_pub: XGBRegressor = field(repr=False)
    regressor_no_pub: XGBRegressor = field(repr=False)
    classifier: XGBClassifier = field(repr=False)
    encoder: OrdinalEncoder = field(repr=False)

    numeric_features_pub: List[str] = field(repr=False)
    numeric_features_no_pub: List[str] = field(repr=False)

    reg_mae_pub: float = 0.0
    reg_mae_no_pub: float = 0.0
    clf_accuracy: float = 0.0
    clf_f1_weighted: float = 0.0

    price_bins_pub: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    mean_abs_pct_error_pub: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    price_bins_no_pub: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    mean_abs_pct_error_no_pub: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------

def train_bundle(
    tipo_transaccion: str,
    segmento: str,
    test_size: float = 0.2,
) -> ModelBundle:
    """
    Entrena los dos regressors y el classifier para una combinación específica.
    """
    idx_pub, idx_no_pub = VALID_COMBINATIONS[(tipo_transaccion, segmento)]
    num_pub, num_no_pub = _NUMERIC_MAP[(tipo_transaccion, segmento)]
    target_col = _target_col(tipo_transaccion)

    print(f"\n[train] {tipo_transaccion}/{segmento} — cargando datos...", flush=True)
    df = fetch_property_analytics(
        tipo_transaccion=tipo_transaccion,
        segmento=segmento,
    )

    # Filtrar filas con target válido
    df = df[df[target_col].notna() & (df[target_col] > 0)].copy()
    df["_target"] = df[target_col].astype(float)

    if len(df) < 10:
        raise ValueError(
            f"Datos insuficientes para {tipo_transaccion}/{segmento}: {len(df)} filas."
        )

    print(f"[train] {tipo_transaccion}/{segmento} — {len(df)} filas válidas.", flush=True)

    # Segmentación para classifier
    y_reg = df["_target"]
    q1, q2 = y_reg.quantile([0.33, 0.66])
    y_clf = y_reg.apply(lambda p: 0 if p <= q1 else (1 if p <= q2 else 2))

    # Preparar features
    X_pub  = _build_features(df, num_pub)
    X_no_p = _build_features(df, num_no_pub)

    X_num_pub  = X_pub[num_pub].astype(float)
    X_num_nop  = X_no_p[num_no_pub].astype(float)
    X_cat      = X_pub[CATEGORICAL_FEATURES].astype(str)

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_cat_enc = encoder.fit_transform(X_cat)

    X_full_pub = np.hstack([X_num_pub.values, X_cat_enc])
    X_full_nop = np.hstack([X_num_nop.values, X_cat_enc])

    (
        X_tr_pub, X_te_pub,
        X_tr_nop, X_te_nop,
        y_reg_tr, y_reg_te,
        y_clf_tr, y_clf_te,
    ) = train_test_split(
        X_full_pub, X_full_nop, y_reg, y_clf,
        test_size=test_size,
        random_state=RANDOM_STATE,
    )

    # --- Regressor con precio_publicacion ---
    reg_pub = XGBRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    reg_pub.fit(X_tr_pub, y_reg_tr)
    y_pred_pub = reg_pub.predict(X_te_pub)
    mae_pub = float(mean_absolute_error(y_reg_te, y_pred_pub))
    bins_pub, pct_pub = _compute_pct_error_bins(y_reg_te.values, y_pred_pub)

    # --- Regressor sin precio_publicacion ---
    reg_nop = XGBRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    reg_nop.fit(X_tr_nop, y_reg_tr)
    y_pred_nop = reg_nop.predict(X_te_nop)
    mae_nop = float(mean_absolute_error(y_reg_te, y_pred_nop))
    bins_nop, pct_nop = _compute_pct_error_bins(y_reg_te.values, y_pred_nop)

    # --- Classifier ---
    clf = XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8,
        objective="multi:softprob", num_class=3,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    clf.fit(X_tr_pub, y_clf_tr)
    y_clf_pred = clf.predict(X_te_pub)
    clf_acc = float(accuracy_score(y_clf_te, y_clf_pred))
    clf_f1  = float(f1_score(y_clf_te, y_clf_pred, average="weighted"))

    print(
        f"[train] {tipo_transaccion}/{segmento} — "
        f"MAE_pub={mae_pub:.2f}  MAE_no_pub={mae_nop:.2f}  "
        f"clf_acc={clf_acc:.3f}  clf_f1={clf_f1:.3f}",
        flush=True,
    )

    return ModelBundle(
        tipo_transaccion=tipo_transaccion,
        segmento=segmento,
        model_idx_pub=idx_pub,
        model_idx_no_pub=idx_no_pub,
        regressor_pub=reg_pub,
        regressor_no_pub=reg_nop,
        classifier=clf,
        encoder=encoder,
        numeric_features_pub=num_pub,
        numeric_features_no_pub=num_no_pub,
        reg_mae_pub=mae_pub,
        reg_mae_no_pub=mae_nop,
        clf_accuracy=clf_acc,
        clf_f1_weighted=clf_f1,
        price_bins_pub=bins_pub,
        mean_abs_pct_error_pub=pct_pub,
        price_bins_no_pub=bins_nop,
        mean_abs_pct_error_no_pub=pct_nop,
    )


def train_all_models(test_size: float = 0.2) -> Dict[tuple, ModelBundle]:
    """Entrena los 4 bundles (8 modelos reales) para todas las combinaciones."""
    bundles: Dict[tuple, ModelBundle] = {}
    for (tipo_transaccion, segmento) in VALID_COMBINATIONS:
        try:
            bundle = train_bundle(tipo_transaccion, segmento, test_size=test_size)
            bundles[(tipo_transaccion, segmento)] = bundle
        except ValueError as e:
            print(f"[train] WARN: {e} — saltando combinación.", flush=True)
    return bundles


# ---------------------------------------------------------------------------
# Persistencia
# ---------------------------------------------------------------------------

def save_bundle(bundle: ModelBundle) -> None:
    """Guarda los artefactos de un bundle en disco."""
    paths_pub  = model_paths(bundle.model_idx_pub)
    paths_nop  = model_paths(bundle.model_idx_no_pub)

    # Encoder compartido (guardado en ambos paths por conveniencia)
    encoder_payload = {
        "encoder": bundle.encoder,
        "categorical_features": CATEGORICAL_FEATURES,
    }
    joblib.dump(encoder_payload, paths_pub["encoder"])
    joblib.dump(encoder_payload, paths_nop["encoder"])

    # Regressor con pub
    joblib.dump(
        {
            "regressor": bundle.regressor_pub,
            "numeric_features": bundle.numeric_features_pub,
            "categorical_features": CATEGORICAL_FEATURES,
            "reg_mae": bundle.reg_mae_pub,
            "price_bins": bundle.price_bins_pub,
            "mean_abs_pct_error_by_bin": bundle.mean_abs_pct_error_pub,
            "tipo_transaccion": bundle.tipo_transaccion,
            "segmento": bundle.segmento,
        },
        paths_pub["regressor"],
    )

    # Regressor sin pub
    joblib.dump(
        {
            "regressor": bundle.regressor_no_pub,
            "numeric_features": bundle.numeric_features_no_pub,
            "categorical_features": CATEGORICAL_FEATURES,
            "reg_mae": bundle.reg_mae_no_pub,
            "price_bins": bundle.price_bins_no_pub,
            "mean_abs_pct_error_by_bin": bundle.mean_abs_pct_error_no_pub,
            "tipo_transaccion": bundle.tipo_transaccion,
            "segmento": bundle.segmento,
        },
        paths_nop["regressor"],
    )

    # Classifier (asociado al modelo con pub por convención)
    joblib.dump(
        {
            "classifier": bundle.classifier,
            "numeric_features": bundle.numeric_features_pub,
            "categorical_features": CATEGORICAL_FEATURES,
            "clf_accuracy": bundle.clf_accuracy,
            "clf_f1_weighted": bundle.clf_f1_weighted,
            "tipo_transaccion": bundle.tipo_transaccion,
            "segmento": bundle.segmento,
        },
        paths_pub["classifier"],
    )

    print(
        f"[save] Bundle {bundle.tipo_transaccion}/{bundle.segmento} "
        f"guardado (modelos {bundle.model_idx_pub}/{bundle.model_idx_no_pub}).",
        flush=True,
    )


def save_all_bundles(bundles: Dict[tuple, ModelBundle]) -> None:
    for bundle in bundles.values():
        save_bundle(bundle)


def load_bundle(tipo_transaccion: str, segmento: str) -> Dict:
    """
    Carga los artefactos para una combinación dada.
    Retorna un dict con claves:
        regressor_pub, regressor_no_pub, classifier, encoder,
        numeric_features_pub, numeric_features_no_pub, categorical_features,
        reg_mae_pub, reg_mae_no_pub, price_bins_pub, mean_abs_pct_error_pub,
        price_bins_no_pub, mean_abs_pct_error_no_pub
    """
    from config import get_model_indices  # importación local para evitar ciclos

    idx_pub, idx_no_pub = get_model_indices(tipo_transaccion, segmento)
    paths_pub = model_paths(idx_pub)
    paths_nop = model_paths(idx_no_pub)

    reg_pub_bundle = joblib.load(paths_pub["regressor"])
    reg_nop_bundle = joblib.load(paths_nop["regressor"])
    clf_bundle     = joblib.load(paths_pub["classifier"])
    enc_bundle     = joblib.load(paths_pub["encoder"])

    return {
        "regressor_pub":          reg_pub_bundle["regressor"],
        "regressor_no_pub":       reg_nop_bundle["regressor"],
        "classifier":             clf_bundle["classifier"],
        "encoder":                enc_bundle["encoder"],
        "numeric_features_pub":   reg_pub_bundle["numeric_features"],
        "numeric_features_no_pub": reg_nop_bundle["numeric_features"],
        "categorical_features":   CATEGORICAL_FEATURES,
        "reg_mae_pub":            reg_pub_bundle["reg_mae"],
        "reg_mae_no_pub":         reg_nop_bundle["reg_mae"],
        "price_bins_pub":         np.array(reg_pub_bundle["price_bins"]),
        "mean_abs_pct_error_pub": np.array(reg_pub_bundle["mean_abs_pct_error_by_bin"]),
        "price_bins_no_pub":      np.array(reg_nop_bundle["price_bins"]),
        "mean_abs_pct_error_no_pub": np.array(reg_nop_bundle["mean_abs_pct_error_by_bin"]),
        "clf_accuracy":           clf_bundle["clf_accuracy"],
        "clf_f1_weighted":        clf_bundle["clf_f1_weighted"],
    }


# ---------------------------------------------------------------------------
# Entrypoint CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    bundles = train_all_models()
    save_all_bundles(bundles)

    print("\n=== Resumen de entrenamiento ===")
    for (tt, seg), b in bundles.items():
        print(
            f"  {tt}/{seg}: "
            f"MAE_pub={b.reg_mae_pub:.2f}  MAE_no_pub={b.reg_mae_no_pub:.2f}  "
            f"acc={b.clf_accuracy:.3f}  f1={b.clf_f1_weighted:.3f}"
        )