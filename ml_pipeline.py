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

from config import RANDOM_STATE, VALID_COMBINATIONS, model_paths
from db import fetch_property_analytics


# ---------------------------------------------------------------------------
# Feature sets
# Removidos: cluster_zona, subtipo_original, categoria_propiedad,
#            mlsid, status, transaction_type
# ---------------------------------------------------------------------------

NUMERIC_FEATURES_VENTA: List[str] = [
    "latitude",
    "longitude",
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

NUMERIC_FEATURES_ALQUILER: List[str] = [
    "latitude",
    "longitude",
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

# Categóricas escalables — el encoder aprende los valores de los datos
# Agregar nuevas categorías en DB no requiere cambio de código
CATEGORICAL_FEATURES: List[str] = [
    "tipo_propiedad",    # "Casa", "Departamento", "Local Comercial", "Oficina", "Otro", "Terreno"
    "estado_propiedad",  # "Nuevo", "En buen estado", "En construcción", "Para renovar", ...
    "segmento",          # "Residencial", "Comercial"
    "ciudad",
    "pais",
]

# Features para el modelo de velocidad de venta
# (tiempo_en_mercado se excluye aquí porque es el target de ese modelo)
NUMERIC_FEATURES_SALE_PROB: List[str] = [
    "latitude",
    "longitude",
    "m2_construidos",
    "m2_terreno",
    "dormitorios",
    "banos",
    "estacionamientos",
    "antiguedad",
    "precio_publicacion",
    "precio_m2",
    "numero_reducciones",
    "diferencia_vs_promedio_zona",
    "ratio_activas_vendidas_zona",
    "mes_publicacion",
    "anio_publicacion",
]

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
    return "precio_alquiler_mes" if tipo_transaccion == "Alquiler" else "precio_venta"


def _sale_speed_label(dias: float) -> int:
    """
    Clasifica el tiempo en mercado en 4 clases:
      0 → vendida en ≤ 30 días
      1 → vendida en 31–60 días
      2 → vendida en 61–90 días
      3 → vendida en > 90 días (o sin vender en el período)
    """
    if dias <= 30:
        return 0
    if dias <= 60:
        return 1
    if dias <= 90:
        return 2
    return 3


def _fill_numeric(df: pd.DataFrame, numeric_features: List[str]) -> pd.DataFrame:
    """Rellena numéricas con 0 y calcula precio_m2 si falta."""
    df = df.copy()
    for col in numeric_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = 0.0

    if "precio_m2" in numeric_features and "precio_m2" in df.columns:
        mask = (df["precio_m2"] == 0) | df["precio_m2"].isna()
        area = df["m2_construidos"].where(df["m2_construidos"] > 0, df["m2_terreno"])
        area = area.where(area > 0, 1.0)
        pub = df.get("precio_publicacion", pd.Series(0.0, index=df.index)).fillna(0.0)
        df.loc[mask, "precio_m2"] = pub[mask] / area[mask]

    return df


def _fill_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Rellena categóricas con 'Desconocido'."""
    df = df.copy()
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna("Desconocido").astype(str)
        else:
            df[col] = "Desconocido"
    return df


def _build_encoder(df: pd.DataFrame) -> OrdinalEncoder:
    """
    Ajusta el OrdinalEncoder sobre los valores presentes en los datos.
    Escalable: si la BD agrega nuevas ciudades/tipos, el encoder las aprende
    en el próximo entrenamiento sin tocar código.
    handle_unknown='use_encoded_value' + unknown_value=-1 evita errores
    en inferencia ante valores no vistos en entrenamiento.
    """
    X_cat = df[CATEGORICAL_FEATURES].fillna("Desconocido").astype(str)
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    encoder.fit(X_cat)
    return encoder


def _compute_pct_bins(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    denom = np.clip(np.abs(y_true), 1e-6, None)
    abs_pct = np.abs(y_true - y_pred) / denom
    bins = np.unique(np.quantile(y_pred, [0.0, 0.25, 0.5, 0.75, 1.0]))
    if bins.shape[0] < 2:
        bins = np.array([y_pred.min(), y_pred.max()])
    idx = np.clip(np.digitize(y_pred, bins, right=True) - 1, 0, bins.shape[0] - 2)
    means = [
        float(abs_pct[idx == b].mean()) if np.any(idx == b) else float(abs_pct.mean())
        for b in range(bins.shape[0] - 1)
    ]
    return bins, np.array(means)


# ---------------------------------------------------------------------------
# ModelBundle
# ---------------------------------------------------------------------------

@dataclass
class ModelBundle:
    tipo_transaccion: str
    segmento: str
    model_idx_pub: int
    model_idx_no_pub: int

    regressor_pub: XGBRegressor = field(repr=False)
    regressor_no_pub: XGBRegressor = field(repr=False)
    classifier: XGBClassifier = field(repr=False)            # segmento de precio bajo/medio/alto
    sale_prob_classifier: XGBClassifier = field(repr=False)  # velocidad de venta 30/60/90/+90 días
    encoder: OrdinalEncoder = field(repr=False)

    numeric_features_pub: List[str] = field(repr=False)
    numeric_features_no_pub: List[str] = field(repr=False)
    numeric_features_sale_prob: List[str] = field(repr=False)

    reg_mae_pub: float = 0.0
    reg_mae_no_pub: float = 0.0
    clf_accuracy: float = 0.0
    clf_f1_weighted: float = 0.0
    sale_prob_accuracy: float = 0.0

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
    idx_pub, idx_no_pub = VALID_COMBINATIONS[(tipo_transaccion, segmento)]
    num_pub, num_no_pub = _NUMERIC_MAP[(tipo_transaccion, segmento)]
    target_col = _target_col(tipo_transaccion)

    print(f"\n[train] {tipo_transaccion}/{segmento} — cargando datos...", flush=True)
    df_raw = fetch_property_analytics(tipo_transaccion=tipo_transaccion, segmento=segmento)

    # Filtrar target válido
    df = df_raw[df_raw[target_col].notna() & (df_raw[target_col] > 0)].copy()
    if len(df) < 10:
        raise ValueError(
            f"Datos insuficientes para {tipo_transaccion}/{segmento}: {len(df)} filas."
        )
    print(f"[train] {tipo_transaccion}/{segmento} — {len(df)} filas válidas.", flush=True)

    # Preprocesar
    df = _fill_numeric(df, num_pub)   # num_pub es el superconjunto
    df = _fill_categorical(df)

    # Encoder escalable
    encoder = _build_encoder(df)
    X_cat_enc = encoder.transform(df[CATEGORICAL_FEATURES].astype(str))

    # Matrices de features
    X_num_pub = df[num_pub].astype(float).values
    X_num_nop = df[num_no_pub].astype(float).values

    # Para el modelo de velocidad de venta, excluimos tiempo_en_mercado
    num_sp = [c for c in NUMERIC_FEATURES_SALE_PROB if c in num_pub]
    X_num_sp  = df[num_sp].astype(float).values

    X_full_pub = np.hstack([X_num_pub, X_cat_enc])
    X_full_nop = np.hstack([X_num_nop, X_cat_enc])
    X_full_sp  = np.hstack([X_num_sp,  X_cat_enc])

    # Targets
    y_reg  = df[target_col].astype(float).values
    q1, q2 = np.quantile(y_reg, [0.33, 0.66])
    y_clf  = np.where(y_reg <= q1, 0, np.where(y_reg <= q2, 1, 2))
    y_sale = (
        df["tiempo_en_mercado"].fillna(999).apply(_sale_speed_label).values
        if "tiempo_en_mercado" in df.columns
        else np.full(len(df), 3)
    )

    # Split único para consistencia
    (
        X_tr_pub, X_te_pub,
        X_tr_nop, X_te_nop,
        X_tr_sp,  X_te_sp,
        y_reg_tr, y_reg_te,
        y_clf_tr, y_clf_te,
        y_sale_tr, y_sale_te,
    ) = train_test_split(
        X_full_pub, X_full_nop, X_full_sp,
        y_reg, y_clf, y_sale,
        test_size=test_size,
        random_state=RANDOM_STATE,
    )

    xgb_reg_params = dict(
        n_estimators=400, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=RANDOM_STATE, n_jobs=-1,
    )

    # Regressor con pub
    reg_pub = XGBRegressor(**xgb_reg_params)
    reg_pub.fit(X_tr_pub, y_reg_tr)
    y_pred_pub = reg_pub.predict(X_te_pub)
    mae_pub = float(mean_absolute_error(y_reg_te, y_pred_pub))
    bins_pub, pct_pub = _compute_pct_bins(y_reg_te, y_pred_pub)

    # Regressor sin pub
    reg_nop = XGBRegressor(**xgb_reg_params)
    reg_nop.fit(X_tr_nop, y_reg_tr)
    y_pred_nop = reg_nop.predict(X_te_nop)
    mae_nop = float(mean_absolute_error(y_reg_te, y_pred_nop))
    bins_nop, pct_nop = _compute_pct_bins(y_reg_te, y_pred_nop)

    # Classifier precio
    clf = XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8,
        objective="multi:softprob", num_class=3,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    clf.fit(X_tr_pub, y_clf_tr)
    clf_acc = float(accuracy_score(y_clf_te, clf.predict(X_te_pub)))
    clf_f1  = float(f1_score(y_clf_te, clf.predict(X_te_pub), average="weighted"))

    # Classifier velocidad de venta
    sale_clf = XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8,
        objective="multi:softprob", num_class=4,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    sale_clf.fit(X_tr_sp, y_sale_tr)
    sale_acc = float(accuracy_score(y_sale_te, sale_clf.predict(X_te_sp)))

    print(
        f"[train] {tipo_transaccion}/{segmento} done — "
        f"MAE_pub={mae_pub:.2f}  MAE_no_pub={mae_nop:.2f}  "
        f"clf_precio_acc={clf_acc:.3f}  clf_venta_acc={sale_acc:.3f}",
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
        sale_prob_classifier=sale_clf,
        encoder=encoder,
        numeric_features_pub=num_pub,
        numeric_features_no_pub=num_no_pub,
        numeric_features_sale_prob=num_sp,
        reg_mae_pub=mae_pub,
        reg_mae_no_pub=mae_nop,
        clf_accuracy=clf_acc,
        clf_f1_weighted=clf_f1,
        sale_prob_accuracy=sale_acc,
        price_bins_pub=bins_pub,
        mean_abs_pct_error_pub=pct_pub,
        price_bins_no_pub=bins_nop,
        mean_abs_pct_error_no_pub=pct_nop,
    )


def train_all_models(test_size: float = 0.2) -> Dict[tuple, ModelBundle]:
    bundles: Dict[tuple, ModelBundle] = {}
    for (tt, seg) in VALID_COMBINATIONS:
        try:
            bundles[(tt, seg)] = train_bundle(tt, seg, test_size=test_size)
        except ValueError as e:
            print(f"[train] WARN: {e} — saltando.", flush=True)
    return bundles


# ---------------------------------------------------------------------------
# Persistencia
# ---------------------------------------------------------------------------

def save_bundle(bundle: ModelBundle) -> None:
    paths_pub = model_paths(bundle.model_idx_pub)
    paths_nop = model_paths(bundle.model_idx_no_pub)

    # Encoder — incluye categorías conocidas para debugging/introspección
    enc_payload = {
        "encoder": bundle.encoder,
        "categorical_features": CATEGORICAL_FEATURES,
        "known_categories": {
            feat: list(cats)
            for feat, cats in zip(CATEGORICAL_FEATURES, bundle.encoder.categories_)
        },
    }
    joblib.dump(enc_payload, paths_pub["encoder"])
    joblib.dump(enc_payload, paths_nop["encoder"])

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

    joblib.dump(
        {
            "classifier": bundle.classifier,
            "sale_prob_classifier": bundle.sale_prob_classifier,
            "numeric_features_pub": bundle.numeric_features_pub,
            "numeric_features_sale_prob": bundle.numeric_features_sale_prob,
            "categorical_features": CATEGORICAL_FEATURES,
            "clf_accuracy": bundle.clf_accuracy,
            "clf_f1_weighted": bundle.clf_f1_weighted,
            "sale_prob_accuracy": bundle.sale_prob_accuracy,
            "tipo_transaccion": bundle.tipo_transaccion,
            "segmento": bundle.segmento,
        },
        paths_pub["classifier"],
    )

    print(
        f"[save] {bundle.tipo_transaccion}/{bundle.segmento} guardado "
        f"(m{bundle.model_idx_pub}/m{bundle.model_idx_no_pub}).",
        flush=True,
    )


def save_all_bundles(bundles: Dict[tuple, ModelBundle]) -> None:
    for b in bundles.values():
        save_bundle(b)


def load_bundle(tipo_transaccion: str, segmento: str) -> dict:
    from config import get_model_indices

    idx_pub, idx_no_pub = get_model_indices(tipo_transaccion, segmento)
    paths_pub = model_paths(idx_pub)
    paths_nop = model_paths(idx_no_pub)

    reg_pub_b = joblib.load(paths_pub["regressor"])
    reg_nop_b = joblib.load(paths_nop["regressor"])
    clf_b     = joblib.load(paths_pub["classifier"])
    enc_b     = joblib.load(paths_pub["encoder"])

    return {
        "regressor_pub":              reg_pub_b["regressor"],
        "regressor_no_pub":           reg_nop_b["regressor"],
        "classifier":                 clf_b["classifier"],
        "sale_prob_classifier":       clf_b["sale_prob_classifier"],
        "encoder":                    enc_b["encoder"],
        "known_categories":           enc_b.get("known_categories", {}),
        "numeric_features_pub":       reg_pub_b["numeric_features"],
        "numeric_features_no_pub":    reg_nop_b["numeric_features"],
        "numeric_features_sale_prob": clf_b.get("numeric_features_sale_prob", NUMERIC_FEATURES_SALE_PROB),
        "categorical_features":       CATEGORICAL_FEATURES,
        "reg_mae_pub":                reg_pub_b["reg_mae"],
        "reg_mae_no_pub":             reg_nop_b["reg_mae"],
        "price_bins_pub":             np.array(reg_pub_b["price_bins"]),
        "mean_abs_pct_error_pub":     np.array(reg_pub_b["mean_abs_pct_error_by_bin"]),
        "price_bins_no_pub":          np.array(reg_nop_b["price_bins"]),
        "mean_abs_pct_error_no_pub":  np.array(reg_nop_b["mean_abs_pct_error_by_bin"]),
        "clf_accuracy":               clf_b["clf_accuracy"],
        "clf_f1_weighted":            clf_b["clf_f1_weighted"],
        "sale_prob_accuracy":         clf_b.get("sale_prob_accuracy", 0.0),
    }


# ---------------------------------------------------------------------------
# Entrypoint CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    bundles = train_all_models()
    save_all_bundles(bundles)

    print("\n=== Resumen ===")
    for (tt, seg), b in bundles.items():
        print(
            f"  {tt}/{seg}: "
            f"MAE_pub={b.reg_mae_pub:.2f}  MAE_no_pub={b.reg_mae_no_pub:.2f}  "
            f"clf_precio_acc={b.clf_accuracy:.3f}  f1={b.clf_f1_weighted:.3f}  "
            f"clf_venta_acc={b.sale_prob_accuracy:.3f}"
        )