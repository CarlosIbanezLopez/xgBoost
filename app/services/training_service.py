"""Model training use cases."""

from app.api.schemas import TrainResponse
from app.ml.pipeline import (
    save_all_bundles,
    save_all_terrain_bundles,
    train_all_models,
    train_all_terrain_models,
)
from app.services.model_registry import invalidate_cache


def train_models() -> TrainResponse:
    bundles = train_all_models()
    save_all_bundles(bundles)
    terrain_bundles = train_all_terrain_models()
    save_all_terrain_bundles(terrain_bundles)
    invalidate_cache()

    results = [
        {
            "tipo_transaccion": bundle.tipo_transaccion,
            "segmento": bundle.segmento,
            "mae_pub": round(bundle.reg_mae_pub, 2),
            "mae_no_pub": round(bundle.reg_mae_no_pub, 2),
            "clf_precio_acc": round(bundle.clf_accuracy, 4),
            "clf_precio_f1": round(bundle.clf_f1_weighted, 4),
            "clf_venta_acc": round(bundle.sale_prob_accuracy, 4),
        }
        for bundle in bundles.values()
    ]
    results.extend(
        {
            "tipo_transaccion": bundle.tipo_transaccion,
            "segmento": "Terreno",
            "mae_pub": round(bundle.reg_mae_pub, 2),
            "mae_no_pub": round(bundle.reg_mae_no_pub, 2),
            "sample_count": bundle.sample_count,
        }
        for bundle in terrain_bundles.values()
    )
    return TrainResponse(results=results)


def train_terrain_models() -> TrainResponse:
    bundles = train_all_terrain_models()
    save_all_terrain_bundles(bundles)
    invalidate_cache()

    return TrainResponse(
        results=[
            {
                "tipo_transaccion": bundle.tipo_transaccion,
                "segmento": "Terreno",
                "mae_pub": round(bundle.reg_mae_pub, 2),
                "mae_no_pub": round(bundle.reg_mae_no_pub, 2),
                "sample_count": bundle.sample_count,
            }
            for bundle in bundles.values()
        ]
    )
