from unittest.mock import patch

import numpy as np

from app.api.schemas import PredictRequest
from app.services import prediction_service


class FakeRegressor:
    def __init__(self, prediction):
        self.prediction = prediction

    def predict(self, _features):
        return np.array([self.prediction])


class FakeClassifier:
    def __init__(self, probabilities):
        self.probabilities = probabilities

    def predict_proba(self, _features):
        return np.array([self.probabilities])


def make_request(**overrides):
    values = {
        "tipo_transaccion": "Venta",
        "segmento": "Residencial",
        "latitude": -17.39,
        "longitude": -66.16,
        "m2_construidos": 100,
        "tipo_propiedad": "Casa",
        "ciudad": "Cochabamba",
        "pais": "Bolivia",
    }
    values.update(overrides)
    return PredictRequest(**values)


def make_bundle():
    return {
        "regressor_pub": FakeRegressor(120_000),
        "regressor_no_pub": FakeRegressor(100_000),
        "classifier": FakeClassifier([0.1, 0.7, 0.2]),
        "sale_prob_classifier": FakeClassifier([0.2, 0.3, 0.1, 0.4]),
        "encoder": None,
        "numeric_features_pub": ["precio_publicacion", "m2_construidos"],
        "numeric_features_no_pub": ["m2_construidos"],
        "numeric_features_classifier_pub": ["precio_publicacion", "m2_construidos"],
        "numeric_features_sale_prob": ["precio_publicacion", "m2_construidos"],
        "categorical_features": [],
        "reg_mae_pub": 5_000,
        "reg_mae_no_pub": 8_000,
        "price_bins_pub": np.array([0, 200_000]),
        "mean_abs_pct_error_pub": np.array([0.1]),
        "price_bins_no_pub": np.array([0, 200_000]),
        "mean_abs_pct_error_no_pub": np.array([0.2]),
        "sale_prob_accuracy": 0.8,
    }


def test_regression_without_publication_price_uses_no_pub_model():
    request = make_request()

    with (
        patch.object(prediction_service, "get_bundle", return_value=make_bundle()),
        patch.object(prediction_service, "fetch_location_market_stats", return_value=None),
        patch.object(prediction_service, "fetch_comparable_listings", return_value=[]),
    ):
        result = prediction_service.predict_regression(request)

    assert result.predicted_price == 100_000
    assert result.expected_abs_error == 8_000
    assert result.expected_pct_error == 0.2
    assert result.model_used == "m2_Venta_Residencial_no_pub"
    assert result.interval_approx == {"lower": 80_000, "upper": 120_000}


def test_price_classification_preserves_labels_and_probabilities():
    with patch.object(prediction_service, "get_bundle", return_value=make_bundle()):
        result = prediction_service.predict_classification(
            make_request(precio_publicacion=120_000)
        )

    assert result.price_segment == "medio"
    assert result.probabilities == {"bajo": 0.1, "medio": 0.7, "alto": 0.2}
    assert result.model_used == "m1_Venta_Residencial_clf_precio"


def test_sale_probability_preserves_accumulated_horizons():
    with patch.object(prediction_service, "get_bundle", return_value=make_bundle()):
        result = prediction_service.predict_sale_probability(
            make_request(precio_publicacion=120_000)
        )

    assert result.prob_30_days == 0.2
    assert result.prob_60_days == 0.5
    assert result.prob_90_days == 0.6
    assert result.prob_over_90_days == 0.4
    assert result.expected_speed == ">90 días"
    assert result.model_accuracy == 0.8
