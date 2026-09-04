from fastapi.testclient import TestClient

from app.application import app


EXPECTED_OPERATIONS = {
    ("POST", "/train"): "train_endpoint_train_post",
    ("POST", "/train/terrain"): "train_terrain_endpoint_train_terrain_post",
    ("POST", "/predict/regression"): "predict_regression_predict_regression_post",
    ("POST", "/predict/classification"): "predict_classification_predict_classification_post",
    ("POST", "/predict/sale-probability"): "predict_sale_probability_predict_sale_probability_post",
    ("GET", "/models"): "list_models_models_get",
    ("GET", "/health"): "health_health_get",
}


def test_public_http_contract_is_preserved():
    schema = app.openapi()

    assert app.title == "XGBoost Property Service"
    assert app.version == "2.1.0"
    for (method, path), operation_id in EXPECTED_OPERATIONS.items():
        assert path in schema["paths"]
        assert schema["paths"][path][method.lower()]["operationId"] == operation_id


def test_health_and_models_endpoints_do_not_need_external_services():
    client = TestClient(app)

    health = client.get("/health")
    models = client.get("/models")

    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "models_cached": 0,
        "terrain_models_cached": 0,
    }
    assert models.status_code == 200
    assert len(models.json()["combinations"]) == 4
    assert len(models.json()["terrain_regressors"]) == 2
