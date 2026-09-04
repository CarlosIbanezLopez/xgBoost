"""FastAPI application factory."""

from fastapi import FastAPI

from app.api.routes.predictions import router as predictions_router
from app.api.routes.system import router as system_router
from app.api.routes.training import router as training_router


def create_app() -> FastAPI:
    application = FastAPI(title="XGBoost Property Service", version="2.1.0")
    application.include_router(training_router)
    application.include_router(predictions_router)
    application.include_router(system_router)
    return application


app = create_app()
