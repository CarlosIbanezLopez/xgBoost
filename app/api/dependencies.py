"""Dependencies shared by API routes."""

import os

from fastapi import Depends, HTTPException
from fastapi.security import APIKeyHeader


API_KEY_TRAIN = os.getenv("API_KEY_TRAIN")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_train_key(api_key: str = Depends(api_key_header)) -> None:
    if not API_KEY_TRAIN:
        raise HTTPException(status_code=500, detail="Train API not configured")
    if api_key != API_KEY_TRAIN:
        raise HTTPException(status_code=401, detail="Invalid API key")
