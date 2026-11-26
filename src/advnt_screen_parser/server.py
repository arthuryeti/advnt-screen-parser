from __future__ import annotations

import hmac
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from .parser import ScreenParser

MAX_IMAGE_BYTES = 5_000_000
SUPPORTED_MEDIA_TYPES = {"image/jpeg", "image/png"}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    parser = await run_in_threadpool(ScreenParser)
    await run_in_threadpool(parser.warm_up)
    app.state.parser = parser
    app.state.ready = True
    yield
    app.state.ready = False


app = FastAPI(
    title="Advnt Screen Parser",
    version="3.0.0",
    lifespan=lifespan,
)


@app.get("/health/live")
async def live() -> dict[str, str]:
    return {"status": "live"}


@app.get("/health/ready")
async def ready(request: Request) -> dict[str, str]:
    if not getattr(request.app.state, "ready", False):
        raise HTTPException(status_code=503, detail={"code": "not_ready"})
    return {"status": "ready"}


@app.post("/v1/parse")
@app.post("/v1/ocr")
async def parse_screen(
    request: Request,
    authorization: str | None = Header(default=None),
) -> dict[str, object]:
    _authorize(authorization)
    content_type = request.headers.get("content-type", "").split(";", 1)[0].lower()
    if content_type not in SUPPORTED_MEDIA_TYPES:
        raise HTTPException(
            status_code=415,
            detail={"code": "unsupported_media_type"},
        )

    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail={"code": "image_too_large"})

    image_bytes = await request.body()
    if not image_bytes or len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail={"code": "image_too_large"})

    try:
        return await run_in_threadpool(request.app.state.parser.parse_bytes, image_bytes)
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail={"code": "invalid_image", "message": str(error)},
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail={"code": "parse_failed"},
        ) from error


def _authorize(authorization: str | None) -> None:
    token = os.environ.get("ADVNT_API_TOKEN") or os.environ.get("OCR_API_TOKEN") or ""
    if not token:
        return
    expected = f"Bearer {token}"
    if authorization is None or not hmac.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail={"code": "unauthorized"})
