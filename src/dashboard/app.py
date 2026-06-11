"""FastAPI application factory for the SpeechEdge dashboard."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse

from src.dashboard.state import build_state
from src.dashboard.store import StateStore

_STATIC_DIR = Path(__file__).parent / "static"


def create_app(store: StateStore) -> FastAPI:
    """Create and return a configured FastAPI application.

    Parameters
    ----------
    store:
        Any object implementing ``snapshot() -> (transcript, tone, markets, fills, resolutions)``.
    """
    app = FastAPI(title="SpeechEdge Dashboard")

    @app.get("/api/state")
    async def api_state() -> JSONResponse:
        transcript, tone, markets, fills, resolutions = store.snapshot()
        state = build_state(transcript, tone, markets, fills, resolutions)
        return JSONResponse(content=state.model_dump())

    @app.get("/")
    async def root() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html", media_type="text/html")

    return app
