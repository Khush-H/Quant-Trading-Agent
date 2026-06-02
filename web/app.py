"""FastAPI dashboard.

Read-only monitoring surface: current mode, positions, recent orders, equity.
Crucially, the dashboard NEVER places orders — it only displays state. Any
control surface added later must still route through
:func:`core.engine.submit_order`; the web layer gets no special path.

Run with the ``scripts/run_paper.py`` helper or directly:
    uvicorn web.app:app --host $WEB_HOST --port $WEB_PORT
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from config import get_settings

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

app = FastAPI(title="Quant Trading Dashboard")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@app.get("/health")
def health() -> dict:
    """Liveness probe with the active mode (handy for confirming you're safe)."""
    settings = get_settings()
    return {"status": "ok", "mode": settings.mode.value}


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    """Dashboard home. Displays mode banner; data wired up during the build."""
    settings = get_settings()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "mode": settings.mode.value,
            "is_live": settings.is_live,
        },
    )
