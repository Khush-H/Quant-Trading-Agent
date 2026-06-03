"""FastAPI dashboard.

Read-only monitoring surface: current mode, position, recent orders, rolling
PnL/drawdown, the SYSTEM_HALT circuit-breaker state, and the last heartbeat.
Crucially, the dashboard NEVER places orders — it only displays state. Any
control surface added later must still route through
:func:`core.engine.submit_order`; the web layer gets no special path.

Read path: every data query goes through :class:`web.data.DashboardData`, which
opens the SQLite store in read-only mode and issues SELECTs only. There is no
INSERT/UPDATE/DELETE anywhere in a data request.

The ONE state-changing action is "reset halt": a POST that clears SYSTEM_HALT
through the SAME mechanism as ``scripts/reset_halt.py`` — it calls
:meth:`core.database.Database.clear_halt` behind the same guard the script
enforces (an explicit confirm, and only when actually halted). It does NOT
introduce a second way to clear the flag.

Rendering is plain Python (no template engine) so the dashboard's only runtime
deps are FastAPI/uvicorn (already pinned). Every dynamic value is HTML-escaped
before it reaches the page — notably the free-text ``reason`` on execution rows
and the halt reason.

Run with::

    uvicorn web.app:app --port 8000
"""

from __future__ import annotations

from urllib.parse import parse_qs

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from config import get_settings
from core.database import Database
from web.data import DashboardData
from web.render import render_index

app = FastAPI(title="Quant Trading Dashboard")


def get_reader() -> DashboardData:
    """Read-only data accessor. Overridable in tests."""
    return DashboardData(get_settings())


@app.get("/health")
def health() -> dict:
    """Liveness probe with the active mode (handy for confirming you're safe)."""
    settings = get_settings()
    return {"status": "ok", "mode": settings.mode.value}


@app.get("/api/state")
def api_state() -> JSONResponse:
    """Read-only JSON snapshot the page polls. Issues SELECT-only queries."""
    return JSONResponse(get_reader().snapshot())


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    """Dashboard home. Renders the current snapshot (read-only)."""
    snap = get_reader().snapshot()
    notice = request.query_params.get("reset")
    return HTMLResponse(render_index(snap, reset_notice=notice))


@app.post("/reset-halt")
async def reset_halt(request: Request) -> RedirectResponse:
    """Clear SYSTEM_HALT — the ONLY state-changing endpoint.

    Guarded exactly like ``scripts/reset_halt.py``:

    * a confirmation is REQUIRED (``confirm`` must be submitted truthy); without
      it nothing is cleared, mirroring the script's ``--confirm`` flag;
    * the flag is only cleared when it is actually set (the script's "nothing
      to do" guard).

    Clearing routes through :meth:`Database.clear_halt` and
    :meth:`Database.reset_exchange_failures` — the SAME calls the script makes.
    There is no second, unguarded path to clear the flag.

    ``confirm`` is read from the urlencoded body or the query string (parsed
    here directly, so no ``python-multipart`` dependency is needed).
    """
    confirm = await _read_confirm(request)
    if not _confirmed(confirm):
        # Guard not satisfied: do NOT clear. 303 back to the page (no write).
        return RedirectResponse(url="/?reset=missing_confirm", status_code=303)

    settings = get_settings()
    db = Database(settings)
    # Only clear when actually halted (matches the script's guard).
    if not db.is_halted():
        return RedirectResponse(url="/?reset=not_halted", status_code=303)

    # The SAME mechanism scripts/reset_halt.py uses to clear the breaker.
    db.clear_halt()
    db.reset_exchange_failures()
    return RedirectResponse(url="/?reset=cleared", status_code=303)


async def _read_confirm(request: Request) -> str:
    """Pull the ``confirm`` field from the urlencoded body, then the query string.

    Parsed by hand to avoid a ``python-multipart`` dependency; the reset form
    submits ``application/x-www-form-urlencoded``.
    """
    body = (await request.body()).decode("utf-8", "ignore")
    if body:
        parsed = parse_qs(body)
        if "confirm" in parsed and parsed["confirm"]:
            return parsed["confirm"][0]
    return request.query_params.get("confirm", "")


def _confirmed(value: str) -> bool:
    """The reset guard: an explicit truthy confirmation, like ``--confirm``."""
    return str(value).strip().lower() in {"1", "true", "yes", "on", "confirm"}
