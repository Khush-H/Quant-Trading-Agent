"""Read-only data access for the dashboard.

The dashboard NEVER writes. Every query it issues is a ``SELECT`` against a
connection opened in SQLite read-only mode (``?mode=ro`` URI). That is
defence-in-depth: even if a query were mistakenly an INSERT/UPDATE/DELETE, the
connection itself rejects it (``attempt to write a readonly database``). The one
state-changing action the dashboard exposes — clearing SYSTEM_HALT — does NOT
go through here; it routes through :class:`core.database.Database` exactly like
``scripts/reset_halt.py`` (see :mod:`web.app`).

This module reuses the same ``system_state`` keys and ``execution_logs`` /
``positions`` schema the rest of the system writes, but only ever reads them.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Optional

from config import Settings, get_settings
from core.database import Database
from core.risk import rolling_drawdown_pct

# Same keys the writer (core.database / core.risk) uses. Centralised on Database
# already; aliased here so the read-only layer reads the exact same flags.
_HALT_KEY = Database.HALT_KEY
_HALT_REASON_KEY = Database.HALT_REASON_KEY
_HEARTBEAT_KEY = Database.HEARTBEAT_KEY

_SQLITE_PREFIX = "sqlite:///"
_DAY_MS = 24 * 3_600_000


class DashboardData:
    """Read-only view over the trading SQLite store.

    Construct once per process (cheap; opens a fresh short-lived connection per
    call). Every public method issues SELECT-only SQL on a ``mode=ro``
    connection, so the dashboard cannot mutate state through this object.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.url = self.settings.database_url
        self.path = self._resolve_path(self.url)

    @staticmethod
    def _resolve_path(url: str) -> str:
        if not url.startswith(_SQLITE_PREFIX):
            raise ValueError(
                f"Dashboard supports only sqlite URLs, got {url!r}."
            )
        return url[len(_SQLITE_PREFIX):]

    # --- connection (READ-ONLY) --------------------------------------------
    def _connect_ro(self) -> sqlite3.Connection:
        """Open a strictly read-only connection.

        Uses SQLite's URI ``mode=ro`` so the engine itself forbids writes. WAL
        readers don't block the ingest/daemon writer. ``:memory:`` is passed
        through verbatim for tests that share an in-memory DB via a URI.
        """
        if self.path == ":memory:":
            conn = sqlite3.connect(self.path)
        else:
            db_uri = f"file:{Path(self.path).as_posix()}?mode=ro"
            conn = sqlite3.connect(db_uri, uri=True)
        conn.row_factory = sqlite3.Row
        # busy_timeout so a concurrent writer's lock doesn't instantly error a
        # dashboard read. No journal_mode change here — read-only can't set it.
        conn.execute("PRAGMA busy_timeout=5000;")
        return conn

    def _scalar_state(self, conn: sqlite3.Connection, key: str) -> Optional[str]:
        row = conn.execute(
            "SELECT value FROM system_state WHERE key = ?;", (key,)
        ).fetchone()
        return row["value"] if row is not None else None

    def _nav_history(self, conn: sqlite3.Connection) -> list[list]:
        import json

        v = self._scalar_state(conn, Database.NAV_HISTORY_KEY)
        if not v:
            return []
        try:
            return list(json.loads(v))
        except (ValueError, TypeError):
            return []

    # --- aggregate snapshot -------------------------------------------------
    def snapshot(self, *, now_ms: Optional[int] = None) -> dict[str, Any]:
        """Return everything the dashboard renders, in one read-only pass.

        Shape (all plain JSON-able types):
            mode, is_live, halted, halt_reason, last_heartbeat_ms,
            last_heartbeat_iso, positions[], executions[], pnl_24h_pct,
            drawdown_24h_pct.
        """
        import time

        now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
        with self._connect_ro() as conn:
            halted = self._scalar_state(conn, _HALT_KEY) == "1"
            halt_reason = self._scalar_state(conn, _HALT_REASON_KEY) or None
            hb_raw = self._scalar_state(conn, _HEARTBEAT_KEY)
            heartbeat_ms = int(hb_raw) if hb_raw not in (None, "") else None
            positions = self._positions(conn)
            executions = self._recent_executions(conn, limit=20)
            nav_hist = self._nav_history(conn)

        return {
            "mode": self.settings.mode.value,
            "is_live": self.settings.is_live,
            "halted": halted,
            "halt_reason": halt_reason,
            "last_heartbeat_ms": heartbeat_ms,
            "last_heartbeat_iso": _ms_to_iso(heartbeat_ms),
            "positions": positions,
            "executions": executions,
            "pnl_24h_pct": _pnl_24h_pct(nav_hist, now_ms),
            "drawdown_24h_pct": rolling_drawdown_pct(nav_hist, now_ms, _DAY_MS),
        }

    def _positions(self, conn: sqlite3.Connection) -> list[dict[str, Any]]:
        """Open spot holdings (quantity > 0), newest update first."""
        rows = conn.execute(
            "SELECT symbol, quantity, avg_entry_price, realized_pnl, "
            "opened_at, updated_at FROM positions "
            "WHERE quantity > 0 ORDER BY updated_at DESC;"
        ).fetchall()
        return [dict(r) for r in rows]

    def _recent_executions(
        self, conn: sqlite3.Connection, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Last ``limit`` execution_logs rows across all symbols, newest first."""
        rows = conn.execute(
            "SELECT id, decided_at, mode, symbol, timeframe, ts, action, "
            "confidence, price, quantity, notional, fee, slippage, accepted, "
            "reason FROM execution_logs ORDER BY id DESC LIMIT ?;",
            (int(limit),),
        ).fetchall()
        return [dict(r) for r in rows]


def _ms_to_iso(ms: Optional[int]) -> Optional[str]:
    if ms is None:
        return None
    from datetime import datetime, timezone

    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat()


def _pnl_24h_pct(nav_history: list, now_ms: int) -> Optional[float]:
    """Percent change of the latest NAV vs the earliest NAV in the trailing 24h.

    Returns None when there is too little history to compute it (fewer than two
    samples in the window, or a non-positive starting NAV).
    """
    pts = [p for p in nav_history if p[0] >= now_ms - _DAY_MS]
    if len(pts) < 2:
        return None
    pts.sort(key=lambda p: p[0])
    start = pts[0][1]
    current = pts[-1][1]
    if start <= 0:
        return None
    return (current / start - 1.0) * 100.0
