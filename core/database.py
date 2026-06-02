"""Persistence layer (data phase).

Owns the SQLite store and the three tables the data layer needs right now:

* ``candles``       — historical OHLCV bars, one row per (symbol, timeframe, ts).
* ``system_state``  — a small key/value table for cursors, flags, run metadata.
* ``positions``     — current spot holdings, one row per symbol.

Later phases add orders, fills, the risk gate, model governance, and the
execution engine. They are intentionally NOT created here.

SQLite pragmas applied on every connection:

* ``journal_mode=WAL``   — readers don't block the ingest writer, and a crash
  mid-write can't corrupt the file. Persists on the database, set per-connection
  to be safe.
* ``busy_timeout=5000``  — wait up to 5s for a lock instead of raising
  ``database is locked`` the instant two processes touch the file (e.g. a
  backtest reading while ``fetch_data`` writes).
* ``foreign_keys=ON``    — enforce the constraints we declare.

This is a SPOT account: positions are flat or long only. There is no short
side anywhere in this schema — quantity is constrained ``>= 0``.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, Optional

from config import Settings, get_settings

# Bumped whenever the schema in init_schema changes in a breaking way.
SCHEMA_VERSION = 1

_SQLITE_PREFIX = "sqlite:///"


class Database:
    """Thin wrapper around the configured SQLite datastore.

    Backtests and live trading share the same store so results are comparable
    and auditable. Only SQLite is implemented in the data phase; a non-sqlite
    ``DATABASE_URL`` is rejected early with a clear message rather than failing
    obscurely later.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.url = self.settings.database_url
        self.path = self._resolve_path(self.url)

    # --- connection ---------------------------------------------------------
    @staticmethod
    def _resolve_path(url: str) -> str:
        """Turn a ``sqlite:///...`` URL into a filesystem path.

        ``sqlite:///:memory:`` is passed through for tests.
        """
        if not url.startswith(_SQLITE_PREFIX):
            raise ValueError(
                f"Only sqlite URLs are supported in the data phase, got {url!r}. "
                "Use e.g. sqlite:///data/trading.db"
            )
        raw = url[len(_SQLITE_PREFIX):]
        return raw if raw == ":memory:" else raw

    def connect(self) -> sqlite3.Connection:
        """Open a connection with WAL, busy_timeout, and FK enforcement.

        The caller owns the connection lifecycle; prefer :meth:`session` for a
        managed transaction.
        """
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    @contextmanager
    def session(self) -> Iterator[sqlite3.Connection]:
        """Managed connection: commits on success, rolls back on error."""
        conn = self.connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # --- schema -------------------------------------------------------------
    def init_schema(self) -> None:
        """Create the data-layer tables if they do not already exist.

        Creates ONLY: candles, system_state, positions. Orders, fills, risk,
        model governance, and the execution engine are deliberately deferred to
        later phases.
        """
        with self.session() as conn:
            conn.executescript(
                """
                -- Historical OHLCV. ts is the bar OPEN time, epoch milliseconds
                -- (UTC), matching ccxt's fetch_ohlcv convention. The primary key
                -- makes re-ingesting a range idempotent via UPSERT.
                CREATE TABLE IF NOT EXISTS candles (
                    symbol    TEXT    NOT NULL,
                    timeframe TEXT    NOT NULL,
                    ts        INTEGER NOT NULL,
                    open      REAL    NOT NULL,
                    high      REAL    NOT NULL,
                    low       REAL    NOT NULL,
                    close     REAL    NOT NULL,
                    volume    REAL    NOT NULL,
                    PRIMARY KEY (symbol, timeframe, ts)
                );

                CREATE INDEX IF NOT EXISTS idx_candles_symbol_tf_ts
                    ON candles (symbol, timeframe, ts);

                -- Small key/value store: ingest cursors, schema version, flags.
                CREATE TABLE IF NOT EXISTS system_state (
                    key        TEXT PRIMARY KEY,
                    value      TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
                );

                -- Current SPOT holdings, one row per symbol. Spot account:
                -- flat or long only. quantity >= 0 is enforced; there is no
                -- short side. A fully-closed position is quantity = 0.
                CREATE TABLE IF NOT EXISTS positions (
                    symbol          TEXT PRIMARY KEY,
                    quantity        REAL NOT NULL DEFAULT 0 CHECK (quantity >= 0),
                    avg_entry_price REAL NOT NULL DEFAULT 0 CHECK (avg_entry_price >= 0),
                    realized_pnl    REAL NOT NULL DEFAULT 0,
                    opened_at       TEXT,
                    updated_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
                );
                """
            )
            conn.execute(
                "INSERT INTO system_state (key, value) VALUES ('schema_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, "
                "updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now');",
                (str(SCHEMA_VERSION),),
            )

    # --- system_state helpers ----------------------------------------------
    def set_state(self, key: str, value: str) -> None:
        with self.session() as conn:
            conn.execute(
                "INSERT INTO system_state (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, "
                "updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now');",
                (key, value),
            )

    def get_state(self, key: str) -> Optional[str]:
        with self.session() as conn:
            row = conn.execute(
                "SELECT value FROM system_state WHERE key = ?;", (key,)
            ).fetchone()
            return row["value"] if row is not None else None

    # --- candles ------------------------------------------------------------
    def upsert_candles(
        self, symbol: str, timeframe: str, rows: Iterable[Iterable[float]]
    ) -> int:
        """Insert/replace OHLCV rows. ``rows`` are ccxt-style [ts,o,h,l,c,v].

        Idempotent: re-ingesting an overlapping range updates in place rather
        than duplicating. Returns the number of rows written.
        """
        payload = [
            (symbol, timeframe, int(r[0]), float(r[1]), float(r[2]),
             float(r[3]), float(r[4]), float(r[5]))
            for r in rows
        ]
        if not payload:
            return 0
        with self.session() as conn:
            conn.executemany(
                "INSERT INTO candles "
                "(symbol, timeframe, ts, open, high, low, close, volume) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(symbol, timeframe, ts) DO UPDATE SET "
                "open=excluded.open, high=excluded.high, low=excluded.low, "
                "close=excluded.close, volume=excluded.volume;",
                payload,
            )
        return len(payload)

    def latest_candle_ts(self, symbol: str, timeframe: str) -> Optional[int]:
        """Open time (epoch ms) of the most recent stored bar, or None."""
        with self.session() as conn:
            row = conn.execute(
                "SELECT MAX(ts) AS ts FROM candles "
                "WHERE symbol = ? AND timeframe = ?;",
                (symbol, timeframe),
            ).fetchone()
            return int(row["ts"]) if row and row["ts"] is not None else None

    def count_candles(self, symbol: str, timeframe: str) -> int:
        with self.session() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM candles "
                "WHERE symbol = ? AND timeframe = ?;",
                (symbol, timeframe),
            ).fetchone()
            return int(row["n"])
