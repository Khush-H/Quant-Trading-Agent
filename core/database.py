"""Persistence layer (data phase).

Owns the SQLite store and the tables built so far:

* ``candles``       — historical OHLCV bars, one row per (symbol, timeframe, ts).
* ``system_state``  — a small key/value table for cursors, flags, run metadata.
* ``positions``     — current spot holdings, one row per symbol.
* ``features``      — the causal feature matrix (features phase), one row per
  (symbol, timeframe, ts) where ``ts`` is the CLOSED candle the features were
  computed from.

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
# v2 adds the features table (features/labels phase).
# v3 adds the execution_logs table (paper-trading phase).
SCHEMA_VERSION = 3

# Ordered feature columns written to the features table. The order is part of
# the feature recipe and is hashed into feature_hash by ml.features.
FEATURE_COLUMNS = (
    "log_ret_1h",
    "log_ret_4h",
    "gk_vol",
    "gk_vol_ma24",
    "z_close_50",
    "z_vol",
)

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

                -- Causal feature matrix. One row per (symbol, timeframe, ts).
                --
                -- ts is the OPEN time (epoch ms, UTC) of the CLOSED candle the
                -- features were computed FROM — the same bar convention the
                -- data layer uses after dropping the incomplete trailing bar.
                -- It is NOT the bar being predicted. The label for this row
                -- looks at candles strictly after ts (see ml.labels), so the
                -- feature window and the label window never overlap.
                --
                -- Every column is a wide REAL feature; the set/order is fixed
                -- by FEATURE_COLUMNS. feature_hash identifies the feature
                -- RECIPE (ordered names + params + code version), so it is the
                -- same for every row of one build and lets the model phase
                -- assert it trains on the exact feature definition it expects.
                CREATE TABLE IF NOT EXISTS features (
                    symbol       TEXT    NOT NULL,
                    timeframe    TEXT    NOT NULL,
                    ts           INTEGER NOT NULL,
                    log_ret_1h   REAL    NOT NULL,
                    log_ret_4h   REAL    NOT NULL,
                    gk_vol       REAL    NOT NULL,
                    gk_vol_ma24  REAL    NOT NULL,
                    z_close_50   REAL    NOT NULL,
                    z_vol        REAL    NOT NULL,
                    feature_hash TEXT    NOT NULL,
                    created_at   TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                    PRIMARY KEY (symbol, timeframe, ts)
                );

                CREATE INDEX IF NOT EXISTS idx_features_symbol_tf_ts
                    ON features (symbol, timeframe, ts);

                -- Audit trail of every daemon decision, including no-trades.
                -- One row per decision cycle per symbol. action is one of
                -- 'buy' | 'sell' | 'hold'. For simulated (paper) fills the
                -- realized fee and slippage are recorded so paper results can
                -- be reconciled against the backtester's cost model. ts is the
                -- CLOSED candle the decision was made on (epoch ms, UTC).
                CREATE TABLE IF NOT EXISTS execution_logs (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    decided_at    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                    mode          TEXT    NOT NULL,
                    symbol        TEXT    NOT NULL,
                    timeframe     TEXT    NOT NULL,
                    ts            INTEGER,
                    action        TEXT    NOT NULL CHECK (action IN ('buy','sell','hold')),
                    confidence    REAL,
                    price         REAL,
                    quantity      REAL    NOT NULL DEFAULT 0 CHECK (quantity >= 0),
                    notional      REAL    NOT NULL DEFAULT 0,
                    fee           REAL    NOT NULL DEFAULT 0,
                    slippage      REAL    NOT NULL DEFAULT 0,
                    accepted      INTEGER NOT NULL DEFAULT 1,
                    reason        TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_exec_logs_symbol_ts
                    ON execution_logs (symbol, ts);
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

    def load_candles(
        self, symbol: str, timeframe: str
    ) -> list[sqlite3.Row]:
        """Return all stored bars for (symbol, timeframe), oldest first.

        Rows carry ts/open/high/low/close/volume. The feature builder wraps
        these into a DataFrame; kept as plain rows here so the data layer has
        no pandas dependency.
        """
        with self.session() as conn:
            return conn.execute(
                "SELECT ts, open, high, low, close, volume FROM candles "
                "WHERE symbol = ? AND timeframe = ? ORDER BY ts ASC;",
                (symbol, timeframe),
            ).fetchall()

    # --- features -----------------------------------------------------------
    def upsert_features(
        self,
        symbol: str,
        timeframe: str,
        rows: Iterable[dict],
        feature_hash: str,
    ) -> int:
        """Insert/replace feature rows for (symbol, timeframe).

        Each item in ``rows`` is a dict with key ``ts`` (open time of the
        CLOSED candle the features were computed from) plus one key per
        :data:`FEATURE_COLUMNS`. Idempotent on (symbol, timeframe, ts).
        Returns the number of rows written.
        """
        payload = [
            (
                symbol,
                timeframe,
                int(r["ts"]),
                *(float(r[c]) for c in FEATURE_COLUMNS),
                feature_hash,
            )
            for r in rows
        ]
        if not payload:
            return 0
        cols = ", ".join(FEATURE_COLUMNS)
        set_clause = ", ".join(f"{c}=excluded.{c}" for c in FEATURE_COLUMNS)
        placeholders = ", ".join(["?"] * (3 + len(FEATURE_COLUMNS) + 1))
        with self.session() as conn:
            conn.executemany(
                f"INSERT INTO features (symbol, timeframe, ts, {cols}, feature_hash) "
                f"VALUES ({placeholders}) "
                "ON CONFLICT(symbol, timeframe, ts) DO UPDATE SET "
                f"{set_clause}, feature_hash=excluded.feature_hash;",
                payload,
            )
        return len(payload)

    # --- positions ----------------------------------------------------------
    def get_position(self, symbol: str) -> Optional[sqlite3.Row]:
        """Return the stored position row for ``symbol``, or None if flat/absent."""
        with self.session() as conn:
            return conn.execute(
                "SELECT symbol, quantity, avg_entry_price, realized_pnl, "
                "opened_at, updated_at FROM positions WHERE symbol = ?;",
                (symbol,),
            ).fetchone()

    def upsert_position(
        self,
        symbol: str,
        quantity: float,
        avg_entry_price: float,
        realized_pnl: float = 0.0,
    ) -> None:
        """Persist the current spot holding for ``symbol``.

        Spot account: ``quantity`` must be ``>= 0`` (the table CHECK enforces it
        too). A fully-closed position is ``quantity = 0``.
        """
        if quantity < 0:
            raise ValueError(
                f"Spot positions cannot be short: quantity={quantity!r} (>= 0)."
            )
        with self.session() as conn:
            conn.execute(
                "INSERT INTO positions "
                "(symbol, quantity, avg_entry_price, realized_pnl, updated_at) "
                "VALUES (?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ','now')) "
                "ON CONFLICT(symbol) DO UPDATE SET "
                "quantity=excluded.quantity, "
                "avg_entry_price=excluded.avg_entry_price, "
                "realized_pnl=excluded.realized_pnl, "
                "updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now');",
                (symbol, float(quantity), float(avg_entry_price), float(realized_pnl)),
            )

    # --- execution logs -----------------------------------------------------
    def log_execution(
        self,
        *,
        mode: str,
        symbol: str,
        timeframe: str,
        action: str,
        ts: Optional[int] = None,
        confidence: Optional[float] = None,
        price: Optional[float] = None,
        quantity: float = 0.0,
        notional: float = 0.0,
        fee: float = 0.0,
        slippage: float = 0.0,
        accepted: bool = True,
        reason: Optional[str] = None,
    ) -> int:
        """Append one decision to ``execution_logs``. Returns the row id.

        Every decision is logged, including holds/no-trades (action='hold',
        quantity=0). For simulated fills ``fee`` and ``slippage`` carry the
        cost the backtester's model would charge, so paper and backtest can be
        reconciled.
        """
        with self.session() as conn:
            cur = conn.execute(
                "INSERT INTO execution_logs "
                "(mode, symbol, timeframe, ts, action, confidence, price, "
                " quantity, notional, fee, slippage, accepted, reason) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);",
                (mode, symbol, timeframe, ts, action, confidence, price,
                 float(quantity), float(notional), float(fee), float(slippage),
                 1 if accepted else 0, reason),
            )
            return int(cur.lastrowid)

    def recent_executions(self, symbol: str, limit: int = 20) -> list[sqlite3.Row]:
        with self.session() as conn:
            return conn.execute(
                "SELECT * FROM execution_logs WHERE symbol = ? "
                "ORDER BY id DESC LIMIT ?;",
                (symbol, limit),
            ).fetchall()

    # --- risk-gate state (all in system_state; HALT must not self-clear) ----
    # Keys used by the risk gate. Centralized so nothing typos a key name.
    HALT_KEY = "SYSTEM_HALT"
    HALT_REASON_KEY = "SYSTEM_HALT_REASON"
    HEARTBEAT_KEY = "heartbeat_ts_ms"
    EXCHANGE_FAILS_KEY = "exchange_consecutive_failures"
    NAV_HISTORY_KEY = "nav_history"

    def is_halted(self) -> bool:
        return self.get_state(self.HALT_KEY) == "1"

    def halt_reason(self) -> Optional[str]:
        return self.get_state(self.HALT_REASON_KEY)

    def set_halt(self, reason: str) -> None:
        """Trip SYSTEM_HALT. Idempotent; never overwrites an earlier reason."""
        if self.get_state(self.HALT_KEY) == "1":
            return  # already halted; keep the original reason
        self.set_state(self.HALT_KEY, "1")
        self.set_state(self.HALT_REASON_KEY, reason)

    def clear_halt(self) -> None:
        """Clear SYSTEM_HALT. ONLY the manual reset script should call this.

        HALT never self-clears anywhere in the engine/risk path — this method
        exists solely for the operator-run reset.
        """
        self.set_state(self.HALT_KEY, "0")
        self.set_state(self.HALT_REASON_KEY, "")

    def record_heartbeat(self, ts_ms: int) -> None:
        self.set_state(self.HEARTBEAT_KEY, str(int(ts_ms)))

    def last_heartbeat_ms(self) -> Optional[int]:
        v = self.get_state(self.HEARTBEAT_KEY)
        return int(v) if v is not None and v != "" else None

    def record_exchange_failure(self) -> int:
        """Increment and return the consecutive-failure counter."""
        cur = self.get_state(self.EXCHANGE_FAILS_KEY)
        n = (int(cur) if cur not in (None, "") else 0) + 1
        self.set_state(self.EXCHANGE_FAILS_KEY, str(n))
        return n

    def reset_exchange_failures(self) -> None:
        self.set_state(self.EXCHANGE_FAILS_KEY, "0")

    def exchange_consecutive_failures(self) -> int:
        v = self.get_state(self.EXCHANGE_FAILS_KEY)
        return int(v) if v not in (None, "") else 0

    def record_nav(self, ts_ms: int, nav: float, keep_ms: int = 48 * 3_600_000) -> None:
        """Append a (ts_ms, nav) sample to the rolling NAV history.

        Stored as a JSON list in system_state; trimmed to ``keep_ms`` of recent
        history so it can't grow unbounded. 48h kept by default (the 24h
        drawdown window plus headroom).
        """
        import json

        hist = self.nav_history()
        hist.append([int(ts_ms), float(nav)])
        cutoff = int(ts_ms) - keep_ms
        hist = [p for p in hist if p[0] >= cutoff]
        self.set_state(self.NAV_HISTORY_KEY, json.dumps(hist))

    def nav_history(self) -> list[list]:
        import json

        v = self.get_state(self.NAV_HISTORY_KEY)
        if not v:
            return []
        try:
            return list(json.loads(v))
        except (ValueError, TypeError):
            return []
