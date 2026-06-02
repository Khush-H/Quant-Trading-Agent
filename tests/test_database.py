"""Tests for the data-layer persistence (candles, system_state, positions).

These lock in the schema additions we deliberately want: separate tables, WAL
mode, a busy_timeout, idempotent candle ingest, and the spot-only (no short)
positions constraint. Orders/fills/risk are NOT part of this phase and are not
tested here.
"""

from __future__ import annotations

import sqlite3

import pytest

from config.settings import Settings
from core.database import SCHEMA_VERSION, Database


def _db(tmp_path) -> Database:
    settings = Settings(_env_file=None, database_url=f"sqlite:///{tmp_path/'t.db'}")
    db = Database(settings)
    db.init_schema()
    return db


def test_only_sqlite_urls_supported():
    with pytest.raises(ValueError, match="sqlite"):
        Database(Settings(_env_file=None, database_url="postgresql://x/y"))


def test_init_creates_exactly_the_data_phase_tables(tmp_path):
    db = _db(tmp_path)
    with db.session() as conn:
        names = {
            r["name"]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table';"
            )
        }
    assert {"candles", "system_state", "positions"} <= names
    # The later-phase tables must NOT exist yet.
    assert not ({"orders", "fills", "runs", "models"} & names)


def test_wal_and_busy_timeout_pragmas(tmp_path):
    db = _db(tmp_path)
    with db.session() as conn:
        mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        busy = conn.execute("PRAGMA busy_timeout;").fetchone()[0]
    assert mode.lower() == "wal"
    assert busy == 5000


def test_schema_version_recorded(tmp_path):
    db = _db(tmp_path)
    assert db.get_state("schema_version") == str(SCHEMA_VERSION)


def test_init_schema_is_idempotent(tmp_path):
    db = _db(tmp_path)
    db.init_schema()  # second call must not raise
    assert db.get_state("schema_version") == str(SCHEMA_VERSION)


def test_system_state_roundtrip(tmp_path):
    db = _db(tmp_path)
    assert db.get_state("missing") is None
    db.set_state("cursor", "123")
    assert db.get_state("cursor") == "123"
    db.set_state("cursor", "456")  # upsert overwrites
    assert db.get_state("cursor") == "456"


def test_candle_upsert_is_idempotent(tmp_path):
    db = _db(tmp_path)
    rows = [
        [1_000, 10.0, 11.0, 9.0, 10.5, 100.0],
        [2_000, 10.5, 12.0, 10.0, 11.5, 120.0],
    ]
    assert db.upsert_candles("BTC/USDT", "1h", rows) == 2
    # Re-ingesting an overlapping range updates rather than duplicating.
    rows[1][4] = 99.0  # change the close of the 2nd bar
    db.upsert_candles("BTC/USDT", "1h", rows)
    assert db.count_candles("BTC/USDT", "1h") == 2
    with db.session() as conn:
        close = conn.execute(
            "SELECT close FROM candles WHERE symbol=? AND timeframe=? AND ts=?;",
            ("BTC/USDT", "1h", 2_000),
        ).fetchone()["close"]
    assert close == 99.0


def test_latest_candle_ts(tmp_path):
    db = _db(tmp_path)
    assert db.latest_candle_ts("BTC/USDT", "1h") is None
    db.upsert_candles(
        "BTC/USDT", "1h",
        [[1_000, 1, 1, 1, 1, 1], [3_000, 1, 1, 1, 1, 1], [2_000, 1, 1, 1, 1, 1]],
    )
    assert db.latest_candle_ts("BTC/USDT", "1h") == 3_000


def test_positions_reject_short_quantity(tmp_path):
    """SPOT account: quantity must be >= 0. A short (negative) is rejected."""
    db = _db(tmp_path)
    with pytest.raises(sqlite3.IntegrityError):
        with db.session() as conn:
            conn.execute(
                "INSERT INTO positions (symbol, quantity) VALUES (?, ?);",
                ("BTC/USDT", -1.0),
            )


def test_positions_allow_flat_and_long(tmp_path):
    db = _db(tmp_path)
    with db.session() as conn:
        conn.execute("INSERT INTO positions (symbol, quantity) VALUES ('A', 0);")
        conn.execute("INSERT INTO positions (symbol, quantity) VALUES ('B', 2.5);")
    with db.session() as conn:
        qtys = {
            r["symbol"]: r["quantity"]
            for r in conn.execute("SELECT symbol, quantity FROM positions;")
        }
    assert qtys == {"A": 0.0, "B": 2.5}
