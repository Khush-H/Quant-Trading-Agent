"""Feature engineering (features/labels phase).

Transforms raw OHLCV into a strictly *causal* feature matrix: a feature row at
time ``t`` uses ONLY the closed candle at ``t`` and bars before it. No rolling
window reaches into ``t+1``, so nothing here can see the bar the label predicts.

``ts`` convention
-----------------
A feature row's ``ts`` is the OPEN time (epoch ms, UTC) of the CLOSED candle the
features were computed FROM — the same bar convention the data layer uses after
dropping the incomplete trailing bar (see :mod:`scripts.fetch_data`). It is NOT
the bar being predicted. The label (:mod:`ml.labels`) looks at the forward
return measured from candles strictly AFTER ``ts``; the windows never overlap.

Feature set (all stationary, all trailing)
-------------------------------------------
* ``log_ret_1h``  — 1-bar log return,  ``ln(close_t / close_{t-1})``.
* ``log_ret_4h``  — 4-bar log return,  ``ln(close_t / close_{t-4})``.
* ``gk_vol``      — Garman-Klass volatility from this bar's OHLC.
* ``gk_vol_ma24`` — 24-period trailing mean of ``gk_vol`` (ending at t).
* ``z_close_50``  — 50-period trailing z-score of close (ending at t).
* ``z_vol``       — 50-period trailing z-score of volume (ending at t).

The currently-forming candle is already dropped at ingest time, so every bar
read from the DB is closed; using bar ``t`` in row ``t`` is therefore causal.
"""

from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np
import pandas as pd

from config import Settings, get_settings
from core.database import FEATURE_COLUMNS, Database

# Lookback windows. Kept as named constants because they are part of the
# feature recipe that feature_hash pins.
GK_VOL_MA_WINDOW = 24
ZSCORE_WINDOW = 50
LOG_RET_LONG_LAG = 4

# Bump when the feature computation changes in a way that should invalidate
# previously-stored feature rows / trained models.
FEATURE_VERSION = 1


def _zscore(series: pd.Series, window: int) -> pd.Series:
    """Trailing z-score: (x_t - mean of last `window`) / std of last `window`.

    Both the mean and std are computed over the trailing window ENDING at t
    (inclusive), so the value at t never uses t+1. ``min_periods=window`` makes
    the warmup rows NaN rather than computing on a short, biased window; they
    are dropped by :func:`build_features`.
    """
    roll = series.rolling(window=window, min_periods=window)
    mean = roll.mean()
    std = roll.std(ddof=0)
    return (series - mean) / std


def _garman_klass_vol(df: pd.DataFrame) -> pd.Series:
    """Per-bar Garman-Klass volatility estimate from OHLC of the SAME bar.

    GK uses only the open/high/low/close of the (closed) bar at t, so it is
    causal. Returns the square root of the variance estimate (a volatility,
    not a variance). Guards against non-positive prices.
    """
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_hl = np.log(h / low)
        log_co = np.log(c / o)
        var = 0.5 * log_hl**2 - (2.0 * np.log(2.0) - 1.0) * log_co**2
    var = np.where(var < 0.0, 0.0, var)  # numerical floor; GK var is >= 0
    return pd.Series(np.sqrt(var), index=df.index)


def compute_feature_hash() -> str:
    """Stable hash of the feature RECIPE (not of any data).

    Combines the ordered feature names, the lookback parameters, and the
    feature-code version. Identical for every row of one build; changes only
    when the recipe changes. The model phase can store this alongside a trained
    artifact and refuse to predict on features built under a different recipe.
    """
    recipe = "|".join(
        [
            f"v{FEATURE_VERSION}",
            "cols=" + ",".join(FEATURE_COLUMNS),
            f"gk_ma={GK_VOL_MA_WINDOW}",
            f"z={ZSCORE_WINDOW}",
            f"ret_lag={LOG_RET_LONG_LAG}",
        ]
    )
    return hashlib.sha256(recipe.encode("utf-8")).hexdigest()[:16]


def build_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Return the causal feature matrix for one symbol/timeframe.

    ``ohlcv`` must be indexed by ``ts`` (bar open time, ms) ascending, with
    columns open/high/low/close/volume — every bar already closed. The result
    is indexed by the same ``ts`` and has exactly the columns in
    :data:`FEATURE_COLUMNS`. Warmup rows (where any trailing window is short)
    are dropped via ``dropna`` so no row is computed on partial history.

    Causality: every feature at row t depends only on close/volume/OHLC at t
    and earlier. No ``shift(-k)`` / forward window appears here.
    """
    if not ohlcv.index.is_monotonic_increasing:
        ohlcv = ohlcv.sort_index()

    close = ohlcv["close"]
    volume = ohlcv["volume"]

    feats = pd.DataFrame(index=ohlcv.index)
    # Log returns: ratio of current close to a PAST close (positive lag).
    feats["log_ret_1h"] = np.log(close / close.shift(1))
    feats["log_ret_4h"] = np.log(close / close.shift(LOG_RET_LONG_LAG))
    # Garman-Klass vol of the current bar, and its trailing mean.
    gk = _garman_klass_vol(ohlcv)
    feats["gk_vol"] = gk
    feats["gk_vol_ma24"] = gk.rolling(
        window=GK_VOL_MA_WINDOW, min_periods=GK_VOL_MA_WINDOW
    ).mean()
    # Trailing z-scores.
    feats["z_close_50"] = _zscore(close, ZSCORE_WINDOW)
    feats["z_vol"] = _zscore(volume, ZSCORE_WINDOW)

    feats = feats[list(FEATURE_COLUMNS)]
    return feats.dropna()


def build_and_store(
    symbol: str,
    timeframe: str,
    db: Optional[Database] = None,
    settings: Optional[Settings] = None,
) -> int:
    """Build features from stored candles and persist them.

    Reads candles via :meth:`Database.load_candles`, builds the causal matrix,
    tags every row with :func:`compute_feature_hash`, and upserts into the
    ``features`` table. Returns the number of feature rows written.
    """
    settings = settings or get_settings()
    db = db or Database(settings)
    db.init_schema()

    rows = db.load_candles(symbol, timeframe)
    if not rows:
        return 0
    ohlcv = pd.DataFrame(
        [dict(r) for r in rows]
    ).set_index("ts").sort_index()

    feats = build_features(ohlcv)
    if feats.empty:
        return 0

    fh = compute_feature_hash()
    payload = [
        {"ts": int(ts), **{c: float(row[c]) for c in FEATURE_COLUMNS}}
        for ts, row in feats.iterrows()
    ]
    return db.upsert_features(symbol, timeframe, payload, feature_hash=fh)
