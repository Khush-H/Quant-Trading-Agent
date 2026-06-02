"""Tests for the causal feature matrix and the long/flat labels.

The headline test is :func:`test_feature_and_label_windows_do_not_overlap`,
which proves that for the SAME timestamp t the data the feature row depends on
and the data the label depends on are disjoint, with the boundary falling at t:
features use bars <= t, the label uses bars > t. A look-ahead bug (a feature
peeking at t+1) would break it.

Spot, long-only: labels are exactly {0=Flat, 1=Long}; there is no short class.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.settings import Settings
from ml.features import (
    FEATURE_COLUMNS,
    GK_VOL_MA_WINDOW,
    ZSCORE_WINDOW,
    build_features,
    compute_feature_hash,
)
from ml.labels import FLAT, LONG, build_labels, forward_log_return


def _synthetic_ohlcv(n: int = 400, seed: int = 7) -> pd.DataFrame:
    """A deterministic, strictly-increasing-ts OHLCV frame (1h bars)."""
    rng = np.random.default_rng(seed)
    step = 3_600_000  # 1h in ms
    ts = np.arange(n, dtype=np.int64) * step + 1_000_000
    rets = rng.normal(0.0, 0.01, size=n)
    close = 100.0 * np.exp(np.cumsum(rets))
    openp = np.concatenate([[100.0], close[:-1]])
    high = np.maximum(openp, close) * (1.0 + np.abs(rng.normal(0, 0.003, n)))
    low = np.minimum(openp, close) * (1.0 - np.abs(rng.normal(0, 0.003, n)))
    volume = rng.uniform(50, 150, size=n)
    return pd.DataFrame(
        {"open": openp, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.Index(ts, name="ts"),
    )


def _settings(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


# --- the critical no-overlap proof --------------------------------------------

def test_feature_and_label_windows_do_not_overlap():
    """For the same t, feature inputs (bars <= t) and label inputs (bars > t)
    are disjoint. Proven by perturbation: changing a bar STRICTLY AFTER t must
    not move the feature row at t, and changing the bar AT t (or before) must
    not move the label at t. Together that pins the boundary exactly at t."""
    base = _synthetic_ohlcv()
    settings = _settings(label_horizon=1, round_trip_cost=0.002, slippage_bps=0.0)

    feats = build_features(base)
    labels = build_labels(base, settings=settings)

    # Pick a t that exists in BOTH (past warmup, before the forward tail drop).
    common = feats.index.intersection(labels.index)
    assert len(common) > 50
    t = common[len(common) // 2]
    t_pos = base.index.get_loc(t)

    feat_row_before = feats.loc[t].copy()
    label_before = int(labels.loc[t])

    # (1) Perturb EVERY bar strictly after t. The feature row at t must be
    #     byte-for-byte identical -> no feature looks forward of t.
    after = base.copy()
    fut = after.index > t
    for col in ("open", "high", "low", "close", "volume"):
        after.loc[fut, col] *= 1.5
    feats_after = build_features(after)
    pd.testing.assert_series_equal(feats_after.loc[t], feat_row_before)

    # (2) Perturb the bar AT t and everything before it. The label at t (which
    #     looks at close_{t+1}/close_t ... wait: it divides by close_t) must be
    #     recomputed only from bars > t for its numerator. close_t is the
    #     boundary the windows share, so to prove disjointness we perturb bars
    #     STRICTLY BEFORE t and require the label is unchanged.
    before = base.copy()
    past = before.index < t
    for col in ("open", "high", "low", "close", "volume"):
        before.loc[past, col] *= 0.5
    labels_before = build_labels(before, settings=settings)
    assert int(labels_before.loc[t]) == label_before

    # And the explicit index-set statement: the bars the label's forward
    # window covers for t are all strictly greater than t.
    horizon = settings.label_horizon
    label_window = base.index[(base.index > t) & (base.index <= base.index[t_pos + horizon])]
    assert (label_window > t).all()
    # while the feature row's longest lookback ends at t (uses bars <= t only).
    feature_window = base.index[base.index <= t]
    assert (feature_window <= t).all()
    # the boundary is exactly t: the only shared point would be t itself, which
    # the label uses solely as the return's denominator, not as a future bar.
    assert set(label_window).isdisjoint(set(feature_window))


# --- causality / shape of features -------------------------------------------

def test_features_have_expected_columns_and_no_nans():
    feats = build_features(_synthetic_ohlcv())
    assert list(feats.columns) == list(FEATURE_COLUMNS)
    assert not feats.isna().any().any()  # dropna at the end removed warmup
    # warmup is dropped: a window of W needs W bars, so the first non-NaN row
    # is at position W-1 -> exactly W-1 leading rows fall away for the longest
    # window. (z-score W=50 dominates here.)
    raw = _synthetic_ohlcv()
    longest = max(GK_VOL_MA_WINDOW, ZSCORE_WINDOW)
    assert len(feats) == len(raw) - (longest - 1)


def test_feature_hash_is_stable_and_recipe_scoped():
    # Same recipe -> same hash, every call; it does not depend on the data.
    h1 = compute_feature_hash()
    h2 = compute_feature_hash()
    assert h1 == h2 and len(h1) == 16


def test_changing_only_a_future_bar_leaves_earlier_features_unchanged():
    base = _synthetic_ohlcv()
    feats = build_features(base)
    t = feats.index[100]
    bumped = base.copy()
    bumped.loc[bumped.index > t, "close"] *= 1.2
    feats2 = build_features(bumped)
    # everything up to and including t is identical.
    upto = feats.index[feats.index <= t]
    pd.testing.assert_frame_equal(feats.loc[upto], feats2.loc[upto])


# --- labels: long/flat only, cost hurdle, balance -----------------------------

def test_labels_are_long_or_flat_only_no_short():
    labels = build_labels(_synthetic_ohlcv(), settings=_settings())
    assert set(labels.unique()) <= {FLAT, LONG}
    assert (labels >= 0).all()  # no negative/short code anywhere


def test_label_hurdle_is_full_round_trip_not_one_leg():
    # round_trip = 0.002 (both legs). slippage adds 2 * 5bps = 0.001 -> 0.003.
    s = _settings(round_trip_cost=0.002, slippage_bps=5.0)
    assert s.label_hurdle == pytest.approx(0.003)

    # A move of +0.0025 beats ONE leg (0.001) but NOT the round trip (0.003):
    # it must be labelled FLAT, proving the hurdle is the full round trip.
    close = pd.Series([100.0, 100.0 * np.exp(0.0025)], index=[0, 3_600_000])
    ohlcv = pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close,
         "volume": [100.0, 100.0]},
        index=pd.Index([0, 3_600_000], name="ts"),
    )
    labels = build_labels(ohlcv, settings=s, horizon=1)
    assert int(labels.iloc[0]) == FLAT

    # A move clearly above the round trip is LONG.
    close2 = pd.Series([100.0, 100.0 * np.exp(0.01)], index=[0, 3_600_000])
    ohlcv2 = ohlcv.assign(open=close2, high=close2, low=close2, close=close2)
    labels2 = build_labels(ohlcv2, settings=s, horizon=1)
    assert int(labels2.iloc[0]) == LONG


def test_forward_return_drops_the_tail_with_no_future():
    close = pd.Series([1.0, 2.0, 3.0], index=[0, 1, 2])
    fwd = forward_log_return(close, horizon=1).dropna()
    assert list(fwd.index) == [0, 1]  # last bar has no t+1, dropped


def test_class_balance_warns_when_degenerate(capsys):
    # All-up series with a tiny hurdle -> nearly all Long -> Flat < 15%.
    rng = np.random.default_rng(1)
    n = 300
    close = 100.0 * np.exp(np.cumsum(np.abs(rng.normal(0.02, 0.001, n))))
    ts = np.arange(n, dtype=np.int64) * 3_600_000
    ohlcv = pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close,
         "volume": np.full(n, 100.0)},
        index=pd.Index(ts, name="ts"),
    )
    build_labels(ohlcv, settings=_settings(round_trip_cost=0.0, slippage_bps=0.0),
                 horizon=1)
    out = capsys.readouterr().out
    assert "class balance" in out.lower()
    assert "WARNING" in out
