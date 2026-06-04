"""REQUIRED leakage test: settled-funding point-in-time alignment (BTC 4h experiment).

This is the derivatives analogue of the OHLCV no-overlap proof in
``test_features_labels.py``. It proves the pre-registered point-in-time rule:

    The funding feature for a 4h candle that closes at t uses ONLY funding that
    has SETTLED at or before t. Funding settling strictly after t is never used.

Proven by perturbation, the same technique as the OHLCV test:

  (1) Corrupting EVERY funding settlement strictly AFTER candle T's close leaves
      T's funding feature row BYTE-IDENTICAL (the core guarantee).
  (2) Converse / anti-vacuity: corrupting the funding that settles AT-OR-BEFORE
      T's close DOES change T's row — so the feature genuinely depends on the
      point-in-time value, and (1) is a real result, not a feature that ignores
      funding.

The experiment module lives outside the package tree (job tmp dir); it is
imported by path. If it is absent (e.g. running the suite on a clean checkout
without the experiment artifacts), the test SKIPS rather than failing — it
guards the experiment, and only applies when the experiment code is present.
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np
import pandas as pd
import pytest

# Locate the experiment's derivatives module in the job tmp dir.
_TMP = r"C:\Users\hkhus\.claude\jobs\5dc85b2c\tmp"
_DERIV = os.path.join(_TMP, "derivatives.py")

if not os.path.exists(_DERIV):
    pytest.skip("experiment derivatives module not present", allow_module_level=True)

_spec = importlib.util.spec_from_file_location("experiment_derivatives", _DERIV)
deriv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(deriv)

FOUR_H = deriv.FOUR_H_MS
EIGHT_H = 8 * 60 * 60 * 1000


def _feature_frame(n_candles: int = 300, seed: int = 11):
    """A minimal feature frame indexed by 4h candle OPEN ts, plus synthetic
    8h funding settlements covering it. Returns (features, funding_list).

    The feature frame only needs an index for the funding alignment; we attach a
    dummy column so add_funding_features has something to copy/return.
    """
    rng = np.random.default_rng(seed)
    open_ts = np.arange(n_candles, dtype=np.int64) * FOUR_H + 1_000_000_000_000
    features = pd.DataFrame(
        {"dummy": rng.normal(size=n_candles)},
        index=pd.Index(open_ts, name="ts"),
    )
    # Funding settles every 8h, starting before the first candle so early rows
    # already have a settled value. Distinct rates so any swap is detectable.
    first_settle = int(open_ts[0]) - EIGHT_H
    last_close = int(open_ts[-1]) + FOUR_H
    settle_ts = np.arange(first_settle, last_close + EIGHT_H, EIGHT_H, dtype=np.int64)
    rates = rng.normal(0.0001, 0.0003, size=len(settle_ts))
    funding = [(int(t), float(r)) for t, r in zip(settle_ts, rates)]
    return features, funding


def _build(features, funding):
    # Small z-window so the synthetic series clears warmup; alignment is the
    # property under test, not the z-window length.
    return deriv.add_funding_features(features, funding, z_window=5)


def test_perturbing_future_funding_leaves_T_row_byte_identical():
    """(1) The core point-in-time guarantee."""
    features, funding = _feature_frame()
    base = _build(features, funding)

    # Choose a candle T comfortably past the z-window warmup.
    T = base.index[len(base) // 2]
    T_close = int(T) + FOUR_H
    row_before = base.loc[T].copy()

    # Corrupt EVERY funding settlement strictly AFTER T's close.
    corrupted = [
        (ts, (rate if ts <= T_close else rate + 999.0))  # absurd perturbation
        for ts, rate in funding
    ]
    after = _build(features, corrupted)

    # T's funding feature row must be byte-for-byte identical.
    pd.testing.assert_series_equal(after.loc[T], row_before)

    # Stronger: the ENTIRE prefix up to and including T is unchanged (no row that
    # closes at/before T saw any future funding).
    prefix = base.index[base.index <= T]
    pd.testing.assert_frame_equal(after.loc[prefix], base.loc[prefix])


def test_perturbing_future_funding_does_change_LATER_rows():
    """Sanity: rows that close AFTER the perturbed settlements DO move, proving
    the perturbation is real and the invariance above is meaningful."""
    features, funding = _feature_frame()
    base = _build(features, funding)

    T = base.index[len(base) // 2]
    T_close = int(T) + FOUR_H
    corrupted = [
        (ts, (rate if ts <= T_close else rate + 999.0))
        for ts, rate in funding
    ]
    after = _build(features, corrupted)

    later = base.index[base.index > T]
    # At least one later row's funding_rate differs (those whose close is past a
    # corrupted settlement pick up the corrupted value).
    assert not np.allclose(
        after.loc[later, "funding_rate"].to_numpy(),
        base.loc[later, "funding_rate"].to_numpy(),
    )


def test_perturbing_the_settled_value_at_or_before_T_changes_T_row():
    """(2) Anti-vacuity: the feature DOES use the most-recent settled value, so
    changing that value moves T's row. Otherwise (1) would be trivially true for
    a feature that ignores funding entirely."""
    features, funding = _feature_frame()
    base = _build(features, funding)

    T = base.index[len(base) // 2]
    T_close = int(T) + FOUR_H

    # Find the settlement in force at T's close: the LAST one with ts <= T_close.
    in_force = max(ts for ts, _ in funding if ts <= T_close)
    bumped = [
        (ts, (rate + 0.05 if ts == in_force else rate))
        for ts, rate in funding
    ]
    after = _build(features, bumped)

    # T's funding_rate must reflect the bumped in-force value (changed).
    assert after.loc[T, "funding_rate"] != base.loc[T, "funding_rate"]


def test_candle_uses_settlement_at_or_before_close_not_after():
    """Explicit boundary check on the index math: for each candle, the chosen
    settlement ts is <= that candle's CLOSE, and it is the most recent such."""
    features, funding = _feature_frame(n_candles=120, seed=3)
    f_ts = np.array([t for t, _ in funding], dtype=np.int64)
    f_rate = np.array([r for _, r in funding], dtype=float)
    open_ts = features.index.to_numpy(dtype=np.int64)

    chosen = deriv.settled_funding_for_closes(open_ts, f_ts, f_rate)
    closes = open_ts + FOUR_H
    for i, close in enumerate(closes):
        eligible = f_ts[f_ts <= close]
        if len(eligible) == 0:
            assert np.isnan(chosen[i])
            continue
        expected_ts = eligible.max()                 # most recent settled <= close
        expected_rate = f_rate[np.where(f_ts == expected_ts)[0][0]]
        assert chosen[i] == pytest.approx(expected_rate)
        # And it is strictly NOT any settlement after the close.
        assert expected_ts <= close
