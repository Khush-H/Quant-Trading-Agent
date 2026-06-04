"""Leakage check for the point-in-time realized-volatility estimate.

Mirrors the OHLCV and funding no-look-ahead proofs: the vol estimate at bar T
uses only returns through T's close, so perturbing ANY bar strictly after T
leaves T's vol byte-identical. Also a converse anti-vacuity check: perturbing a
bar at/before T DOES move T's vol (the estimate genuinely uses recent returns).

The vol module lives in the job tmp dir (experiment code, outside the package);
imported by path, skips cleanly if absent.
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np
import pandas as pd
import pytest

_TMP = r"C:\Users\hkhus\.claude\jobs\5dc85b2c\tmp"
_VOL = os.path.join(_TMP, "volsizing.py")
if not os.path.exists(_VOL):
    pytest.skip("experiment volsizing module not present", allow_module_level=True)

_spec = importlib.util.spec_from_file_location("experiment_volsizing", _VOL)
vol = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vol)

W = vol.VOL_WINDOW


def _close(n=200, seed=4):
    rng = np.random.default_rng(seed)
    ts = np.arange(n, dtype=np.int64) * (4 * 3_600_000) + 1_000_000_000_000
    c = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    return pd.Series(c, index=pd.Index(ts, name="ts"))


def test_perturbing_future_bars_leaves_T_vol_byte_identical():
    close = _close()
    base = vol.realized_vol_pit(close)
    # A T well past the warmup and before the end.
    valid = base.dropna().index
    T = valid[len(valid) // 2]
    v_before = base.loc[T]

    bumped = close.copy()
    bumped.loc[bumped.index > T] *= 1.5     # corrupt every bar strictly after T
    after = vol.realized_vol_pit(bumped)

    assert after.loc[T] == v_before or (np.isnan(after.loc[T]) and np.isnan(v_before))
    # Whole prefix up to and including T is unchanged.
    upto = base.index[base.index <= T]
    pd.testing.assert_series_equal(after.loc[upto], base.loc[upto])


def test_perturbing_a_recent_bar_does_change_T_vol():
    """Anti-vacuity: a bar inside T's trailing window moves T's vol."""
    close = _close()
    base = vol.realized_vol_pit(close)
    valid = base.dropna().index
    T = valid[len(valid) // 2]
    T_pos = close.index.get_loc(T)

    bumped = close.copy()
    # Perturb a bar a few steps before T (inside the trailing W window).
    inside = close.index[T_pos - 3]
    bumped.loc[inside] *= 1.10
    after = vol.realized_vol_pit(bumped)
    assert after.loc[T] != base.loc[T]


def test_vol_uses_exactly_the_trailing_window_no_centering():
    """Direct formula check: vol_T == sqrt(mean(r^2)) over the last W returns,
    with NO mean subtraction."""
    close = _close(n=80, seed=1)
    r = np.log(close / close.shift(1))
    out = vol.realized_vol_pit(close)
    valid = out.dropna().index
    T = valid[len(valid) // 2]
    T_pos = close.index.get_loc(T)
    window_rets = r.iloc[T_pos - W + 1:T_pos + 1].to_numpy()
    assert len(window_rets) == W
    expected = np.sqrt(np.mean(window_rets ** 2))   # no centering
    assert out.loc[T] == pytest.approx(expected)
    # And it is NOT the centered (std) version, in general.
    centered = np.sqrt(np.mean((window_rets - window_rets.mean()) ** 2))
    assert not np.isclose(out.loc[T], centered) or np.isclose(window_rets.mean(), 0.0)
