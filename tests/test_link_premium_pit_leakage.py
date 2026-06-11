"""REQUIRED leakage test: LINK-premium T-inclusive PIT rule (Experiment 9).

The daily-resolution analogue of ``test_coinbase_premium_pit_leakage.py``.
The pre-registered rule:

    A feature row for the 1d bar at T uses ONLY coinbase_premium values dated
    T or earlier. The premium at T comes from both exchanges' CLOSED daily
    bars at T, so T itself is legitimately usable; what must be proven is
    that the rolling windows (168-DAY z-score, 24-DAY momentum — same bar
    counts as Experiment 8, daily bars) never reach past T.

Proven four ways on a 100-row sample:

  (1) Perturbation invariance: for EVERY sampled row T, corrupting every
      premium value dated STRICTLY AFTER T leaves all three features at row T
      byte-identical.
  (2) Literal recompute: each row's coinbase_premium / premium_zscore_168h /
      premium_mom_24h equals a by-hand computation that is only ever handed
      values dated T or earlier.
  (3) Anti-vacuity: bumping the premium dated exactly T moves ALL THREE
      features at row T — they genuinely consume the data, so (1) is not
      vacuous.
  (4) Injection: a deliberately leaky builder that feeds row T the value dated
      T+1 makes the rule check FAIL — the test can catch the exact leak it
      guards against.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.onchain.premium_features import (
    PREMIUM_FEATURE_COLUMNS,
    PREMIUM_MOM_LAG,
    PREMIUM_Z_WINDOW,
    add_premium_features,
)

_DAY = pd.Timedelta(days=1)
_DAY_MS = 24 * 60 * 60 * 1000
_SAMPLE_ROWS = 100

# Sized so >= 100 rows sit past the 168d z-warmup + 24d momentum lag: 500
# daily candles, premium covering 200 days of lead and 30 days of trail
# (so even the LAST row has strictly-future values to perturb).
_N_BARS = 500
_PREMIUM_LEAD_D = 200
_PREMIUM_TRAIL_D = 30
_WARMUP = PREMIUM_Z_WINDOW + PREMIUM_MOM_LAG  # 192
_FIRST_BAR = pd.Timestamp("2021-03-01 00:00:00")


def _make_ohlcv(seed: int = 17) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = _FIRST_BAR.value // 10**6 + np.arange(_N_BARS, dtype=np.int64) * _DAY_MS
    close = 15.0 * np.exp(np.cumsum(rng.normal(0.0, 0.03, _N_BARS)))
    spread = np.abs(rng.normal(0.0, 0.01, _N_BARS))
    return pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0.0, 0.005, _N_BARS)),
            "high": close * (1 + spread),
            "low": close * (1 - spread),
            "close": close,
            "volume": rng.uniform(10_000, 500_000, _N_BARS),
        },
        index=pd.Index(ts, name="ts"),
    )


def _make_premium(seed: int = 23) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    times = pd.date_range(
        _FIRST_BAR - _PREMIUM_LEAD_D * _DAY,
        _FIRST_BAR + (_N_BARS - 1 + _PREMIUM_TRAIL_D) * _DAY,
        freq="D",
    )
    # Premium-like values, all nonzero and distinct so any swap is detectable
    # and no momentum denominator is exactly zero.
    values = 0.05 + rng.normal(0.0, 0.30, len(times))
    values = np.where(np.abs(values) < 1e-6, 1e-3, values)
    return pd.DataFrame({"timestamp_utc": times, "coinbase_premium": values})


def _time_of(row_ts: int) -> pd.Timestamp:
    return pd.Timestamp(int(row_ts), unit="ms")


def _sampled_rows(ohlcv: pd.DataFrame) -> list[int]:
    """100 deterministic row timestamps past the 192-bar warmup."""
    positions = np.linspace(_WARMUP, len(ohlcv) - 1, _SAMPLE_ROWS).astype(int)
    assert len(set(positions)) == _SAMPLE_ROWS
    return [int(ohlcv.index[p]) for p in positions]


def _corrupt_strictly_after(premium: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    out = premium.copy()
    mask = out["timestamp_utc"] > cutoff
    assert mask.any(), "perturbation must touch at least one value to be real"
    out.loc[mask, "coinbase_premium"] = out.loc[mask, "coinbase_premium"] + 999.0
    return out


def _assert_row_uses_only_T_or_earlier(builder, ohlcv, premium, row_ts: int) -> None:
    """The rule check: corrupting values dated > T must not move row T."""
    base = builder(ohlcv, premium)
    corrupted = _corrupt_strictly_after(premium, _time_of(row_ts))
    after = builder(ohlcv, corrupted)
    pd.testing.assert_series_equal(after.loc[row_ts], base.loc[row_ts])


def test_each_sampled_row_ignores_premium_dated_after_T():
    """(1) For all 100 sampled rows: values dated strictly after T never
    matter."""
    ohlcv, premium = _make_ohlcv(), _make_premium()
    for row_ts in _sampled_rows(ohlcv):
        _assert_row_uses_only_T_or_earlier(
            add_premium_features, ohlcv, premium, row_ts
        )


def test_full_prefix_unchanged_when_future_is_corrupted():
    """(1b) Stronger, for a mid row T: the ENTIRE prefix <= T is unchanged."""
    ohlcv, premium = _make_ohlcv(), _make_premium()
    base = add_premium_features(ohlcv, premium)
    row_ts = int(ohlcv.index[_N_BARS // 2])
    after = add_premium_features(
        ohlcv, _corrupt_strictly_after(premium, _time_of(row_ts))
    )
    prefix = base.index[base.index <= row_ts]
    pd.testing.assert_frame_equal(after.loc[prefix], base.loc[prefix])


def test_features_match_hand_computation_from_T_or_earlier():
    """(2) Literal check: each feature value equals a recompute that is only
    ever handed premium values dated T or earlier — windows measured in DAYS."""
    ohlcv, premium = _make_ohlcv(), _make_premium()
    feats = add_premium_features(ohlcv, premium)
    by_time = premium.set_index("timestamp_utc")["coinbase_premium"].astype(float)

    for row_ts in _sampled_rows(ohlcv):
        t = _time_of(row_ts)
        # The ONLY values handed to the recompute: dated <= T.
        window = by_time.loc[t - (PREMIUM_Z_WINDOW - 1) * _DAY: t]
        assert len(window) == PREMIUM_Z_WINDOW
        assert window.index.max() <= t  # nothing dated after T
        mean, std = window.mean(), window.std(ddof=0)
        expected_z = (by_time.loc[t] - mean) / std if std > 0 else 0.0

        denom = by_time.loc[t - PREMIUM_MOM_LAG * _DAY]
        expected_mom = (by_time.loc[t] / denom - 1.0) if denom != 0 else 0.0

        assert feats.loc[row_ts, "coinbase_premium"] == pytest.approx(
            by_time.loc[t]
        )
        assert feats.loc[row_ts, "premium_zscore_168h"] == pytest.approx(expected_z)
        assert feats.loc[row_ts, "premium_mom_24h"] == pytest.approx(expected_mom)


def test_bumping_premium_at_T_changes_all_three_features():
    """(3) Anti-vacuity: row T genuinely consumes the value dated T — all
    three features move when it moves."""
    ohlcv, premium = _make_ohlcv(), _make_premium()
    base = add_premium_features(ohlcv, premium)
    row_ts = int(ohlcv.index[_N_BARS // 2])
    t = _time_of(row_ts)

    bumped = premium.copy()
    mask = bumped["timestamp_utc"] == t
    assert mask.sum() == 1
    bumped.loc[mask, "coinbase_premium"] += 5.0
    after = add_premium_features(ohlcv, bumped)

    for col in PREMIUM_FEATURE_COLUMNS:
        assert after.loc[row_ts, col] != base.loc[row_ts, col], col


def test_injected_T_plus_1_value_is_caught_by_the_rule_check():
    """(4) Deliberate leak: a builder that feeds row T the value dated T+1
    must make the rule check fail. Proves the test's power."""

    def leaky_builder(ohlcv: pd.DataFrame, premium: pd.DataFrame) -> pd.DataFrame:
        # Re-dating every value one DAY earlier makes the T sampling inside
        # add_premium_features read the true T+1 value: exactly the
        # "use premium[T+1] for row T" leak.
        return add_premium_features(
            ohlcv, premium.assign(timestamp_utc=premium["timestamp_utc"] - _DAY)
        )

    ohlcv, premium = _make_ohlcv(), _make_premium()
    row_ts = int(ohlcv.index[_N_BARS // 2])
    with pytest.raises(AssertionError):
        _assert_row_uses_only_T_or_earlier(leaky_builder, ohlcv, premium, row_ts)


def test_resolution_mismatch_is_refused():
    """Guard: handing an HOURLY premium frame to daily candles (or any mixed
    resolution) raises instead of silently computing hour-based windows."""
    ohlcv = _make_ohlcv()
    hourly_times = pd.date_range(_FIRST_BAR, periods=400, freq="h")
    hourly_premium = pd.DataFrame(
        {"timestamp_utc": hourly_times,
         "coinbase_premium": np.linspace(0.01, 0.4, len(hourly_times))}
    )
    with pytest.raises(ValueError, match="resolution mismatch"):
        add_premium_features(ohlcv, hourly_premium)
