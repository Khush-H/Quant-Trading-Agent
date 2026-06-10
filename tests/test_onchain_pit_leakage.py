"""REQUIRED leakage test: on-chain AdrActCnt D-1 point-in-time lag (Experiment 6).

The on-chain analogue of the OHLCV no-overlap and settled-funding PIT proofs.
The pre-registered rule under test:

    A feature row for trading day D may use ONLY AdrActCnt values dated D-1 or
    earlier. CoinMetrics publishes day D's count at end of day D UTC, so day D
    is not available at trading open on day D.

Proven four ways on a 100-row sample:

  (1) Perturbation invariance: for EVERY sampled row D, corrupting every
      AdrActCnt value dated D or LATER leaves D's three on-chain features
      byte-identical.
  (2) Literal recompute: each row's adr_zscore_28d / adr_mom_7d /
      adr_price_diverge_28d equals a by-hand computation that is only ever
      handed values dated D-1 or earlier.
  (3) Anti-vacuity: bumping the single value dated D-1 DOES move row D — the
      features genuinely depend on the point-in-time value, so (1) is not
      vacuous.
  (4) Injection: a deliberately leaky builder that feeds row D the value dated
      D (instead of D-1) makes the rule check FAIL — the test has the power to
      catch the exact leak it guards against.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.onchain.features import (
    ADR_MOM_LAG,
    ADR_Z_WINDOW,
    ONCHAIN_FEATURE_COLUMNS,
    add_onchain_features,
)

_DAY = pd.Timedelta(days=1)
_DAY_MS = 24 * 60 * 60 * 1000
_SAMPLE_ROWS = 100

# Sized so >= 100 rows sit past the 28-day warmup: 200 candles, with the
# on-chain series starting 40 days earlier and running 10 days past the end
# (so even the LAST row has future-dated values to perturb).
_N_CANDLES = 200
_ADR_LEAD_DAYS = 40
_ADR_TRAIL_DAYS = 10
_FIRST_CANDLE = pd.Timestamp("2021-01-01")


def _make_ohlcv(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = _FIRST_CANDLE.value // 10**6 + np.arange(_N_CANDLES, dtype=np.int64) * _DAY_MS
    close = 30_000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.03, _N_CANDLES)))
    spread = np.abs(rng.normal(0.0, 0.01, _N_CANDLES))
    return pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0.0, 0.005, _N_CANDLES)),
            "high": close * (1 + spread),
            "low": close * (1 - spread),
            "close": close,
            "volume": rng.uniform(1e3, 5e4, _N_CANDLES),
        },
        index=pd.Index(ts, name="ts"),
    )


def _make_adr(seed: int = 13) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(
        _FIRST_CANDLE - _ADR_LEAD_DAYS * _DAY,
        _FIRST_CANDLE + (_N_CANDLES - 1 + _ADR_TRAIL_DAYS) * _DAY,
        freq="D",
    )
    # Strictly distinct values so ANY swap or shift is detectable.
    values = 900_000.0 + np.arange(len(dates)) * 17.0 + rng.uniform(0, 5, len(dates))
    return pd.DataFrame({"date": dates, "AdrActCnt": values})


def _date_of(row_ts: int) -> pd.Timestamp:
    return pd.Timestamp(int(row_ts), unit="ms").normalize()


def _sampled_rows(ohlcv: pd.DataFrame) -> list[int]:
    """100 deterministic row timestamps past the 28-day warmup."""
    positions = np.linspace(ADR_Z_WINDOW, _N_CANDLES - 1, _SAMPLE_ROWS).astype(int)
    assert len(set(positions)) == _SAMPLE_ROWS
    return [int(ohlcv.index[p]) for p in positions]


def _corrupt_on_or_after(adr: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    out = adr.copy()
    mask = out["date"] >= cutoff
    assert mask.any(), "perturbation must touch at least one value to be real"
    out.loc[mask, "AdrActCnt"] = out.loc[mask, "AdrActCnt"] + 9.9e9
    return out


def _assert_row_uses_only_past(builder, ohlcv, adr, row_ts: int) -> None:
    """The rule check: corrupting values dated >= D must not move row D."""
    base = builder(ohlcv, adr)
    corrupted = _corrupt_on_or_after(adr, _date_of(row_ts))
    after = builder(ohlcv, corrupted)
    pd.testing.assert_series_equal(after.loc[row_ts], base.loc[row_ts])


def test_each_sampled_row_ignores_adr_dated_D_or_later():
    """(1) For all 100 sampled rows: values dated D or later never matter."""
    ohlcv, adr = _make_ohlcv(), _make_adr()
    for row_ts in _sampled_rows(ohlcv):
        _assert_row_uses_only_past(add_onchain_features, ohlcv, adr, row_ts)


def test_full_prefix_unchanged_when_future_is_corrupted():
    """(1b) Stronger, for a mid row D: the ENTIRE prefix <= D is unchanged."""
    ohlcv, adr = _make_ohlcv(), _make_adr()
    base = add_onchain_features(ohlcv, adr)
    row_ts = int(ohlcv.index[_N_CANDLES // 2])
    after = add_onchain_features(
        ohlcv, _corrupt_on_or_after(adr, _date_of(row_ts))
    )
    prefix = base.index[base.index <= row_ts]
    pd.testing.assert_frame_equal(after.loc[prefix], base.loc[prefix])


def test_features_match_hand_computation_from_D_minus_1_or_earlier():
    """(2) Literal check: each feature value equals a recompute that is only
    ever handed AdrActCnt dated D-1 or earlier (and closes through D)."""
    ohlcv, adr = _make_ohlcv(), _make_adr()
    feats = add_onchain_features(ohlcv, adr)
    adr_by_date = adr.set_index("date")["AdrActCnt"].astype(float)
    close = ohlcv["close"].astype(float)

    for row_ts in _sampled_rows(ohlcv):
        d = _date_of(row_ts)
        # The ONLY on-chain values handed to the recompute: dated <= D-1.
        window = adr_by_date.loc[d - ADR_Z_WINDOW * _DAY: d - _DAY]
        assert len(window) == ADR_Z_WINDOW
        assert window.index.max() <= d - _DAY  # nothing dated D or later
        mean, std = window.mean(), window.std(ddof=0)
        expected_z = (adr_by_date.loc[d - _DAY] - mean) / std

        expected_mom = (
            adr_by_date.loc[d - _DAY] / adr_by_date.loc[d - (ADR_MOM_LAG + 1) * _DAY]
        ) - 1.0

        pos = ohlcv.index.get_loc(row_ts)
        cwin = close.iloc[pos - ADR_Z_WINDOW + 1: pos + 1]
        price_z = (close.iloc[pos] - cwin.mean()) / cwin.std(ddof=0)

        assert feats.loc[row_ts, "adr_zscore_28d"] == pytest.approx(expected_z)
        assert feats.loc[row_ts, "adr_mom_7d"] == pytest.approx(expected_mom)
        assert feats.loc[row_ts, "adr_price_diverge_28d"] == pytest.approx(
            expected_z - price_z
        )


def test_bumping_the_D_minus_1_value_changes_row_D():
    """(3) Anti-vacuity: row D genuinely depends on the value dated D-1."""
    ohlcv, adr = _make_ohlcv(), _make_adr()
    base = add_onchain_features(ohlcv, adr)
    row_ts = int(ohlcv.index[_N_CANDLES // 2])
    d = _date_of(row_ts)

    bumped = adr.copy()
    mask = bumped["date"] == d - _DAY
    assert mask.sum() == 1
    bumped.loc[mask, "AdrActCnt"] += 250_000.0
    after = add_onchain_features(ohlcv, bumped)

    for col in ONCHAIN_FEATURE_COLUMNS:
        assert after.loc[row_ts, col] != base.loc[row_ts, col], col


def test_injected_future_value_is_caught_by_the_rule_check():
    """(4) Deliberate leak: a builder that feeds row D the value dated D
    (instead of D-1) must make the rule check fail. Proves the test's power."""

    def leaky_builder(ohlcv: pd.DataFrame, adr: pd.DataFrame) -> pd.DataFrame:
        # Re-dating every measurement one day EARLIER makes the D-1 sampling
        # inside add_onchain_features read the true day-D value: exactly the
        # "use AdrActCnt[D] instead of AdrActCnt[D-1]" leak.
        return add_onchain_features(ohlcv, adr.assign(date=adr["date"] - _DAY))

    ohlcv, adr = _make_ohlcv(), _make_adr()
    row_ts = int(ohlcv.index[_N_CANDLES // 2])
    with pytest.raises(AssertionError):
        _assert_row_uses_only_past(leaky_builder, ohlcv, adr, row_ts)
