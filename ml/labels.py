"""Label construction (features/labels phase).

Builds the supervised target aligned to the feature matrix. Labels look into
the FUTURE by definition; the contract is that the feature row at time ``t``
(computed from the closed candle ``t``) never sees the candles the label
measures. The label here is the N-period-forward log return of close,
thresholded against the full round-trip trading cost.

SPOT, long-only account — exactly TWO classes:

* ``1`` = Long  — forward return clears the cost hurdle; worth holding.
* ``0`` = Flat  — it does not; stay out / exit.

There is NO short class. The hurdle is the FULL round-trip cost (both legs)
plus per-leg slippage, exposed via :pyattr:`config.Settings.label_hurdle`
(``round_trip_cost + 2 * slippage_bps / 10_000``), so a label of 1 means the
move is expected to beat the cost of getting in AND out, not just one leg.

Alignment / no look-ahead
-------------------------
For row ``t`` the forward return is ``ln(close_{t+N} / close_t)``. The features
for row ``t`` come from candle ``t`` and earlier; the label's window is
``(t, t+N]`` — strictly AFTER ``t``. The two windows touch at the boundary
close_t but never overlap, and the feature window never includes ``t+1``. Rows
whose forward window runs off the end of the data are dropped (NaN), never
filled.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config import Settings, get_settings

logger = logging.getLogger(__name__)

# Class codes. Spot, long-only: two classes, no short.
FLAT = 0
LONG = 1

# Warn if either class is rarer than this share of samples — a degenerate
# balance makes the downstream classifier near-useless.
MIN_CLASS_SHARE = 0.15


def forward_log_return(close: pd.Series, horizon: int) -> pd.Series:
    """``ln(close_{t+N} / close_t)`` for each t. Uses shift(-N) (the future).

    The tail N rows have no future bar and come back NaN; callers drop them.
    """
    if horizon <= 0:
        raise ValueError(f"horizon must be >= 1, got {horizon!r}")
    future_close = close.shift(-horizon)
    return np.log(future_close / close)


def build_labels(
    ohlcv: pd.DataFrame,
    settings: Optional[Settings] = None,
    horizon: Optional[int] = None,
) -> pd.Series:
    """Return the long/flat target Series indexed like ``ohlcv``.

    ``horizon`` defaults to ``settings.label_horizon``. A row is ``LONG`` (1)
    when its N-period forward log return exceeds ``settings.label_hurdle``,
    else ``FLAT`` (0). Rows whose forward window runs past the data end are
    dropped. Logs the class balance and warns on a degenerate split.
    """
    settings = settings or get_settings()
    n = settings.label_horizon if horizon is None else horizon
    hurdle = settings.label_hurdle

    if not ohlcv.index.is_monotonic_increasing:
        ohlcv = ohlcv.sort_index()

    fwd = forward_log_return(ohlcv["close"], n)
    fwd = fwd.dropna()  # drop the tail rows with no full forward window

    labels = pd.Series(
        np.where(fwd > hurdle, LONG, FLAT), index=fwd.index, name="label"
    ).astype(int)

    _report_class_balance(labels, hurdle=hurdle, horizon=n)
    return labels


def _report_class_balance(labels: pd.Series, hurdle: float, horizon: int) -> None:
    """Print and log the Long/Flat balance; warn if either class < 15%."""
    total = len(labels)
    if total == 0:
        logger.warning("No labels produced (empty after dropping forward tail).")
        print("Label class balance: 0 samples.")
        return

    n_long = int((labels == LONG).sum())
    n_flat = int((labels == FLAT).sum())
    share_long = n_long / total
    share_flat = n_flat / total

    print(
        f"Label class balance (N={horizon}, hurdle={hurdle:.4%}, {total} samples):\n"
        f"  Long (1): {n_long:>7} ({share_long:6.2%})\n"
        f"  Flat (0): {n_flat:>7} ({share_flat:6.2%})"
    )

    for name, share in (("Long", share_long), ("Flat", share_flat)):
        if share < MIN_CLASS_SHARE:
            msg = (
                f"Class imbalance: {name} is only {share:.2%} of samples "
                f"(< {MIN_CLASS_SHARE:.0%}). The classifier may degenerate; "
                "consider a longer horizon or a lower hurdle."
            )
            logger.warning(msg)
            print(f"  WARNING: {msg}")
