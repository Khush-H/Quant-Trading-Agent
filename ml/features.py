"""Feature engineering.

Transforms raw OHLCV (and later order-book/alt data) into the model's feature
matrix. Keep features causal — only information available at decision time may
enter a row, or the backtest will be optimistic. Implemented in the
features/labels phase.
"""

from __future__ import annotations

import pandas as pd


def build_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Return a feature matrix indexed like the input.

    Must be strictly causal: no look-ahead. Add indicators (returns, vol,
    momentum, etc.) here during the build.
    """
    raise NotImplementedError("Features implemented in features/labels phase.")
