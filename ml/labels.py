"""Label construction.

Builds supervised targets (e.g. forward returns, triple-barrier labels) aligned
to the feature matrix. Labels look into the FUTURE by definition; the contract
is that features at time t never see the label's future window. Implemented in
the features/labels phase.
"""

from __future__ import annotations

import pandas as pd


def build_labels(ohlcv: pd.DataFrame) -> pd.Series:
    """Return a target Series aligned to the feature index.

    Be explicit about the horizon and ensure rows whose future window extends
    past the data are dropped, not silently filled.
    """
    raise NotImplementedError("Labels implemented in features/labels phase.")
