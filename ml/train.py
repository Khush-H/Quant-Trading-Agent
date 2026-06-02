"""Model training.

Trains the XGBoost model on features+labels with time-aware (walk-forward)
validation and writes the artifact to ``ml/models/``. Implemented in the train
phase, which comes AFTER a working backtest so the model is evaluated against
realistic costs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from config import Settings, get_settings

MODELS_DIR = Path(__file__).resolve().parent / "models"


def train(
    features: pd.DataFrame,
    labels: pd.Series,
    settings: Optional[Settings] = None,
) -> Path:
    """Train and persist a model; return the artifact path.

    Use walk-forward / purged splits — never a random shuffle on time series.
    """
    settings = settings or get_settings()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError("Training implemented in train phase.")
