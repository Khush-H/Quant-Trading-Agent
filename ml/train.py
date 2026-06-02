"""Walk-forward XGBoost training with strict out-of-sample evaluation.

The model exists only to be judged out-of-sample, so this module never reports
a score on data the model trained on. It produces a single concatenated vector
of OOS predictions via expanding-window walk-forward:

    [ train block 0 ][embargo][ test block 0 ]
    [    train block 0 + 1   ][embargo][ test block 1 ]
    ...

At each step the model trains on everything up to a cutoff, an EMBARGO gap of
``embargo`` bars is skipped, and it predicts the next test block. Predictions
from every test block are concatenated; no bar is ever predicted by a model
that saw it (or its label window) in training.

Why the embargo: a label at bar t looks ``label_horizon`` bars into the future.
Without a gap, the last few training rows' labels overlap the first test rows'
feature windows, leaking future information across the split. The embargo must
be at least ``label_horizon`` bars; we default to it.

Class imbalance (Flat dominates Long, typically) is handled with XGBoost's
``scale_pos_weight`` computed per-fold from the TRAIN labels only — NOT by
moving the label hurdle, which would change what we're predicting.

Artifacts are saved with a timestamp, the ordered feature list, and the recipe
``feature_hash``; :func:`load_model` refuses to return a model whose stored
feature hash/list does not match the features it is about to score.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import Settings, get_settings
from ml.features import compute_feature_hash
from ml.labels import FLAT, LONG

MODELS_DIR = Path(__file__).resolve().parent / "models"

# XGBoost params per the spec. Binary objective because labels are Flat/Long.
XGB_PARAMS = {
    "objective": "binary:logistic",
    "tree_method": "hist",
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.7,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "eval_metric": "logloss",
    "n_estimators": 300,
}


@dataclass
class WalkForwardResult:
    """Out-of-sample predictions plus calibration and per-fold bookkeeping."""

    oos_proba: pd.Series          # P(Long) for each OOS bar, indexed by ts
    oos_labels: pd.Series         # the true label for each OOS bar
    folds: list = field(default_factory=list)
    calibration: Optional[pd.DataFrame] = None
    feature_hash: str = ""

    def oos_signal(self, threshold: float = 0.5) -> pd.Series:
        """Turn OOS probabilities into a LONG/FLAT signal at ``threshold``."""
        return pd.Series(
            np.where(self.oos_proba >= threshold, LONG, FLAT),
            index=self.oos_proba.index,
            name="signal",
        ).astype(int)


def _scale_pos_weight(y: pd.Series) -> float:
    """negatives / positives on the TRAIN fold. Balances Flat-heavy data."""
    pos = int((y == LONG).sum())
    neg = int((y == FLAT).sum())
    if pos == 0:
        return 1.0
    return neg / pos


def _fit_one(X_train: pd.DataFrame, y_train: pd.Series):
    """Fit a single XGBClassifier on a train fold. Imported lazily."""
    from xgboost import XGBClassifier

    params = dict(XGB_PARAMS)
    params["scale_pos_weight"] = _scale_pos_weight(y_train)
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model


def walk_forward(
    features: pd.DataFrame,
    labels: pd.Series,
    settings: Optional[Settings] = None,
    *,
    n_splits: int = 5,
    embargo: Optional[int] = None,
    min_train: Optional[int] = None,
) -> WalkForwardResult:
    """Expanding-window walk-forward; returns concatenated OOS predictions.

    Args:
        features: causal feature matrix (rows = bars, cols = FEATURE_COLUMNS).
        labels: aligned Flat/Long labels.
        n_splits: number of sequential test blocks.
        embargo: bars skipped between train and test. Defaults to
            ``settings.label_horizon`` so a train label cannot overlap a test
            feature window.
        min_train: minimum bars in the first train block. Defaults to one block.

    There is intentionally no parameter that would evaluate on training data.
    """
    settings = settings or get_settings()
    embargo = settings.label_horizon if embargo is None else embargo

    # Align features and labels on common, sorted timestamps.
    common = features.index.intersection(labels.index)
    X = features.loc[common].sort_index()
    y = labels.loc[common].sort_index().astype(int)
    n = len(X)
    if n < (n_splits + 1) * 2:
        raise ValueError(
            f"Not enough samples ({n}) for {n_splits} walk-forward splits."
        )

    block = n // (n_splits + 1)  # first block is train-only; rest are tested
    if min_train is None:
        min_train = block

    feature_hash = compute_feature_hash()
    oos_proba_parts: list[pd.Series] = []
    oos_label_parts: list[pd.Series] = []
    folds: list[dict] = []

    for k in range(n_splits):
        test_start = min_train + k * block
        # Train uses everything up to the embargo gap before the test block.
        train_end = test_start - embargo
        test_end = min(test_start + block, n)
        if train_end <= 0 or test_start >= n or test_start >= test_end:
            continue

        X_tr, y_tr = X.iloc[:train_end], y.iloc[:train_end]
        X_te, y_te = X.iloc[test_start:test_end], y.iloc[test_start:test_end]
        # A fold with a single-class train set can't fit a useful classifier.
        if y_tr.nunique() < 2 or len(X_te) == 0:
            continue

        model = _fit_one(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]  # P(class == LONG)
        oos_proba_parts.append(pd.Series(proba, index=X_te.index))
        oos_label_parts.append(y_te)
        folds.append({
            "fold": k,
            "train_rows": int(train_end),
            "embargo": int(embargo),
            "test_start": int(test_start),
            "test_rows": int(len(X_te)),
            "scale_pos_weight": _scale_pos_weight(y_tr),
        })

    if not oos_proba_parts:
        raise RuntimeError("Walk-forward produced no OOS predictions.")

    oos_proba = pd.concat(oos_proba_parts).sort_index()
    oos_labels = pd.concat(oos_label_parts).sort_index()
    calibration = reliability_table(oos_proba, oos_labels)

    return WalkForwardResult(
        oos_proba=oos_proba,
        oos_labels=oos_labels,
        folds=folds,
        calibration=calibration,
        feature_hash=feature_hash,
    )


def reliability_table(proba: pd.Series, labels: pd.Series, bins: int = 10) -> pd.DataFrame:
    """Calibration / reliability check on OOS predictions.

    Buckets predicted P(Long) into ``bins`` and compares the mean predicted
    probability against the observed Long frequency in each bucket. A
    well-calibrated model has predicted ~ observed, which is what lets a
    confidence threshold mean something downstream.
    """
    df = pd.DataFrame({"p": proba.to_numpy(), "y": (labels == LONG).astype(int).to_numpy()})
    edges = np.linspace(0.0, 1.0, bins + 1)
    df["bucket"] = pd.cut(df["p"], bins=edges, include_lowest=True)
    grouped = df.groupby("bucket", observed=True)
    table = grouped.agg(
        n=("y", "size"),
        predicted=("p", "mean"),
        observed=("y", "mean"),
    ).reset_index(drop=True)
    return table


# --- persistence with feature-hash safety -------------------------------------

def save_model(model, feature_list: list[str], feature_hash: str,
               name: Optional[str] = None) -> Path:
    """Persist a fitted model with its feature list and recipe hash.

    The sidecar JSON records the timestamp, ordered feature list, and the
    recipe ``feature_hash`` so :func:`load_model` can refuse a mismatch.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = name or f"xgb_{ts}"
    model_path = MODELS_DIR / f"{name}.json"
    meta_path = MODELS_DIR / f"{name}.meta.json"
    model.save_model(str(model_path))
    meta_path.write_text(json.dumps({
        "saved_at": ts,
        "feature_list": list(feature_list),
        "feature_hash": feature_hash,
        "xgb_params": XGB_PARAMS,
    }, indent=2), encoding="utf-8")
    return model_path


class FeatureMismatchError(RuntimeError):
    """Raised when a model's stored features don't match what it's scoring."""


def load_model(name: str, expected_features: list[str],
               expected_hash: Optional[str] = None):
    """Load a saved model, refusing a feature/recipe mismatch.

    ``expected_hash`` defaults to the current recipe hash. If the stored hash
    or ordered feature list differs, raises :class:`FeatureMismatchError`
    instead of silently scoring features the model was not trained on.
    """
    from xgboost import XGBClassifier

    expected_hash = expected_hash or compute_feature_hash()
    model_path = MODELS_DIR / f"{name}.json"
    meta_path = MODELS_DIR / f"{name}.meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    if meta["feature_hash"] != expected_hash:
        raise FeatureMismatchError(
            f"Model {name} was trained with feature_hash={meta['feature_hash']!r} "
            f"but the current recipe is {expected_hash!r}. Refusing to load."
        )
    if list(meta["feature_list"]) != list(expected_features):
        raise FeatureMismatchError(
            f"Model {name} feature list {meta['feature_list']} does not match "
            f"the features to score {list(expected_features)}. Refusing to load."
        )

    model = XGBClassifier()
    model.load_model(str(model_path))
    return model
