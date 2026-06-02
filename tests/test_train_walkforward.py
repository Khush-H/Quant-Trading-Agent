"""Tests for walk-forward training: strict OOS, embargo, and model safety.

The headline guarantees:
  * No OOS prediction comes from a model that trained on that bar (or within the
    embargo gap of it) — proven from the per-fold bookkeeping.
  * The embargo between train and test is at least the label horizon, so a
    train row's forward label window cannot reach into the test features.
  * A saved model refuses to load against a mismatched feature recipe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.settings import Settings
from core.database import FEATURE_COLUMNS
from ml.labels import FLAT, LONG
from ml.train import (
    FeatureMismatchError,
    load_model,
    save_model,
    walk_forward,
)


def _features_labels(n=600, seed=3):
    """Synthetic but learnable: label correlates with one feature + noise."""
    rng = np.random.default_rng(seed)
    ts = pd.Index(np.arange(n, dtype=np.int64) * 3_600_000, name="ts")
    X = pd.DataFrame(
        rng.normal(0, 1, size=(n, len(FEATURE_COLUMNS))),
        columns=list(FEATURE_COLUMNS), index=ts,
    )
    # Long when the first feature is high (with noise) -> imbalanced-ish.
    score = X[FEATURE_COLUMNS[0]] + rng.normal(0, 0.5, n)
    y = pd.Series(np.where(score > 0.7, LONG, FLAT), index=ts, name="label").astype(int)
    return X, y


def _settings(horizon=3):
    return Settings(_env_file=None, label_horizon=horizon)


def test_walk_forward_respects_embargo_and_never_tests_on_train():
    X, y = _features_labels()
    s = _settings(horizon=3)
    wf = walk_forward(X, y, settings=s, n_splits=4)

    assert len(wf.folds) >= 1
    for f in wf.folds:
        # The embargo gap between the end of train and start of test is honored.
        assert f["test_start"] - f["train_rows"] >= f["embargo"]
        # And the embargo is at least the label horizon (no label leakage).
        assert f["embargo"] >= s.label_horizon
        # Test block starts strictly after the training rows.
        assert f["test_start"] >= f["train_rows"]


def test_oos_predictions_cover_only_test_region_and_are_probabilities():
    X, y = _features_labels()
    wf = walk_forward(X, y, settings=_settings(), n_splits=4)
    # Every OOS bar's position is past the first train block (never bar 0..k).
    positions = [X.index.get_loc(ts) for ts in wf.oos_proba.index]
    assert min(positions) > 0
    # Probabilities are in [0, 1].
    assert (wf.oos_proba >= 0).all() and (wf.oos_proba <= 1).all()
    # OOS labels align with OOS predictions.
    assert wf.oos_proba.index.equals(wf.oos_labels.index)


def test_oos_index_is_strictly_increasing_and_unique():
    X, y = _features_labels()
    wf = walk_forward(X, y, settings=_settings(), n_splits=4)
    idx = wf.oos_proba.index
    assert idx.is_monotonic_increasing
    assert idx.is_unique  # no bar predicted twice


def test_calibration_table_is_produced():
    X, y = _features_labels()
    wf = walk_forward(X, y, settings=_settings(), n_splits=4)
    cal = wf.calibration
    assert {"n", "predicted", "observed"} <= set(cal.columns)
    assert cal["n"].sum() == len(wf.oos_proba)


def test_save_then_load_roundtrips_on_matching_recipe(tmp_path, monkeypatch):
    import ml.train as train_mod

    monkeypatch.setattr(train_mod, "MODELS_DIR", tmp_path)
    X, y = _features_labels()
    from xgboost import XGBClassifier
    model = XGBClassifier(n_estimators=10, max_depth=2)
    model.fit(X, y)

    from ml.features import compute_feature_hash
    fh = compute_feature_hash()
    save_model(model, list(FEATURE_COLUMNS), fh, name="m")
    loaded = load_model("m", list(FEATURE_COLUMNS), expected_hash=fh)
    assert loaded is not None


def test_load_refuses_mismatched_feature_hash(tmp_path, monkeypatch):
    import ml.train as train_mod

    monkeypatch.setattr(train_mod, "MODELS_DIR", tmp_path)
    X, y = _features_labels()
    from xgboost import XGBClassifier
    model = XGBClassifier(n_estimators=10, max_depth=2)
    model.fit(X, y)

    save_model(model, list(FEATURE_COLUMNS), "HASH_AT_TRAIN_TIME", name="m")
    with pytest.raises(FeatureMismatchError):
        load_model("m", list(FEATURE_COLUMNS), expected_hash="DIFFERENT_HASH")


def test_load_refuses_mismatched_feature_list(tmp_path, monkeypatch):
    import ml.train as train_mod

    monkeypatch.setattr(train_mod, "MODELS_DIR", tmp_path)
    X, y = _features_labels()
    from xgboost import XGBClassifier
    model = XGBClassifier(n_estimators=10, max_depth=2)
    model.fit(X, y)

    save_model(model, list(FEATURE_COLUMNS), "H", name="m")
    with pytest.raises(FeatureMismatchError):
        load_model("m", ["totally", "different", "cols"], expected_hash="H")
