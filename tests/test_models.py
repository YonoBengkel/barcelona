"""Model design, fitting and the numpy inference that the submission relies on."""

import numpy as np
import pytest

from possession.features import NAMES
from possession.models import DESIGN_NAMES, design, fit, load, probability, save, \
    state_probabilities


def synthetic(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(DESIGN_NAMES)))
    y = (X[:, 0] - 0.5 * X[:, 3] + rng.normal(scale=0.5, size=n)) > 0
    X[rng.random(X.shape) < 0.05] = np.nan
    return X, y


def test_design_adds_the_context_columns():
    X = design(np.ones((50, len(NAMES)), dtype=np.float32))
    assert X.shape == (50, len(DESIGN_NAMES))


def test_numpy_inference_matches_scikit_learn():
    from sklearn.linear_model import LogisticRegression

    X, y = synthetic()
    model = fit(X, y, C=0.5)
    mean, scale = np.array(model["mean"]), np.array(model["scale"])
    Z = (np.where(np.isnan(X), mean, X) - mean) / scale
    reference = LogisticRegression(C=0.5, max_iter=5000).fit(Z, y).predict_proba(Z)[:, 1]
    assert np.allclose(probability(model, X), reference, atol=1e-9)


def test_three_states_sum_to_one(tmp_path):
    X, y = synthetic()
    models = {"dead": fit(X, y), "team": fit(X, ~y)}
    path = tmp_path / "models.json"
    save(models, str(path))
    p = state_probabilities(load(str(path)), X)
    assert p.shape == (len(X), 3) and np.allclose(p.sum(axis=1), 1.0)


def test_loading_refuses_a_model_fitted_on_other_features(tmp_path):
    X, y = synthetic()
    model = fit(X, y)
    model["names"] = model["names"][:-1]
    save({"dead": model, "team": model}, str(tmp_path / "bad.json"))
    with pytest.raises(ValueError):
        load(str(tmp_path / "bad.json"))
