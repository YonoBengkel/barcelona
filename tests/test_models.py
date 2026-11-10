"""Model design, fitting and the numpy inference that the submission relies on."""

import numpy as np
import pytest

from possession.features import ALL_NAMES, EXTRA_NAMES, NAMES
from possession.models import DESIGN_NAMES, SHORT_CONTEXT, design, design_names, fit, \
    fit_trees, load, log_sigmoid, log_state_probabilities, probability, save, score, \
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


def test_design_v3_adds_the_last_touch_columns_and_v1_ignores_them():
    features = np.ones((50, len(ALL_NAMES)), dtype=np.float32)
    X1, X3 = design(features), design(features, "v3")
    assert X1.shape[1] == len(design_names("v1"))
    assert X3.shape[1] == len(design_names("v3")) == X1.shape[1] + len(EXTRA_NAMES)
    assert design_names("v3")[len(NAMES):len(ALL_NAMES)] == EXTRA_NAMES


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


def test_log_evidence_matches_the_probabilities_and_stays_finite():
    X, y = synthetic()
    models = {"dead": fit(X, y), "team": fit(X, ~y)}
    log_p = log_state_probabilities(models, X)
    assert np.allclose(np.exp(log_p), state_probabilities(models, X), atol=1e-12)
    assert np.isfinite(log_sigmoid(np.array([-800.0, 0.0, 800.0]))).all()


def test_tree_inference_matches_scikit_learn():
    from sklearn.ensemble import HistGradientBoostingClassifier

    X, y = synthetic()  # with missing values, which the trees route on their own
    params = {"max_iter": 30, "max_leaf_nodes": 15, "learning_rate": 0.2, "min_samples_leaf": 10}
    model = fit_trees(X, y, **params)
    reference = HistGradientBoostingClassifier(early_stopping=False, random_state=0,
                                               **params).fit(X, y).decision_function(X)
    assert np.allclose(score(model, X), reference, atol=1e-9)


def test_a_tree_team_model_round_trips_and_gives_probabilities(tmp_path):
    X, y = synthetic()
    models = {"dead": fit(X, y), "team": fit_trees(X, ~y, max_iter=20)}
    path = str(tmp_path / "trees.json")
    save(models, path)
    loaded = load(path)
    assert loaded["team"]["kind"] == "trees"
    assert np.allclose(probability(loaded["team"], X), probability(models["team"], X))
    assert np.allclose(state_probabilities(loaded, X).sum(axis=1), 1.0)


def test_the_second_design_adds_short_context_and_loads_by_name(tmp_path):
    X1 = design(np.ones((50, len(NAMES)), dtype=np.float32))
    X2 = design(np.ones((50, len(NAMES)), dtype=np.float32), "v2")
    assert X2.shape[1] == len(design_names("v2")) == X1.shape[1] + len(SHORT_CONTEXT)
    assert np.array_equal(X2[:, :X1.shape[1]], X1)  # v2 only appends columns
    rng = np.random.default_rng(2)
    X = rng.normal(size=(400, len(design_names("v2"))))
    y = X[:, 0] > 0
    models = {"design": "v2", "dead": fit(X, y, version="v2"), "team": fit(X, ~y, version="v2")}
    save(models, str(tmp_path / "v2.json"))
    assert load(str(tmp_path / "v2.json"))["design"] == "v2"
    models["design"] = "v1"
    save(models, str(tmp_path / "wrong.json"))
    with pytest.raises(ValueError):
        load(str(tmp_path / "wrong.json"))


def test_loading_refuses_a_model_fitted_on_other_features(tmp_path):
    X, y = synthetic()
    model = fit(X, y)
    model["names"] = model["names"][:-1]
    save({"dead": model, "team": model}, str(tmp_path / "bad.json"))
    with pytest.raises(ValueError):
        load(str(tmp_path / "bad.json"))
