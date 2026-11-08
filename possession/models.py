"""Per-frame probabilities: is the ball dead, and if not, does the home side have it.

Two logistic regressions on possession.features, fitted on the practice ground (degraded IDSSE,
labels aligned with the challenge) and nowhere else. Logistic regression rather than something
deeper: the 33 position features already carry the signal (scripts/feature_report.py), the model
has to transfer from seven Bundesliga matches to one LaLiga match, and a linear model is the least
able to memorise its training matches. It also yields probabilities the decoder can weigh as
evidence.

scikit-learn fits; the fitted model is a JSON of means, scales and coefficients, and applying it
needs numpy alone (decision of 2026-11-08). The linear score is summed column by column in a
fixed order rather than through a matrix product, which BLAS may sum in a CPU-dependent order:
the submission has to come out byte for byte on the organisers' machine.

The team model sees every half twice, once with home and away exchanged
(possession.features.swapped), so it treats both sides alike instead of learning whichever side
the practice matches happened to favour.
"""

import json

import numpy as np

from possession.features import FPS, NAMES, rolling_mean

# Skewed, non-negative features modelled on a log scale.
LOG1P = ("ball_speed", "speed_all", "speed_home", "speed_away", "near_home", "near_away",
         "crowd2_home", "crowd2_away", "crowd5_home", "crowd5_away", "players_in_boxes")
# (feature, seconds): centred rolling means that give a frame the context of its neighbours.
CONTEXT = (("speed_all", 5.0), ("ball_seen", 5.0), ("ball_speed", 2.5), ("near_diff", 2.5),
           ("width_home", 5.0), ("width_away", 5.0), ("forward_home", 5.0),
           ("forward_away", 5.0), ("ball_velocity_home", 2.5), ("crowd2_home", 2.5),
           ("crowd2_away", 2.5), ("advance_home", 5.0), ("advance_away", 5.0))
DESIGN_NAMES = NAMES + tuple("{}_ctx{:g}s".format(name, seconds) for name, seconds in CONTEXT)


def design(features):
    """(frames, len(DESIGN_NAMES)) float64: log1p on skewed features, then context means."""
    X = features.astype(np.float64)
    index = {name: i for i, name in enumerate(NAMES)}
    for name in LOG1P:
        X[:, index[name]] = np.log1p(np.maximum(X[:, index[name]], 0.0))
    context = [rolling_mean(X[:, index[name]], int(seconds * FPS / 2))
               for name, seconds in CONTEXT]
    return np.column_stack([X] + context)


def fit(X, y, C=1.0):
    """A JSON-ready logistic regression: NaN imputed by the training mean, then standardised."""
    from sklearn.linear_model import LogisticRegression

    mean = np.nanmean(X, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    X = np.where(np.isnan(X), mean, X)
    scale = X.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    model = LogisticRegression(C=C, max_iter=5000)
    model.fit((X - mean) / scale, y)
    return {"names": list(DESIGN_NAMES), "C": C, "mean": mean.tolist(), "scale": scale.tolist(),
            "coef": model.coef_[0].tolist(), "intercept": float(model.intercept_[0])}


def score(model, X):
    """Linear score, summed column by column in a fixed order (no BLAS)."""
    z = np.full(len(X), model["intercept"], dtype=np.float64)
    for j, (mean, scale, coef) in enumerate(zip(model["mean"], model["scale"], model["coef"])):
        column = X[:, j]
        z += coef * ((np.where(np.isnan(column), mean, column) - mean) / scale)
    return z


def probability(model, X):
    """P(positive class) per row, numpy only."""
    z = score(model, X)
    positive = z >= 0
    out = np.empty_like(z)
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    e = np.exp(z[~positive])
    out[~positive] = e / (1.0 + e)
    return out


def state_probabilities(models, X):
    """(frames, 3) probabilities of H, A, D from the dead-ball and team models."""
    dead = probability(models["dead"], X)
    home = probability(models["team"], X)
    return np.column_stack([(1 - dead) * home, (1 - dead) * (1 - home), dead])


def save(models, path):
    with open(path, "w", newline="\n") as handle:
        json.dump(models, handle, indent=1, sort_keys=True)
        handle.write("\n")


def load(path):
    with open(path) as handle:
        models = json.load(handle)
    for name in ("dead", "team"):
        if tuple(models[name]["names"]) != DESIGN_NAMES:
            raise ValueError("{} model was fitted on other features than this code computes"
                             .format(name))
    return models
