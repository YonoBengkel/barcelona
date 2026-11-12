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

M11d adds a second kind of model, gradient-boosted trees (fit_trees), for the team model above
all: the perfect-evidence check showed the team call holds most of the missing score, and trees
can use interactions a linear score cannot. They are kept as arrays and applied with numpy alone
as well (score_trees).

M11l tries two more ways to the same two log-odds (scores): a temporal convolutional network
(possession.tcn), alone or averaged with the trees, and a stage two of trees that reads the first
stage's log-odds around each frame (possession.stacking). Both are applied with numpy alone too.
A file that holds only the trees gives exactly the numbers it gave before.
"""

import json

import numpy as np

from possession.features import ALL_NAMES, FPS, NAMES, rolling_mean

# Skewed, non-negative features modelled on a log scale.
LOG1P = ("ball_speed", "speed_all", "speed_home", "speed_away", "near_home", "near_away",
         "crowd2_home", "crowd2_away", "crowd5_home", "crowd5_away", "players_in_boxes")
# (feature, seconds): centred rolling means that give a frame the context of its neighbours.
CONTEXT = (("speed_all", 5.0), ("ball_seen", 5.0), ("ball_speed", 2.5), ("near_diff", 2.5),
           ("width_home", 5.0), ("width_away", 5.0), ("forward_home", 5.0),
           ("forward_away", 5.0), ("ball_velocity_home", 2.5), ("crowd2_home", 2.5),
           ("crowd2_away", 2.5), ("advance_home", 5.0), ("advance_away", 5.0))
# M11a: one-second context beside the longer one, so that a possession or a stoppage of a second
# or two is not averaged away (possession.diagnose: 73% of the missed possessions went to the
# wrong team or were merged with a neighbour, most of them short).
SHORT_CONTEXT = (("near_diff", 1.0), ("near_home", 1.0), ("near_away", 1.0),
                 ("crowd2_home", 1.0), ("crowd2_away", 1.0), ("ball_seen", 1.0))
DESIGNS = {"v1": CONTEXT, "v2": CONTEXT + SHORT_CONTEXT, "v3": CONTEXT}
# The feature columns each design starts from. M11i: v3 adds the last-touch features
# (possession.features.EXTRA_NAMES), which compute() appends after NAMES.
RAW = {"v1": NAMES, "v2": NAMES, "v3": ALL_NAMES}


def design_names(version="v1"):
    """The columns design() makes for one version of the design."""
    return RAW[version] + tuple("{}_ctx{:g}s".format(name, seconds)
                                for name, seconds in DESIGNS[version])


DESIGN_NAMES = design_names("v1")


def design(features, version="v1"):
    """(frames, len(design_names(version))) float64: the design's feature columns, log1p on
    skewed features, then context means."""
    X = features[:, :len(RAW[version])].astype(np.float64)
    index = {name: i for i, name in enumerate(NAMES)}
    for name in LOG1P:
        X[:, index[name]] = np.log1p(np.maximum(X[:, index[name]], 0.0))
    context = [rolling_mean(X[:, index[name]], int(seconds * FPS / 2))
               for name, seconds in DESIGNS[version]]
    return np.column_stack([X] + context)


def fit(X, y, C=1.0, version="v1"):
    """A JSON-ready logistic regression: NaN imputed by the training mean, then standardised."""
    from sklearn.linear_model import LogisticRegression

    mean = np.nanmean(X, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    X = np.where(np.isnan(X), mean, X)
    scale = X.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    model = LogisticRegression(C=C, max_iter=5000)
    model.fit((X - mean) / scale, y)
    return {"names": list(design_names(version)), "C": C, "mean": mean.tolist(),
            "scale": scale.tolist(), "coef": model.coef_[0].tolist(),
            "intercept": float(model.intercept_[0])}


def fit_trees(X, y, max_iter=200, max_leaf_nodes=31, learning_rate=0.1, min_samples_leaf=20,
              version="v1"):
    """A JSON-ready gradient-boosted tree model (M11d): scikit-learn fits, arrays are kept.

    The trees are scikit-learn's own (HistGradientBoostingClassifier, no early stopping, a fixed
    random state), copied node by node: feature, threshold, which branch a missing value takes,
    children, leaf flag and value. score_trees applies them with numpy alone. NaN is left to the
    trees, which learn where a missing value goes, so nothing is imputed.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    params = {"max_iter": max_iter, "max_leaf_nodes": max_leaf_nodes,
              "learning_rate": learning_rate, "min_samples_leaf": min_samples_leaf}
    model = HistGradientBoostingClassifier(early_stopping=False, random_state=0, **params)
    model.fit(X, y)
    trees = []
    for predictors in model._predictors:  # one tree per iteration for a binary target
        nodes = predictors[0].nodes
        trees.append({"feature": nodes["feature_idx"].tolist(),
                      "threshold": nodes["num_threshold"].tolist(),
                      "missing_left": nodes["missing_go_to_left"].tolist(),
                      "left": nodes["left"].tolist(), "right": nodes["right"].tolist(),
                      "leaf": nodes["is_leaf"].tolist(), "value": nodes["value"].tolist()})
    return {"kind": "trees", "names": list(design_names(version)), "params": params,
            "baseline": float(np.ravel(model._baseline_prediction)[0]), "trees": trees}


def score_trees(model, X):
    """Log-odds of a fit_trees model, numpy only.

    Every row walks every tree with scikit-learn's rule (a missing value takes the node's
    missing branch; otherwise x <= threshold goes left), and the leaf values are added tree by
    tree in a fixed order, so the score is the same on any machine.
    """
    X = np.asarray(X, dtype=np.float64)
    z = np.full(len(X), model["baseline"], dtype=np.float64)
    for tree in model["trees"]:
        feature = np.asarray(tree["feature"], dtype=np.int64)
        threshold = np.asarray(tree["threshold"], dtype=np.float64)
        missing_left = np.asarray(tree["missing_left"], dtype=bool)
        left = np.asarray(tree["left"], dtype=np.int64)
        right = np.asarray(tree["right"], dtype=np.int64)
        leaf = np.asarray(tree["leaf"], dtype=bool)
        node = np.zeros(len(X), dtype=np.int64)
        walking = np.flatnonzero(~leaf[node])
        while len(walking):
            at = node[walking]
            x = X[walking, feature[at]]
            go_left = np.where(np.isnan(x), missing_left[at], x <= threshold[at])
            node[walking] = np.where(go_left, left[at], right[at])
            walking = walking[~leaf[node[walking]]]
        z += np.asarray(tree["value"], dtype=np.float64)[node]
    return z


def score(model, X):
    """Log-odds: a linear model's, summed column by column in a fixed order (no BLAS), or a tree
    model's (score_trees)."""
    if model.get("kind") == "trees":
        return score_trees(model, X)
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


def log_sigmoid(z):
    """log(1 / (1 + exp(-z))) for any z, without overflow."""
    z = np.asarray(z, dtype=np.float64)
    return np.minimum(z, 0.0) - np.log1p(np.exp(-np.abs(z)))


def combine(parts):
    """(dead, home) from several models' (dead, home) log-odds: one part as it is, several
    averaged, summed in the order given."""
    if len(parts) == 1:
        return parts[0]
    dead, home = parts[0]
    for more_dead, more_home in parts[1:]:
        dead, home = dead + more_dead, home + more_home
    return dead / len(parts), home / len(parts)


def scores(models, X):
    """(dead, home): per frame of one half, the log-odds that the ball is dead and, if it is
    alive, that the home side has it, from whichever models the file holds.

    The frozen file holds the two tree models, 'dead' and 'team', and gives exactly the numbers
    it always gave. M11l's candidates add a temporal convolutional network, 'tcn'
    (possession.tcn), alone or averaged with the trees, and a stage two, 'stack'
    (possession.stacking), which corrects the log-odds from how they move around each frame.
    M11m's 'tcn' may be a list of networks, an ensemble whose log-odds are averaged in order; a
    single network gives exactly its own numbers.
    """
    parts = []
    if "dead" in models:
        parts.append((score(models["dead"], X), score(models["team"], X)))
    if "tcn" in models:
        from possession import tcn
        networks = models["tcn"] if isinstance(models["tcn"], list) else [models["tcn"]]
        parts.append(combine([tcn.scores(network, X) for network in networks]))
    if not parts:
        raise ValueError("the models hold neither trees nor a network")
    dead, home = combine(parts)
    if "stack" in models:
        from possession import stacking
        dead, home = stacking.apply(models["stack"], dead, home)
    return dead, home


def log_state_probabilities(models, X):
    """(frames, 3) log-probabilities of H, A, D: the evidence possession.decode weighs.

    Taken from the linear scores rather than as log(state_probabilities), so a frame the models
    are sure about gets a large finite number instead of log(0).
    """
    dead, home = scores(models, X)
    alive = log_sigmoid(-dead)
    return np.column_stack([alive + log_sigmoid(home), alive + log_sigmoid(-home),
                            log_sigmoid(dead)])


def save(models, path):
    with open(path, "w", newline="\n") as handle:
        json.dump(models, handle, indent=1, sort_keys=True)
        handle.write("\n")


def load(path):
    """The models, checked against the design they name ('design', v1 when absent) and, for a
    stage two, against the features possession.stacking computes."""
    with open(path) as handle:
        models = json.load(handle)
    version = models.get("design", "v1")
    if version not in DESIGNS:
        raise ValueError("unknown design {}".format(version))
    if "dead" not in models and "tcn" not in models:
        raise ValueError("the models hold neither trees nor a network")
    for name in ("dead", "team", "tcn"):
        entries = models.get(name, [])
        for entry in entries if isinstance(entries, list) else [entries]:
            if tuple(entry["names"]) != design_names(version):
                raise ValueError("{} model was fitted on other features than this code computes"
                                 .format(name))
    if "stack" in models:
        from possession import stacking
        if tuple(models["stack"]["names"]) != stacking.names():
            raise ValueError("the stage two was fitted on other features than this code computes")
    return models
