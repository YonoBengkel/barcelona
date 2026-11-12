"""possession.stacking and scripts/train_stacking.py (M11l): the stage-two features against their
definition and a plain loop, the home-away mirror, the training rows, fitting and applying, and
the comparison end to end on synthetic matches."""

import importlib.util
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

from possession import stacking
from possession.features import NAMES, rolling_mean
from possession.models import design_names, fit_trees, log_state_probabilities, score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "train_stacking", os.path.join(ROOT, "scripts", "train_stacking.py"))
train_stacking = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_stacking)
tm = train_stacking.tm

TINY = {"max_iter": 10, "max_leaf_nodes": 7, "learning_rate": 0.3, "min_samples_leaf": 5}


def test_the_features_follow_the_names_with_nan_only_beyond_the_half():
    names = stacking.names()
    assert len(names) == len(set(names)) == 30
    assert names[:6] == ("dead", "home", "dead_lag5", "dead_lead5", "home_lag5", "home_lead5")
    assert names[18:20] == ("dead_mean12", "home_mean12")
    assert names[-4:] == ("home_min25", "home_max25", "home_min62", "home_max62")
    rng = np.random.default_rng(1)
    n = 300
    dead, home = rng.normal(size=n), rng.normal(size=n)
    F = stacking.features(dead, home)
    assert F.dtype == np.float32 and F.shape == (n, len(names))
    column = dict(zip(names, F.T))
    assert np.array_equal(column["dead"], dead.astype(np.float32))
    assert np.array_equal(column["home"], home.astype(np.float32))
    frames = np.arange(n)
    for offset in stacking.OFFSETS:
        for name, values in (("dead", dead), ("home", home)):
            lagged = column["{}_lag{}".format(name, offset)]
            led = column["{}_lead{}".format(name, offset)]
            assert np.array_equal(np.isnan(lagged), frames < offset)
            assert np.array_equal(np.isnan(led), frames >= n - offset)
            assert np.array_equal(lagged[offset:], values[:n - offset].astype(np.float32))
            assert np.array_equal(led[:n - offset], values[offset:].astype(np.float32))
    for window in stacking.WINDOWS:
        assert np.array_equal(column["dead_mean{}".format(window)],
                              rolling_mean(dead, window).astype(np.float32))
        assert np.array_equal(column["home_mean{}".format(window)],
                              rolling_mean(home, window).astype(np.float32))
    shifted = [j for j, name in enumerate(names) if "_lag" in name or "_lead" in name]
    assert len(shifted) == 16
    assert np.isfinite(np.delete(F, shifted, axis=1)).all()  # nothing else leaves the half


def test_negating_the_team_log_odds_mirrors_every_column():
    rng = np.random.default_rng(2)
    dead, home = rng.normal(size=400), 3.0 * rng.normal(size=400)
    names = stacking.names()
    F, mirrored = stacking.features(dead, home), stacking.features(dead, -home)
    expected = np.empty_like(F)
    for j, name in enumerate(names):
        if name.startswith("dead"):
            expected[:, j] = F[:, j]  # which side is home does not change the dead-ball call
        elif name.startswith("home_min") or name.startswith("home_max"):
            other = name.replace("min", "max") if "min" in name else name.replace("max", "min")
            expected[:, j] = -F[:, names.index(other)]
        else:
            expected[:, j] = -F[:, j]
    assert np.array_equal(mirrored, expected, equal_nan=True)  # exact, not close


def test_the_rolling_extremes_are_those_of_a_plain_loop():
    values = np.random.default_rng(3).normal(size=40)
    for window in (0, 1, 3, 25, 62):
        low, high = stacking.rolling_min(values, window), stacking.rolling_max(values, window)
        assert low.shape == high.shape == values.shape
        for t in range(len(values)):
            part = values[max(0, t - window):t + window + 1]
            assert low[t] == min(part) and high[t] == max(part)


def test_the_rows_keep_every_step_th_frame_both_ways():
    rng = np.random.default_rng(4)
    first, second = np.array(list("HHHHDDDAAAAAHHD")), np.array(list("AAAADDDDHH"))
    halves = [(rng.normal(size=len(labels)), rng.normal(size=len(labels)), labels)
              for labels in (first, second)]
    (dead_X, dead_y), (team_X, team_y) = stacking.rows(halves, step=4)
    # Kept: frames 0, 4, 8, 12 of the first half (H D A H) and 0, 4, 8 of the second (A D H).
    assert dead_X.dtype == team_X.dtype == np.float32 and dead_y.dtype == team_y.dtype == bool
    assert dead_X.shape == (2 * (4 + 3), 30) and team_X.shape == (2 * (3 + 2), 30)
    assert dead_y.tolist() == [False, True, False, False] * 2 + [False, True, False] * 2
    # Live kept frames: H A H, then A H; labelled home as they are and away mirrored.
    assert team_y.tolist() == [True, False, True, False, True, False, False, True, True, False]
    dead, home, _ = halves[0]
    as_is, mirrored = stacking.features(dead, home), stacking.features(dead, -home)
    assert np.array_equal(dead_X[:4], as_is[[0, 4, 8, 12]], equal_nan=True)
    assert np.array_equal(dead_X[4:8], mirrored[[0, 4, 8, 12]], equal_nan=True)
    assert np.array_equal(team_X[:3], as_is[[0, 8, 12]], equal_nan=True)
    assert np.array_equal(team_X[3:6], mirrored[[0, 8, 12]], equal_nan=True)


def synthetic_labels(rng):
    """One half's labels: twelve runs, four each of H, A and D in a random order, so that every
    half holds every label."""
    return np.repeat(rng.permutation(list("HAD" * 4)), rng.integers(30, 120, size=12))


def synthetic_log_odds(rng, count):
    """Halves as stacking.rows takes them: first-stage log-odds that tell the truth under noise."""
    halves = []
    for _ in range(count):
        labels = synthetic_labels(rng)
        dead = np.where(labels == "D", 1.5, -1.5) + rng.normal(size=len(labels))
        home = np.where(labels == "H", 1.0, np.where(labels == "A", -1.0, 0.0)) \
            + rng.normal(size=len(labels))
        halves.append((dead, home, labels))
    return halves


def test_stage_two_is_deterministic_and_sharpens_both_calls(monkeypatch):
    monkeypatch.setattr(stacking, "TREES", TINY)
    rng = np.random.default_rng(5)
    halves = synthetic_log_odds(rng, 4)
    stack = stacking.fit(halves)
    assert stack["kind"] == "stack" and stack["settings"]["trees"] == TINY
    assert stack["names"] == stack["dead"]["names"] == stack["team"]["names"] \
        == list(stacking.names())

    dead, home, labels = synthetic_log_odds(rng, 1)[0]  # a half it never saw
    corrected = stacking.apply(stack, dead, home)
    again = stacking.apply(json.loads(json.dumps(stack)), dead, home)  # through a model file
    for new, same in zip(corrected, again):
        assert new.dtype == np.float64 and new.shape == dead.shape
        assert new.tobytes() == same.tobytes()
    live = labels != "D"
    assert roc_auc_score(labels == "D", corrected[0]) > 0.75
    assert roc_auc_score(labels[live] == "H", corrected[1][live]) > 0.75


def test_the_first_stage_gives_the_frozen_evidence_on_the_seed_zero_copies():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(300, len(design_names("v1"))))
    y = X[:, 0] + rng.normal(scale=0.5, size=300) > 0
    models = {"dead": fit_trees(X, y, max_iter=5), "team": fit_trees(X, ~y, max_iter=5)}
    dead, home = score(models["dead"], X), score(models["team"], X)
    assert np.array_equal(train_stacking.log_probabilities(dead, home),
                          log_state_probabilities(models, X))
    labels = np.where(y, "H", "A")
    copies = [{"X": X, "labels": labels, "seed": seed} for seed in (0, 1, 0)]
    out = train_stacking.training_evidence(models, copies)
    assert len(out) == 2 and out[0][2] is labels
    assert np.array_equal(out[0][0], dead) and np.array_equal(out[1][1], home)


def synthetic_copy(rng, labels, width):
    """A design matrix for labels: near_diff says which team has the ball and speed_all whether
    it is dead, under noise."""
    near, speed = NAMES.index("near_diff"), NAMES.index("speed_all")
    X = rng.normal(size=(len(labels), width)).astype(np.float32)
    X[:, near] += np.where(labels == "H", -1.5, np.where(labels == "A", 1.5, 0.0))
    X[:, speed] += np.where(labels == "D", -1.0, 1.0)
    return X


def synthetic_inputs(rng, matches):
    """(train, sequences, held_out, confirm, labels per half) shaped as train_models.build()
    makes them: two training copies of every half (seeds 0 and 1), kept whole in sequences and
    every fourth frame as rows, and a held-out copy, with a little more noise in confirm."""
    width = len(design_names("v1"))
    near = NAMES.index("near_diff")
    train, sequences, held_out, confirm, truth = {}, {}, {}, {}, {}
    for match in matches:
        halves = [synthetic_labels(rng) for _ in range(2)]
        rows = {"dead": ([], []), "team": ([], [])}
        for seed in range(2):
            for section, labels in zip(("firstHalf", "secondHalf"), halves):
                X = synthetic_copy(rng, labels, width)
                mirrored = X.copy()
                mirrored[:, near] *= -1
                every, live = np.arange(0, len(labels), 4), np.flatnonzero(labels != "D")[::4]
                rows["dead"][0].append(X[every])
                rows["dead"][1].append(labels[every] == "D")
                rows["team"][0].extend([X[live], mirrored[live]])
                rows["team"][1].extend([labels[live] == "H", labels[live] == "A"])
                sequences.setdefault(match, []).append(
                    {"X": X, "mirrored": mirrored, "labels": labels, "seed": seed,
                     "section": section})
        train[match] = rows
        labels = np.concatenate(halves)
        X = np.concatenate([synthetic_copy(rng, half, width) for half in halves])
        part = {"labels": labels, "lengths": [len(half) for half in halves],
                "nearest": np.select([labels == "H", labels == "A"], [0, 1], -1).astype(np.int8),
                "ball_known": rng.random(len(labels)) < 0.6}
        held_out[match] = dict(part, X=X)
        confirm[match] = dict(part, X=(X + rng.normal(0, 0.3, X.shape)).astype(np.float32))
        truth[match] = halves
    return train, sequences, held_out, confirm, truth


def test_the_comparison_runs_end_to_end(tmp_path, monkeypatch, capsys):
    rng = np.random.default_rng(7)
    matches = ["M1", "M2", "M3"]
    train, sequences, held_out, confirm, truth = synthetic_inputs(rng, matches)
    small = ({"max_iter": 15, "max_leaf_nodes": 7, "learning_rate": 0.3, "min_samples_leaf": 5},)
    axes = {"team_switch": (4, 16), "dead_switch": (16,), "dead_bias": (0.0,)}
    monkeypatch.setattr(tm, "labels_at",
                        lambda match, tolerances: {t: truth[match] for t in tolerances})
    monkeypatch.setattr(tm, "DECODER_AXES", axes)
    monkeypatch.setattr(tm, "TREES", {("team", "trees-wide"): small, ("dead", "trees"): small})
    monkeypatch.setattr(stacking, "TREES", TINY)

    result = train_stacking.run(train, sequences, held_out, confirm, matches, str(tmp_path), 0.0)

    reference, candidate = "v1+trees-wide+deadtrees", "v1+trees-wide+deadtrees+stack"
    assert result["verdict"] in (None, candidate)
    assert set(result["summaries"]) == {tm.EVAL_SEED, 98}
    for summaries in result["summaries"].values():
        assert set(summaries) == {reference, candidate}
    # Read as JSON: possession.models.load of a stage two is the models module's to test.
    with open(str(tmp_path / "possession_{}.json".format(candidate))) as handle:
        models = json.load(handle)
    assert set(models) == {"design", "dead", "team", "stack", "decoder", "fitted_on"}
    assert models["dead"]["kind"] == models["team"]["kind"] == "trees"
    assert models["stack"]["kind"] == "stack" and models["stack"]["settings"]["trees"] == TINY
    assert models["stack"]["names"] == list(stacking.names())
    assert set(models["decoder"]) == set(axes)
    assert models["fitted_on"]["confirm_seed"] == 98 and models["fitted_on"]["matches"] == matches
    X = held_out["M1"]["X"][:200]
    dead, home = stacking.apply(models["stack"], score(models["dead"], X),
                                score(models["team"], X))
    assert np.isfinite(dead).all() and np.isfinite(home).all()
    printed = capsys.readouterr().out
    assert "stage two held out on M3" in printed and "against " + reference in printed
    assert "verdict:" in printed and "M11l stacking verdict" in printed and "seed 98" in printed
