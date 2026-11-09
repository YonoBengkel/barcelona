"""scripts/train_models.py's decoder search, end to end on three synthetic matches.

The real search runs on Kaggle for half an hour; this catches a broken plumbing in a second.
"""

import contextlib
import importlib.util
import io
import itertools
import os
import time
import types

import numpy as np

from possession.features import NAMES
from possession.models import design_names, load

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "train_models", os.path.join(ROOT, "scripts", "train_models.py"))
train_models = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_models)


def synthetic_match(rng):
    """(labels per half, held-out part, log evidence per half) for one made-up match."""
    halves = [np.concatenate([np.full(rng.integers(50, 300), rng.choice(list("HAD")))
                              for _ in range(30)]) for _ in range(2)]
    evidence = []
    for labels in halves:
        p = 0.2 + 0.6 * np.stack([labels == s for s in "HAD"], axis=1) \
            + rng.uniform(0, 0.5, size=(len(labels), 3))
        evidence.append(np.log(p / p.sum(axis=1, keepdims=True)))
    nearest = [np.select([h == "H", h == "A"], [0, 1], -1).astype(np.int8) for h in halves]
    labels = np.concatenate(halves)
    X = np.zeros((len(labels), len(NAMES)), dtype=np.float32)  # only the ball columns matter
    X[:, NAMES.index("ball_seen")] = rng.random(len(labels)) < 0.6
    X[:, NAMES.index("gap_length")] = np.log1p(rng.uniform(0, 30, len(labels)))
    part = {"labels": labels, "nearest": np.concatenate(nearest), "X": X,
            "lengths": [len(h) for h in halves]}
    return halves, part, evidence


def test_the_decoder_search_runs_end_to_end(monkeypatch, capsys):
    rng = np.random.default_rng(0)
    matches = ["M1", "M2", "M3"]
    truth, held_out, oof = {}, {}, {}
    for match in matches:
        truth[match], held_out[match], oof[match] = synthetic_match(rng)
    axes = {"team_switch": (1, 8), "dead_switch": (1, 8), "dead_bias": (0.0,)}
    monkeypatch.setattr(train_models, "labels_at",
                        lambda match, tolerances: {t: truth[match] for t in tolerances})
    monkeypatch.setattr(train_models, "DECODER_AXES", axes)
    monkeypatch.setattr(train_models, "DECODER_GRID", tuple(itertools.product(*axes.values())))

    setting, practice_range, predicted, summary = train_models.fit_decoder(held_out, oof, matches)
    train_models.report_errors(held_out, predicted, matches)
    train_models.oracle_headroom(held_out, oof, matches)

    assert set(setting) == set(axes) and setting["team_switch"] in axes["team_switch"]
    for name in ("labels", "ours"):
        low, high = practice_range[name]["spells_per_10min"]
        assert 0 < low <= high
    assert all(len(predicted[m]) == len(held_out[m]["labels"]) + 1 for m in matches)
    assert set(summary["per_match"]) == set(matches) and set(summary["pooled"]) == {0.5, 1.0, 2.0}
    printed = capsys.readouterr().out
    assert "baseline_smooth50" in printed and "chosen on all seven" in printed
    assert "possessions at IoU 0.5" in printed and "dead-ball share per match" in printed
    assert "perfect-evidence check" in printed and "true both" in printed


def test_oracle_evidence_turns_the_call_and_keeps_the_confidence():
    rng = np.random.default_rng(3)
    log_p = np.log(rng.dirichlet(np.ones(3), size=300))
    truth = rng.integers(0, 3, size=300)  # H 0, A 1, D 2, as possession_metrics codes them
    assert np.allclose(train_models.oracle_evidence(log_p, truth, "model"), log_p)
    for variant in train_models.ORACLES:
        assert np.allclose(np.exp(train_models.oracle_evidence(log_p, truth, variant)).sum(1), 1)

    team = np.exp(train_models.oracle_evidence(log_p, truth, "true team"))
    alive = truth != 2
    right = team[np.flatnonzero(alive), truth[alive]]
    wrong = team[np.flatnonzero(alive), 1 - truth[alive]]
    assert (right >= wrong).all()
    p = np.exp(log_p)
    before = np.sort(p[:, :2] / p[:, :2].sum(1, keepdims=True), axis=1)
    after = np.sort(team[:, :2] / team[:, :2].sum(1, keepdims=True), axis=1)
    assert np.allclose(before, after)  # the same confidence, pointed the right way

    dead = np.exp(train_models.oracle_evidence(log_p, truth, "true dead"))[:, 2]
    assert (dead[truth == 2] >= 0.5).all() and (dead[truth != 2] <= 0.5).all()


def synthetic_design_inputs(rng, matches):
    """(train, held_out, labels per half) shaped as build() makes them, for made-up matches."""
    width = len(design_names("v1"))
    near, speed = NAMES.index("near_diff"), NAMES.index("speed_all")
    train, held_out, truth = {}, {}, {}
    for match in matches:
        halves = [np.concatenate([np.full(rng.integers(40, 150), rng.choice(list("HAD")))
                                  for _ in range(12)]) for _ in range(2)]
        Xs = []
        for labels in halves:
            X = rng.normal(size=(len(labels), width)).astype(np.float32)
            X[:, near] += np.where(labels == "H", -1.5, np.where(labels == "A", 1.5, 0.0))
            X[:, speed] += np.where(labels == "D", -1.0, 1.0)
            Xs.append(X)
        labels, X = np.concatenate(halves), np.concatenate(Xs)
        live = labels != "D"
        mirrored = X[live].copy()
        mirrored[:, near] *= -1
        train[match] = {"dead": ([X], [labels == "D"]),
                        "team": ([X[live], mirrored], [labels[live] == "H", labels[live] == "A"])}
        held_out[match] = {"X": X, "labels": labels, "lengths": [len(h) for h in halves],
                           "nearest": np.select([labels == "H", labels == "A"], [0, 1], -1)
                           .astype(np.int8), "ball_known": rng.random(len(labels)) < 0.6}
        truth[match] = halves
    return train, held_out, truth


def test_fit_design_writes_one_model_file_per_pair_of_kinds(tmp_path, monkeypatch, capsys):
    rng = np.random.default_rng(5)
    matches = ["M1", "M2", "M3"]
    train, held_out, truth = synthetic_design_inputs(rng, matches)
    axes = {"team_switch": (4, 16), "dead_switch": (16,), "dead_bias": (0.0,)}
    small = ({"max_iter": 15, "max_leaf_nodes": 7, "learning_rate": 0.3, "min_samples_leaf": 5},)
    monkeypatch.setattr(train_models, "labels_at",
                        lambda match, tolerances: {t: truth[match] for t in tolerances})
    monkeypatch.setattr(train_models, "DECODER_AXES", axes)
    monkeypatch.setattr(train_models, "DECODER_GRID", tuple(itertools.product(*axes.values())))
    monkeypatch.setattr(train_models, "GRID", (1.0,))
    monkeypatch.setattr(train_models, "TREES", {("team", "trees"): small,
                                                ("dead", "trees"): small})
    args = types.SimpleNamespace(seeds=2, step=8, out_dir=str(tmp_path), oracle=False)

    summaries = train_models.fit_design("v1", ["logistic", "trees"], ["logistic", "trees"],
                                        train, held_out, matches, args, time.time())

    assert set(summaries) == {"v1", "v1+trees", "v1+deadtrees", "v1+trees+deadtrees"}
    for variant in summaries:
        models = load(str(tmp_path / "possession_{}.json".format(variant)))
        assert models["design"] == "v1" and "decoder" in models
    both = load(str(tmp_path / "possession_v1+trees+deadtrees.json"))
    assert both["team"]["kind"] == both["dead"]["kind"] == "trees"
    assert "kind" not in load(str(tmp_path / "possession_v1.json"))["dead"]  # logistic
    train_models.compare(summaries, matches, "v1+trees")
    printed = capsys.readouterr().out
    assert "team trees" in printed and "dead trees" in printed
    assert "v1+trees+deadtrees against v1+trees" in printed and "verdict:" in printed


def test_the_best_passing_variant_is_the_one_adopted():
    matches = ["M1", "M2", "M3", "M4", "M5", "M6", "M7"]

    def summary(level):
        return {"pooled": {t: (level, 0.88) for t in (0.5, 1.0, 2.0)},
                "per_match": {m: level for m in matches}}

    with contextlib.redirect_stdout(io.StringIO()):
        winner = train_models.compare({"ref": summary(0.60), "a": summary(0.62),
                                       "b": summary(0.63), "c": summary(0.605)}, matches, "ref")
        nobody = train_models.compare({"ref": summary(0.60), "c": summary(0.605)}, matches, "ref")
    assert winner == "b" and nobody is None


def test_a_design_is_adopted_only_by_the_rule():
    matches = ["M1", "M2", "M3", "M4", "M5", "M6", "M7"]
    reference = {"pooled": {t: (0.60, 0.88) for t in (0.5, 1.0, 2.0)},
                 "per_match": {m: 0.60 for m in matches}}

    def candidate(gain, better_on, macro=0.0):
        return {"pooled": {t: (0.60 + gain, 0.88 + macro) for t in (0.5, 1.0, 2.0)},
                "per_match": {m: 0.60 + (0.02 if i < better_on else -0.01)
                              for i, m in enumerate(matches)}}

    def verdict(summary):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            train_models.compare({"v1": reference, "v2": summary}, matches)
        return "ADOPT v2" in out.getvalue()

    assert verdict(candidate(0.02, 6))
    assert not verdict(candidate(0.005, 6))          # too small a gain
    assert not verdict(candidate(0.02, 4))           # better on too few matches
    assert not verdict(candidate(0.02, 6, -0.01))    # costs too much macroF1
