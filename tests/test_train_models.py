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

from possession import practice
from possession.decode import viterbi
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
    args = types.SimpleNamespace(seeds=2, step=8, out_dir=str(tmp_path), oracle=False,
                                 grids=["base"], speed_factors=[])

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


def test_the_wide_grid_and_the_speed_check_run_end_to_end(tmp_path, monkeypatch, capsys):
    rng = np.random.default_rng(7)
    matches = ["M1", "M2", "M3"]
    train, held_out, truth = synthetic_design_inputs(rng, matches)
    speed = design_names("v1").index("speed_all")
    for part in held_out.values():
        slow = part["X"].copy()
        slow[:, speed] -= 0.5  # slower players: the synthetic dead frames are the slow ones
        part["slow"] = {0.8: slow}
    axes = {"team_switch": (4, 16), "dead_switch": (16,), "dead_bias": (0.0,)}
    monkeypatch.setattr(train_models, "labels_at",
                        lambda match, tolerances: {t: truth[match] for t in tolerances})
    monkeypatch.setattr(train_models, "DECODER_AXES", axes)
    monkeypatch.setattr(train_models, "WIDE_DEAD_BIAS", (1.0,))
    monkeypatch.setattr(train_models, "GRID", (1.0,))
    args = types.SimpleNamespace(seeds=2, step=8, out_dir=str(tmp_path), oracle=False,
                                 grids=["base", "wide"], speed_factors=[0.8])

    summaries = train_models.fit_design("v1", ["logistic"], ["logistic"], train, held_out,
                                        matches, args, time.time())

    assert set(summaries) == {"v1", "v1+widebias"}
    wide = load(str(tmp_path / "possession_v1+widebias.json"))
    assert wide["fitted_on"]["decoder_grid"]["dead_bias"] == [0.0, 1.0]
    assert load(str(tmp_path / "possession_v1.json"))["fitted_on"]["decoder_grid"][
        "dead_bias"] == [0.0]
    for summary in summaries.values():
        assert set(summary["speed"]) == {1.0, 0.8}
        # Unscaled, the check decodes exactly what the held-out score was computed from.
        assert summary["speed"][1.0][:2] == summary["pooled"][1.0]
        assert summary["speed"][0.8][2] > summary["speed"][1.0][2]  # slower, so more dead
        assert set(summary["nested"]) == set(matches)
    printed = capsys.readouterr().out
    assert "decoder grid wide" in printed and "speed check" in printed
    assert "speed x0.8" in printed and "speed x1.0" in printed


def test_the_old_ground_is_fitted_there_and_scored_on_this_runs_matches(monkeypatch, capsys):
    rng = np.random.default_rng(13)
    matches = ["M1", "M2", "M3"]
    train, held_out, truth = synthetic_design_inputs(rng, matches)

    def noisier(X):
        return (X + rng.normal(0, 0.5, X.shape)).astype(np.float32)

    # The same matches and labels on another ground: only the features differ.
    old_train = {m: {model: ([noisier(X) for X in Xs], ys) for model, (Xs, ys) in rows.items()}
                 for m, rows in train.items()}
    old_held_out = {m: dict(part, X=noisier(part["X"])) for m, part in held_out.items()}
    axes = {"team_switch": (4, 16), "dead_switch": (16,), "dead_bias": (0.0,)}
    monkeypatch.setattr(train_models, "labels_at",
                        lambda match, tolerances: {t: truth[match] for t in tolerances})
    monkeypatch.setattr(train_models, "DECODER_AXES", axes)
    monkeypatch.setattr(train_models, "GRID", (1.0,))

    old = train_models.fit_old_ground("v1", ["logistic"], ["logistic"], old_train, old_held_out,
                                      held_out, matches, time.time())

    assert set(old) == {"v1 (old ground)"}
    summary = old["v1 (old ground)"]
    assert set(summary["pooled"]) == {0.5, 1.0, 2.0} and set(summary["per_match"]) == set(matches)
    printed = capsys.readouterr().out
    assert "practice ground before" in printed and "on this run's held-out matches" in printed


def test_scoring_fixed_settings_reproduces_the_nested_score(monkeypatch):
    rng = np.random.default_rng(17)
    matches = ["M1", "M2", "M3"]
    truth, held_out, oof = {}, {}, {}
    for match in matches:
        truth[match], held_out[match], oof[match] = synthetic_match(rng)
    axes = {"team_switch": (1, 8), "dead_switch": (1, 8), "dead_bias": (0.0, 0.5)}
    monkeypatch.setattr(train_models, "labels_at",
                        lambda match, tolerances: {t: truth[match] for t in tolerances})
    monkeypatch.setattr(train_models, "DECODER_AXES", axes)
    with contextlib.redirect_stdout(io.StringIO()):
        summary = train_models.fit_decoder(held_out, oof, matches)[3]
    again = train_models.score_settings(held_out, oof, summary["nested"], matches)
    assert again["pooled"] == summary["pooled"] and again["per_match"] == summary["per_match"]


def test_slowed_scales_the_players_speeds_alone():
    rng = np.random.default_rng(11)
    features = rng.uniform(0.5, 5.0, size=(40, len(NAMES))).astype(np.float32)
    features[3, NAMES.index("speed_home")] = np.nan
    out = train_models.slowed(features, 0.8)
    for j, name in enumerate(NAMES):
        if name in train_models.PLAYER_SPEEDS:
            assert np.allclose(out[:, j], 0.8 * features[:, j], equal_nan=True)
        else:
            assert np.array_equal(out[:, j], features[:, j])
    assert np.isnan(out[3, NAMES.index("speed_home")])
    assert out[0, NAMES.index("speed_all")] < features[0, NAMES.index("speed_all")]  # a copy


def test_the_wide_grid_holds_every_base_setting():
    base, wide = train_models.decoder_axes("base"), train_models.decoder_axes("wide")
    assert list(wide) == list(base)  # the order decoded() hands them to viterbi in
    assert set(itertools.product(*base.values())) < set(itertools.product(*wide.values()))
    assert max(wide["dead_bias"]) > max(base["dead_bias"])
    assert train_models.variant_name("v1", "trees", "trees-wide", "wide") == \
        "v1+trees-wide+deadtrees+widebias"
    assert train_models.variant_name("v1", "trees", "trees-wide") == "v1+trees-wide+deadtrees"


def test_the_mindur_grid_holds_every_base_setting():
    base, held = train_models.decoder_axes("base"), train_models.decoder_axes("mindur")
    assert list(held)[:3] == list(base) and list(held)[3:] == list(train_models.MINIMA)
    plain = {setting + (1, 1) for setting in itertools.product(*base.values())}
    assert plain < set(itertools.product(*held.values()))
    assert min(held["team_switch"]) < min(base["team_switch"])
    assert train_models.variant_name("v1", "trees", "trees-wide", "mindur") == \
        "v1+trees-wide+deadtrees+mindur"


def test_decoded_groups_the_settings_by_their_minima():
    rng = np.random.default_rng(19)
    log_ps = [np.log(rng.dirichlet(np.ones(3) * 0.5, size=n)) for n in (120, 90)]
    settings = [(8, 8, 0.0), (16, 4, 0.5, 3, 5), (8, 8, 0.0, 1, 1), (4, 8, -0.5, 3, 5),
                (8, 16, 0.0, 6, 1)]
    out = list(train_models.decoded(log_ps, settings))
    assert np.array_equal(out[0], out[2])  # minima of one frame are the plain model
    for setting, codes in zip(settings, out):
        minima = dict(zip(train_models.MINIMA, setting[3:5]))
        halves = [viterbi(log_p, *setting[:3], **minima)[:, 0] for log_p in log_ps]
        assert np.array_equal(codes, practice.joined(halves, gap=train_models.pm.X))


def test_the_mindur_grid_runs_end_to_end(tmp_path, monkeypatch, capsys):
    rng = np.random.default_rng(23)
    matches = ["M1", "M2", "M3"]
    train, held_out, truth = synthetic_design_inputs(rng, matches)
    speed = design_names("v1").index("speed_all")
    for part in held_out.values():
        slow = part["X"].copy()
        slow[:, speed] -= 0.5
        part["slow"] = {0.8: slow}
    axes = {"team_switch": (4, 16), "dead_switch": (16,), "dead_bias": (0.0,)}
    monkeypatch.setattr(train_models, "labels_at",
                        lambda match, tolerances: {t: truth[match] for t in tolerances})
    monkeypatch.setattr(train_models, "DECODER_AXES", axes)
    monkeypatch.setattr(train_models, "MIN_TEAM_SWITCH", (2,))
    monkeypatch.setattr(train_models, "TEAM_MIN", (1, 3))
    monkeypatch.setattr(train_models, "DEAD_MIN", (1, 5))
    monkeypatch.setattr(train_models, "GRID", (1.0,))
    args = types.SimpleNamespace(seeds=2, step=8, out_dir=str(tmp_path), oracle=False,
                                 grids=["base", "mindur"], speed_factors=[0.8])

    summaries = train_models.fit_design("v1", ["logistic"], ["logistic"], train, held_out,
                                        matches, args, time.time())

    assert set(summaries) == {"v1", "v1+mindur"}
    held = load(str(tmp_path / "possession_v1+mindur.json"))
    assert set(held["decoder"]) == set(train_models.decoder_axes("mindur"))
    assert held["fitted_on"]["decoder_grid"]["team_switch"] == [2, 4, 16]
    assert held["fitted_on"]["decoder_grid"]["team_min"] == [1, 3]
    assert set(load(str(tmp_path / "possession_v1.json"))["decoder"]) == set(axes)
    for summary in summaries.values():
        # The speed check decodes the nested settings again, minima and all.
        assert summary["speed"][1.0][:2] == summary["pooled"][1.0]
    train_models.compare(summaries, matches, "v1")
    printed = capsys.readouterr().out
    assert "decoder grid mindur" in printed and "v1+mindur against v1" in printed


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
