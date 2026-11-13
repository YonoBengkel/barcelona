"""scripts/train_tcn_variants.py (M11n): the first copies of a larger build are the halves a
smaller build gives, a wider network is still the numpy one, the trees and the networks are
averaged as a file holding both averages them, and the comparison runs end to end on synthetic
matches. The tests that need PyTorch run on Kaggle, which installs it for the network's jobs."""

import importlib.util
import os

import numpy as np
import pytest

from possession import tcn
from possession.features import NAMES
from possession.models import design_names, fit_trees, load, log_state_probabilities, score, \
    scores

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "train_tcn_variants", os.path.join(ROOT, "scripts", "train_tcn_variants.py"))
variants = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(variants)
tm, tt = variants.tm, variants.tt

WIDTH = len(design_names("v1"))


def random_network(rng, channels=4, dilations=(1, 2)):
    """A network with random weights on the v1 design, shaped as train_tcn.export writes one."""
    def layer(outputs, inputs, taps):
        return {"weight": rng.normal(0, 0.3, (outputs, inputs, taps)).tolist(),
                "bias": rng.normal(0, 0.1, outputs).tolist()}

    return {"kind": "tcn", "names": list(design_names("v1")), "mean": [0.0] * WIDTH,
            "scale": [1.0] * WIDTH, "input": layer(channels, 2 * WIDTH, 1),
            "blocks": [{"dilation": d, "dilated": layer(channels, channels, 3),
                        "pointwise": layer(channels, channels, 1)} for d in dilations],
            "output": layer(2, channels, 1)}


def test_the_first_copies_are_the_halves_a_smaller_build_gives_in_its_order():
    halves = [{"seed": seed, "section": section} for seed in range(4)
              for section in ("firstHalf", "secondHalf")]
    two = variants.copies({"M1": halves}, 2)["M1"]
    assert [(h["seed"], h["section"]) for h in two] == [
        (0, "firstHalf"), (0, "secondHalf"), (1, "firstHalf"), (1, "secondHalf")]
    assert variants.copies({"M1": halves}, 4)["M1"] == halves


def test_the_plan_fits_the_reference_first_and_averages_it_with_the_next_seeds():
    plan = variants.specs(dict(tt.TRAINING), 3, (1, 2, 4), 2, 8)
    assert [(key, spec["seed"], count) for key, spec, count in plan] == [
        ("reference", 0, 2), ("more", 0, 8), ("wider", 0, 2), ("seed1", 1, 2), ("seed2", 2, 2)]
    # The reference is the adopted network's settings exactly, so it is fitted the same way.
    assert plan[0][1] == tt.TRAINING and "dilations" not in plan[0][1]
    assert plan[2][1]["dilations"] == [1, 2, 4]


def test_the_trees_and_the_networks_are_averaged_as_a_file_of_both_is():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(300, WIDTH)).astype(np.float32)
    X[rng.random(X.shape) < 0.05] = np.nan
    y = np.nan_to_num(X[:, 0]) > 0
    dead, team = fit_trees(X, y, max_iter=10), fit_trees(X, ~y, max_iter=10)
    networks = [random_network(rng) for _ in range(3)]
    got = variants.with_trees([(score(dead, X), score(team, X))],
                              [[tcn.scores(network, X)] for network in networks])[0]
    expected = scores({"design": "v1", "dead": dead, "team": team, "tcn": networks}, X)
    assert got[0].tobytes() == expected[0].tobytes()
    assert got[1].tobytes() == expected[1].tobytes()


def test_a_wider_network_is_still_the_numpy_one(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(tcn, "CHANNELS", 6)
    monkeypatch.setattr(tcn, "DILATIONS", (1, 2))
    torch.manual_seed(0)
    columns = 5
    settings = dict(tt.TRAINING, dropout=0.0, dilations=[1, 2, 4, 8])
    net = tt.network(2 * columns, settings).double().eval()
    rng = np.random.default_rng(2)
    X = rng.normal(size=(150, columns)).astype(np.float32)
    X[rng.random(X.shape) < 0.1] = np.nan
    model = tt.export(net, rng.normal(size=columns), rng.uniform(0.5, 2.0, columns),
                      ["c{}".format(i) for i in range(columns)], settings)
    with torch.no_grad():
        expected = net(torch.from_numpy(tcn.inputs(model, X))[None])[0].numpy()
    dead, home = tcn.scores(model, X)
    assert np.allclose(dead, expected[0], rtol=0, atol=1e-10)
    assert np.allclose(home, expected[1], rtol=0, atol=1e-10)
    assert [block["dilation"] for block in model["blocks"]] == [1, 2, 4, 8]
    # Without the setting the network keeps possession.tcn's dilations, as the adopted one does.
    plain = tt.network(2 * columns, dict(tt.TRAINING, dropout=0.0))
    assert [int(layer.dilation[0]) for layer in plain.dilated] == [1, 2]


def synthetic_halves(rng, count, seed):
    """Halves shaped as build()'s sequences entries of one copy: near_diff says which team has
    the ball and speed_all whether it is dead, under noise."""
    near, speed = NAMES.index("near_diff"), NAMES.index("speed_all")
    halves = []
    for number in range(count):
        labels = np.concatenate([np.full(rng.integers(30, 120), rng.choice(list("HAD")))
                                 for _ in range(10)])
        X = rng.normal(size=(len(labels), WIDTH)).astype(np.float32)
        X[:, near] += np.where(labels == "H", -1.5, np.where(labels == "A", 1.5, 0.0))
        X[:, speed] += np.where(labels == "D", -1.0, 1.0)
        mirrored = X.copy()
        mirrored[:, near] *= -1
        halves.append({"X": X, "mirrored": mirrored, "labels": labels, "seed": seed,
                       "section": ("firstHalf", "secondHalf")[number % 2]})
    return halves


def synthetic_inputs(rng, matches, copies):
    """(train, sequences, held_out, confirm, labels per half) shaped as build() makes them: the
    trees' rows from the first two copies, the whole halves of every copy."""
    train, sequences, held_out, confirm, truth = {}, {}, {}, {}, {}
    for match in matches:
        halves = [half for seed in range(copies) for half in synthetic_halves(rng, 2, seed)]
        sequences[match] = halves
        first = [half for half in halves if half["seed"] < 2]
        live = [h["labels"] != "D" for h in first]
        train[match] = {
            "dead": ([h["X"][::4] for h in first], [h["labels"][::4] == "D" for h in first]),
            "team": ([part for h, l in zip(first, live) for part in (h["X"][l], h["mirrored"][l])],
                     [part for h, l in zip(first, live)
                      for part in (h["labels"][l] == "H", h["labels"][l] == "A")])}
        tested = synthetic_halves(rng, 2, 99)
        labels = np.concatenate([h["labels"] for h in tested])
        part = {"labels": labels, "lengths": [len(h["labels"]) for h in tested],
                "nearest": np.select([labels == "H", labels == "A"], [0, 1], -1).astype(np.int8),
                "ball_known": rng.random(len(labels)) < 0.6}
        X = np.concatenate([h["X"] for h in tested])
        held_out[match] = dict(part, X=X)
        confirm[match] = dict(part, X=(X + rng.normal(0, 0.3, X.shape)).astype(np.float32))
        truth[match] = [h["labels"] for h in tested]
    return train, sequences, held_out, confirm, truth


def test_the_comparison_runs_end_to_end(tmp_path, monkeypatch, capsys):
    pytest.importorskip("torch")
    rng = np.random.default_rng(5)
    matches = ["M1", "M2", "M3"]
    train, sequences, held_out, confirm, truth = synthetic_inputs(rng, matches, copies=3)
    small = ({"max_iter": 15, "max_leaf_nodes": 7, "learning_rate": 0.3, "min_samples_leaf": 5},)
    monkeypatch.setattr(tm, "labels_at",
                        lambda match, tolerances: {t: truth[match] for t in tolerances})
    monkeypatch.setattr(tm, "DECODER_AXES",
                        {"team_switch": (4, 16), "dead_switch": (16,), "dead_bias": (0.0,)})
    monkeypatch.setattr(tm, "TREES", {("team", "trees-wide"): small, ("dead", "trees"): small})
    monkeypatch.setattr(tcn, "CHANNELS", 6)
    monkeypatch.setattr(tcn, "DILATIONS", (1, 2))
    both = "v1+trees-wide+deadtrees+tcn3"
    # The rule passes the averaged candidate alone, so the final fit and its file run too.
    monkeypatch.setattr(tm, "compare", lambda summaries, matches, reference: both in summaries)
    tiny = {"steps": 30, "learning_rate": 1e-2, "weight_decay": 0.0, "dropout": 0.0,
            "clip": 1.0, "seed": 0}

    result = variants.run(train, sequences, held_out, confirm, matches, str(tmp_path), 0.0,
                          settings=tiny, workers=2, copies_count=2, more_copies=3,
                          wide_dilations=(1, 2, 4), networks=3)

    assert result["verdict"] == both
    assert set(result["summaries"]) == {tm.EVAL_SEED, 96}
    assert set(result["summaries"][96]) == {"v1+tcn", "v1+tcn+copies3", "v1+tcn+dil4", both}
    assert os.listdir(str(tmp_path)) == ["possession_{}.json".format(both)]
    models = load(str(tmp_path / "possession_{}.json".format(both)))
    assert "dead" in models and "team" in models and len(models["tcn"]) == 3
    assert models["fitted_on"]["network"]["seeds"] == [0, 1, 2]
    log_p = log_state_probabilities(models, held_out["M1"]["X"][:200])
    assert log_p.shape == (200, 3) and np.isfinite(log_p).all()
    assert "M11n verdict: adopt " + both in capsys.readouterr().out
