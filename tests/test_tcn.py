"""possession.tcn and scripts/train_tcn.py (M11l): the network's numpy forward against a plain loop
and against PyTorch, its determinism and reach, fitting, and the comparison end to end on
synthetic matches.

PyTorch is installed on Kaggle for the network's job alone (requirements-torch.txt), so the tests
that need it are skipped everywhere else.
"""

import importlib.util
import os

import numpy as np
import pytest

from possession import tcn
from possession.features import NAMES
from possession.models import design_names, load, log_state_probabilities

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "train_tcn", os.path.join(ROOT, "scripts", "train_tcn.py"))
train_tcn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_tcn)
tm = train_tcn.tm


def random_network(rng, columns, channels=5, dilations=(1, 2, 4)):
    """A network with random weights, shaped as train_tcn.export writes one."""
    def layer(outputs, inputs, taps):
        return {"weight": rng.normal(0, 0.4, (outputs, inputs, taps)).tolist(),
                "bias": rng.normal(0, 0.1, outputs).tolist()}

    return {"kind": "tcn", "mean": [0.0] * columns, "scale": [1.0] * columns,
            "input": layer(channels, 2 * columns, 1),
            "blocks": [{"dilation": d, "dilated": layer(channels, channels, 3),
                        "pointwise": layer(channels, channels, 1)} for d in dilations],
            "output": layer(2, channels, 1)}


def test_inputs_standardise_the_columns_and_flag_the_missing():
    X = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)
    x = tcn.inputs({"mean": [2.0, 1.0], "scale": [1.0, 3.0]}, X)
    assert x.shape == (4, 2)
    assert np.array_equal(x, [[-1.0, 1.0], [0.0, 1.0], [0.0, 0.0], [1.0, 0.0]])


def test_a_convolution_is_the_sum_it_says_in_its_order():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 20))
    layer = {"weight": rng.normal(size=(4, 3, 3)).tolist(), "bias": rng.normal(size=4).tolist()}
    out = tcn.convolve(x, layer, dilation=2)
    weight, bias = np.array(layer["weight"]), np.array(layer["bias"])
    expected = np.zeros((4, 20))
    for o in range(4):
        for t in range(20):
            total = bias[o]
            for k in range(3):
                for i in range(3):
                    source = t + (k - 1) * 2
                    if 0 <= source < 20:
                        total += weight[o, i, k] * x[i, source]
            expected[o, t] = total
    assert np.array_equal(out, expected)


def test_the_network_is_deterministic_and_reaches_no_further_than_it_says():
    rng = np.random.default_rng(1)
    model = random_network(rng, 4)
    X = rng.normal(size=(90, 4)).astype(np.float32)
    dead, home = tcn.scores(model, X)
    again = tcn.scores(model, X)
    assert dead.tobytes() == again[0].tobytes() and home.tobytes() == again[1].tobytes()
    changed = X.copy()
    changed[45, 2] += 3.0
    moved_dead, moved_home = tcn.scores(model, changed)
    moved = np.flatnonzero((moved_dead != dead) | (moved_home != home))
    reach = tcn.reach((1, 2, 4), 3)
    assert reach == 7 and moved.size > 0
    assert moved.min() >= 45 - reach and moved.max() <= 45 + reach


def test_the_labels_mirror_and_the_normaliser_ignores_the_missing():
    assert list(train_tcn.mirrored_labels(np.array(list("HADH")))) == list("AHDA")
    X = np.array([[1.0, np.nan], [3.0, 5.0]], dtype=np.float32)
    mean, scale = train_tcn.normaliser([{"X": X, "mirrored": X}])
    assert np.allclose(mean, [2.0, 5.0]) and np.allclose(scale, [1.0, 1.0])  # 5.0 never varies


def test_the_numpy_network_is_the_pytorch_one(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(tcn, "CHANNELS", 6)
    monkeypatch.setattr(tcn, "DILATIONS", (1, 2, 4))
    torch.manual_seed(0)
    columns = 5
    net = train_tcn.network(2 * columns, dict(train_tcn.TRAINING, dropout=0.0)).double().eval()
    rng = np.random.default_rng(2)
    X = rng.normal(size=(120, columns)).astype(np.float32)
    X[rng.random(X.shape) < 0.1] = np.nan
    model = train_tcn.export(net, rng.normal(size=columns), rng.uniform(0.5, 2.0, columns),
                             ["c{}".format(i) for i in range(columns)], train_tcn.TRAINING)
    with torch.no_grad():
        expected = net(torch.from_numpy(tcn.inputs(model, X))[None])[0].numpy()
    dead, home = tcn.scores(model, X)
    assert np.allclose(dead, expected[0], rtol=0, atol=1e-10)
    assert np.allclose(home, expected[1], rtol=0, atol=1e-10)
    assert [block["dilation"] for block in model["blocks"]] == [1, 2, 4]


def synthetic_halves(rng, count, width):
    """Halves shaped as build()'s sequences entries: the near_diff column says which team has
    the ball and speed_all whether it is dead, under noise."""
    near, speed = NAMES.index("near_diff"), NAMES.index("speed_all")
    halves = []
    for number in range(count):
        labels = np.concatenate([np.full(rng.integers(30, 120), rng.choice(list("HAD")))
                                 for _ in range(10)])
        X = rng.normal(size=(len(labels), width)).astype(np.float32)
        X[:, near] += np.where(labels == "H", -1.5, np.where(labels == "A", 1.5, 0.0))
        X[:, speed] += np.where(labels == "D", -1.0, 1.0)
        mirrored = X.copy()
        mirrored[:, near] *= -1
        halves.append({"X": X, "mirrored": mirrored, "labels": labels, "seed": 0,
                       "section": ("firstHalf", "secondHalf")[number % 2]})
    return halves


TINY = {"steps": 120, "learning_rate": 1e-2, "weight_decay": 0.0, "dropout": 0.0, "clip": 1.0,
        "seed": 0}


def test_fitting_learns_both_calls_on_a_synthetic_signal(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setattr(tcn, "CHANNELS", 8)
    monkeypatch.setattr(tcn, "DILATIONS", (1, 2))
    rng = np.random.default_rng(3)
    width = len(design_names("v1"))
    model = train_tcn.fit_network(synthetic_halves(rng, 4, width), TINY, "cpu",
                                  design_names("v1"))
    test = synthetic_halves(rng, 1, width)[0]
    dead, home = tcn.scores(model, test["X"])
    labels, live = test["labels"], test["labels"] != "D"
    assert np.mean((dead > 0) == (labels == "D")) > 0.75
    assert np.mean((home[live] > 0) == (labels[live] == "H")) > 0.75


def synthetic_inputs(rng, matches):
    """(train, sequences, held_out, confirm, labels per half) shaped as build() makes them."""
    width = len(design_names("v1"))
    train, sequences, held_out, confirm, truth = {}, {}, {}, {}, {}
    for match in matches:
        halves = synthetic_halves(rng, 2, width)
        sequences[match] = halves
        live = [h["labels"] != "D" for h in halves]
        train[match] = {
            "dead": ([h["X"][::4] for h in halves], [h["labels"][::4] == "D" for h in halves]),
            "team": ([part for h, l in zip(halves, live) for part in (h["X"][l], h["mirrored"][l])],
                     [part for h, l in zip(halves, live)
                      for part in (h["labels"][l] == "H", h["labels"][l] == "A")])}
        tested = synthetic_halves(rng, 2, width)
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
    train, sequences, held_out, confirm, truth = synthetic_inputs(rng, matches)
    small = ({"max_iter": 15, "max_leaf_nodes": 7, "learning_rate": 0.3, "min_samples_leaf": 5},)
    monkeypatch.setattr(tm, "labels_at",
                        lambda match, tolerances: {t: truth[match] for t in tolerances})
    monkeypatch.setattr(tm, "DECODER_AXES",
                        {"team_switch": (4, 16), "dead_switch": (16,), "dead_bias": (0.0,)})
    monkeypatch.setattr(tm, "TREES", {("team", "trees-wide"): small, ("dead", "trees"): small})
    monkeypatch.setattr(tcn, "CHANNELS", 6)
    monkeypatch.setattr(tcn, "DILATIONS", (1, 2))

    result = train_tcn.run(train, sequences, held_out, confirm, matches, str(tmp_path), 0.0,
                           settings=dict(TINY, steps=40))

    alone, both = "v1+tcn", "v1+trees-wide+deadtrees+tcn"
    assert result["verdict"] in (None, alone, both)
    assert set(result["summaries"]) == {tm.EVAL_SEED, 98}
    for name in (alone, both):
        models = load(str(tmp_path / "possession_{}.json".format(name)))
        assert set(models["decoder"]) == {"team_switch", "dead_switch", "dead_bias"}
        log_p = log_state_probabilities(models, held_out["M1"]["X"][:200])
        assert log_p.shape == (200, 3) and np.isfinite(log_p).all()
    assert "dead" in load(str(tmp_path / "possession_{}.json".format(both)))
    printed = capsys.readouterr().out
    assert "M11l network verdict" in printed and "seed 98" in printed
    assert "against v1+trees-wide+deadtrees" in printed
