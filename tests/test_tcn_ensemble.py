"""scripts/train_tcn_ensemble.py (M11m): scoring in worker processes gives the numbers scoring in
turn gives, an ensemble file averages its networks in order, and the comparison runs end to end on
synthetic matches. The last test needs PyTorch, which Kaggle installs for the network's jobs."""

import importlib.util
import os

import numpy as np
import pytest

from possession import tcn
from possession.features import NAMES
from possession.models import combine, design_names, load, log_state_probabilities, save, \
    scores

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "train_tcn_ensemble", os.path.join(ROOT, "scripts", "train_tcn_ensemble.py"))
ensemble = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ensemble)
tm = ensemble.tm

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


def test_scoring_in_worker_processes_gives_the_numbers_of_scoring_in_turn():
    rng = np.random.default_rng(0)
    models = [random_network(rng) for _ in range(3)]
    Xs = [rng.normal(size=(150, WIDTH)).astype(np.float32) for _ in range(3)]
    in_turn = ensemble.scored(models, Xs)
    pool = ensemble.worker_pool(2)
    try:
        pooled = ensemble.scored(models, Xs, pool)
    finally:
        pool.shutdown()
    assert len(pooled) == len(in_turn) == 3
    for (dead, home), (pooled_dead, pooled_home) in zip(in_turn, pooled):
        assert dead.tobytes() == pooled_dead.tobytes()
        assert home.tobytes() == pooled_home.tobytes()
    assert ensemble.worker_pool(1) is None


def test_an_ensemble_file_averages_its_networks_in_order(tmp_path):
    rng = np.random.default_rng(1)
    networks = [random_network(rng) for _ in range(3)]
    X = rng.normal(size=(120, WIDTH)).astype(np.float32)
    X[rng.random(X.shape) < 0.05] = np.nan
    path = str(tmp_path / "ensemble.json")
    save({"design": "v1", "tcn": networks}, path)
    expected = combine([tcn.scores(network, X) for network in networks])
    dead, home = scores(load(path), X)
    assert dead.tobytes() == expected[0].tobytes() and home.tobytes() == expected[1].tobytes()
    # One network, as the adopted file holds it, still gives exactly its own numbers.
    alone = scores({"design": "v1", "tcn": networks[0]}, X)
    assert alone[0].tobytes() == tcn.scores(networks[0], X)[0].tobytes()
    save({"design": "v1", "tcn": [networks[0], dict(networks[1], names=list(NAMES))]},
         str(tmp_path / "bad.json"))
    with pytest.raises(ValueError):
        load(str(tmp_path / "bad.json"))


def synthetic_halves(rng, count):
    """Halves shaped as build()'s sequences entries: near_diff says which team has the ball and
    speed_all whether it is dead, under noise."""
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
        halves.append({"X": X, "mirrored": mirrored, "labels": labels, "seed": 0,
                       "section": ("firstHalf", "secondHalf")[number % 2]})
    return halves


def synthetic_inputs(rng, matches):
    """(sequences, held_out, confirm, labels per half) shaped as build() makes them."""
    sequences, held_out, confirm, truth = {}, {}, {}, {}
    for match in matches:
        sequences[match] = synthetic_halves(rng, 2)
        tested = synthetic_halves(rng, 2)
        labels = np.concatenate([h["labels"] for h in tested])
        part = {"labels": labels, "lengths": [len(h["labels"]) for h in tested],
                "nearest": np.select([labels == "H", labels == "A"], [0, 1], -1).astype(np.int8),
                "ball_known": rng.random(len(labels)) < 0.6}
        X = np.concatenate([h["X"] for h in tested])
        held_out[match] = dict(part, X=X)
        confirm[match] = dict(part, X=(X + rng.normal(0, 0.3, X.shape)).astype(np.float32))
        truth[match] = [h["labels"] for h in tested]
    return sequences, held_out, confirm, truth


def test_the_comparison_runs_end_to_end(tmp_path, monkeypatch, capsys):
    pytest.importorskip("torch")
    rng = np.random.default_rng(5)
    matches = ["M1", "M2", "M3"]
    sequences, held_out, confirm, truth = synthetic_inputs(rng, matches)
    monkeypatch.setattr(tm, "labels_at",
                        lambda match, tolerances: {t: truth[match] for t in tolerances})
    monkeypatch.setattr(tm, "DECODER_AXES",
                        {"team_switch": (4, 16), "dead_switch": (16,), "dead_bias": (0.0,)})
    monkeypatch.setattr(tm, "WIDE_DEAD_BIAS", (1.0,))
    monkeypatch.setattr(tcn, "CHANNELS", 6)
    monkeypatch.setattr(tcn, "DILATIONS", (1, 2))
    tiny = {"steps": 30, "learning_rate": 1e-2, "weight_decay": 0.0, "dropout": 0.0,
            "clip": 1.0, "seed": 0}

    result = ensemble.run({}, sequences, held_out, confirm, matches, str(tmp_path), 0.0,
                          networks=2, settings=tiny, workers=2)

    alone, wide = "v1+tcn2", "v1+tcn2+widebias"
    assert result["verdict"] in (None, alone, wide)
    assert set(result["summaries"]) == {tm.EVAL_SEED, 97}
    for name in (alone, wide):
        models = load(str(tmp_path / "possession_{}.json".format(name)))
        assert isinstance(models["tcn"], list) and len(models["tcn"]) == 2
        log_p = log_state_probabilities(models, held_out["M1"]["X"][:200])
        assert log_p.shape == (200, 3) and np.isfinite(log_p).all()
    wide_file = load(str(tmp_path / "possession_{}.json".format(wide)))
    assert wide_file["fitted_on"]["decoder_grid"]["dead_bias"] == [0.0, 1.0]
    printed = capsys.readouterr().out
    assert "M11m ensemble verdict" in printed and "seed 97" in printed
    assert "against v1+tcn" in printed
