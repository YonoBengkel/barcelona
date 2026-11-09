"""scripts/train_models.py's decoder search, end to end on three synthetic matches.

The real search runs on Kaggle for half an hour; this catches a broken plumbing in a second.
"""

import importlib.util
import itertools
import os

import numpy as np

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
    part = {"labels": np.concatenate(halves), "nearest": np.concatenate(nearest),
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

    setting, practice_range = train_models.fit_decoder(held_out, oof, matches)

    assert set(setting) == set(axes) and setting["team_switch"] in axes["team_switch"]
    for name in ("labels", "ours"):
        low, high = practice_range[name]["spells_per_10min"]
        assert 0 < low <= high
    printed = capsys.readouterr().out
    assert "baseline_smooth50" in printed and "chosen on all seven" in printed
