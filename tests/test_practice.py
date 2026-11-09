"""Pooled practice-ground scores equal the organisers' evaluate() on the joined matches."""

import numpy as np
import pytest

from possession import practice

pm = practice.pm


def random_runs(rng, n_runs):
    """Labels in runs of 1 to 200 frames, as letters."""
    return np.concatenate([np.full(rng.integers(1, 200), rng.choice(["H", "A", "D"]))
                           for _ in range(n_runs)])


def test_pooled_counts_score_like_evaluate_on_the_joined_matches():
    rng = np.random.default_rng(0)
    matches = []
    for _ in range(3):
        halves = [random_runs(rng, 60), random_runs(rng, 50)]
        guesses = [h.copy() for h in halves]
        for guess in guesses:  # damage the truth: flip some runs, shift a boundary or two
            guess[rng.random(len(guess)) < 0.002] = "A"
            guess[: rng.integers(0, 40)] = "D"
        matches.append((practice.joined([practice.codes(h) for h in halves], gap=pm.X),
                        practice.joined([practice.codes(g) for g in guesses], gap=pm.X)))

    parts = [practice.counts(gt, pred) for gt, pred in matches]
    reference = pm.evaluate(practice.joined([gt for gt, _ in matches], gap=pm.X),
                            practice.joined([pred for _, pred in matches], gap=pm.X))
    spell_f1, macro_f1 = practice.pooled(parts)
    assert spell_f1 == pytest.approx(reference["spell_f1_mean"], abs=1e-12)
    assert macro_f1 == pytest.approx(reference["macro_f1"], abs=1e-12)

    alone = pm.evaluate(*matches[1])
    assert practice.pooled([parts[1]])[0] == pytest.approx(alone["spell_f1_mean"], abs=1e-12)


def test_codes_refuse_unknown_labels():
    assert practice.codes(np.array(["H", "A", "D"])).tolist() == [pm.H, pm.A, pm.D]
    with pytest.raises(ValueError):
        practice.codes(np.array(["H", "X"]))
