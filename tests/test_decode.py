"""The decoder on synthetic evidence: exact optimum, flicker removed, batches equal single runs."""

import itertools

import numpy as np
import pytest

from possession.decode import A, D, H, costs, decode, viterbi


def evidence(*runs):
    """(frames, 3) log-probabilities from (frames, P(H), P(A), P(D)) runs."""
    rows = [np.log([[h, a, d]] * n) for n, h, a, d in runs]
    return np.concatenate(rows)


def path_score(path, emission, team_switch, dead_switch, dead_bias):
    cost = costs(team_switch, dead_switch)[0]
    total = sum(emission[t, s] + (dead_bias if s == D else 0.0) for t, s in enumerate(path))
    return total - sum(cost[a, b] for a, b in zip(path[:-1], path[1:]))


def test_viterbi_finds_the_best_path_of_all():
    rng = np.random.default_rng(0)
    for _ in range(20):
        emission = np.log(rng.dirichlet(np.ones(3), size=6))
        params = (rng.uniform(0, 3), rng.uniform(0, 3), rng.uniform(-1, 1))
        best = max(path_score(p, emission, *params)
                   for p in itertools.product(range(3), repeat=len(emission)))
        found = viterbi(emission, *params)[:, 0]
        assert path_score(found, emission, *params) == pytest.approx(best, abs=1e-9)


def test_a_change_of_team_is_found_where_the_evidence_turns():
    emission = evidence((100, 0.8, 0.15, 0.05), (100, 0.15, 0.8, 0.05))
    path = viterbi(emission, 5.0, 5.0, 0.0)[:, 0]
    assert (path[:100] == H).all() and (path[100:] == A).all()


def test_a_short_contrary_run_costs_more_than_it_earns():
    emission = evidence((50, 0.8, 0.15, 0.05), (3, 0.15, 0.8, 0.05), (50, 0.8, 0.15, 0.05))
    assert (viterbi(emission, 10.0, 10.0, 0.0)[:, 0] == H).all()
    assert (viterbi(emission, 0.1, 10.0, 0.0)[50:53, 0] == A).all()


def test_dead_bias_moves_the_stoppages():
    emission = evidence((40, 0.5, 0.1, 0.4), (40, 0.3, 0.1, 0.6), (40, 0.5, 0.1, 0.4))
    dead = [int((viterbi(emission, 5.0, 5.0, bias)[:, 0] == D).sum()) for bias in (-1, 0, 1)]
    assert dead[0] < dead[1] < dead[2]


def test_a_batch_equals_its_single_runs():
    rng = np.random.default_rng(1)
    emission = np.log(rng.dirichlet(np.ones(3) * 0.5, size=300))
    team, dead, bias = np.array([1.0, 4.0, 16.0]), np.array([2.0, 8.0, 0.5]), np.array([0.0, 1.0, -1.0])
    batch = viterbi(emission, team, dead, bias)
    for k in range(3):
        assert np.array_equal(batch[:, k], viterbi(emission, team[k], dead[k], bias[k])[:, 0])


def test_ties_go_to_the_lower_state_and_decode_names_them():
    emission = np.log(np.full((5, 3), 1 / 3))
    assert decode(emission, {"team_switch": 1, "dead_switch": 1, "dead_bias": 0.0}).tolist() \
        == ["H"] * 5


def test_evidence_must_be_finite():
    emission = np.array([[0.0, -np.inf, -np.inf]])
    with pytest.raises(ValueError):
        viterbi(emission, 1.0, 1.0, 0.0)
