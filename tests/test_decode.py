"""The decoder on synthetic evidence: exact optimum, flicker removed, batches equal single runs,
and with minimum spell lengths (M11j) the best path of those that keep them."""

import itertools

import numpy as np
import pytest

from possession.decode import A, D, H, _held, costs, decode, viterbi


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


def keeps_minima(path, team_min, dead_min):
    """Whether every spell of path lasts its minimum, except a first or last one cut by the edge."""
    spells, start = [], 0
    for t in range(1, len(path) + 1):
        if t == len(path) or path[t] != path[start]:
            spells.append((path[start], t - start))
            start = t
    minimum = {H: team_min, A: team_min, D: dead_min}
    return all(length >= minimum[state] for state, length in spells[1:-1])


def test_held_spells_give_the_best_path_that_keeps_the_minima():
    rng = np.random.default_rng(2)
    for team_min, dead_min in ((2, 2), (3, 1), (1, 3), (2, 4)):
        for _ in range(8):
            emission = np.log(rng.dirichlet(np.ones(3), size=7))
            params = (rng.uniform(0, 3), rng.uniform(0, 3), rng.uniform(-1, 1))
            best = max(path_score(p, emission, *params)
                       for p in itertools.product(range(3), repeat=len(emission))
                       if keeps_minima(p, team_min, dead_min))
            found = viterbi(emission, *params, team_min=team_min, dead_min=dead_min)[:, 0]
            assert keeps_minima(found, team_min, dead_min)
            assert path_score(found, emission, *params) == pytest.approx(best, abs=1e-9)


def test_minima_of_one_frame_give_the_plain_model():
    rng = np.random.default_rng(4)
    emission = np.log(rng.dirichlet(np.ones(3) * 0.5, size=400))
    team, dead, bias = np.array([1.0, 4.0, 16.0]), np.array([2.0, 8.0, 0.5]), np.array([0.0, 1.0, -1.0])
    plain = viterbi(emission, team, dead, bias)
    assert np.array_equal(viterbi(emission, team, dead, bias, team_min=1, dead_min=1), plain)
    on_d = np.zeros((3, 3))
    on_d[:, D] = bias
    # The held code itself, made to run with minima of one, finds the same paths.
    assert np.array_equal(_held(emission, costs(team, dead), on_d, np.array([1, 1, 1])), plain)


def test_no_spell_inside_the_stretch_is_shorter_than_its_minimum():
    weak = (0.45, 0.5, 0.05)  # barely for A: three frames pay for two cheap switches, no more
    emission = evidence((50, 0.8, 0.15, 0.05), (3, *weak), (50, 0.8, 0.15, 0.05))
    assert (viterbi(emission, 0.01, 10.0, 0.0)[50:53, 0] == A).all()  # the plain model takes it
    assert (viterbi(emission, 0.01, 10.0, 0.0, team_min=4)[:, 0] == H).all()
    emission = evidence((50, 0.8, 0.15, 0.05), (6, *weak), (50, 0.8, 0.15, 0.05))
    held = viterbi(emission, 0.01, 10.0, 0.0, team_min=4)[:, 0]
    assert (held[:50] == H).all() and (held[50:56] == A).all() and (held[56:] == H).all()


def test_a_spell_cut_by_the_start_or_the_end_may_be_short():
    emission = evidence((50, 0.8, 0.15, 0.05), (3, 0.05, 0.9, 0.05))
    held = viterbi(emission, 0.01, 10.0, 0.0, team_min=10)[:, 0]
    assert (held[:50] == H).all() and (held[50:] == A).all()
    emission = evidence((3, 0.05, 0.9, 0.05), (50, 0.8, 0.15, 0.05))
    held = viterbi(emission, 0.01, 10.0, 0.0, team_min=10)[:, 0]
    assert (held[:3] == A).all() and (held[3:] == H).all()


def test_a_held_batch_equals_its_single_runs():
    rng = np.random.default_rng(6)
    emission = np.log(rng.dirichlet(np.ones(3) * 0.5, size=300))
    team, dead, bias = np.array([0.5, 4.0, 16.0]), np.array([2.0, 8.0, 0.5]), np.array([0.0, 1.0, -1.0])
    batch = viterbi(emission, team, dead, bias, team_min=5, dead_min=12)
    for k in range(3):
        single = viterbi(emission, team[k], dead[k], bias[k], team_min=5, dead_min=12)[:, 0]
        assert np.array_equal(batch[:, k], single)
        assert keeps_minima(batch[:, k], 5, 12)


def test_minima_must_be_whole_frames_of_one_or_more():
    emission = np.log(np.full((4, 3), 1 / 3))
    for team_min, dead_min in ((0, 1), (1, 0), (1.5, 1)):
        with pytest.raises(ValueError):
            viterbi(emission, 1.0, 1.0, 0.0, team_min=team_min, dead_min=dead_min)


def test_decode_reads_the_minima_from_the_model_file():
    emission = evidence((50, 0.8, 0.15, 0.05), (3, 0.45, 0.5, 0.05), (50, 0.8, 0.15, 0.05))
    params = {"team_switch": 0.01, "dead_switch": 10.0, "dead_bias": 0.0}
    assert decode(emission, params)[51] == "A"
    assert decode(emission, dict(params, team_min=4, dead_min=1))[51] == "H"
