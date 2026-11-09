"""The error views on hand-made possessions: each miss gets the cause a person would name."""

import numpy as np

from possession import diagnose, practice
from possession.features import NAMES

pm = practice.pm


def codes(*runs):
    """Codes from (letter, frames) runs."""
    return practice.codes(np.concatenate([np.full(n, letter) for letter, n in runs]))


def test_each_miss_is_put_down_to_one_cause():
    gt = codes(("H", 20), ("D", 10), ("A", 20), ("D", 10), ("H", 20), ("D", 10), ("A", 20))
    pred = codes(("A", 20), ("D", 10), ("D", 20), ("D", 10), ("H", 9), ("D", 2), ("H", 9),
                 ("D", 10), ("A", 20))
    out = diagnose.spell_outcomes(gt, pred)
    assert (out["true"], out["predicted"], out["matched"]) == (4, 4, 1)
    assert out["missed"] == {"wrong team": 1, "called dead": 1, "split": 1}
    assert out["spurious"] == {"wrong team": 1, "piece of one": 2}
    assert out["by_length"]["<2 s"] == [1, 4]


def test_one_prediction_over_two_possessions_is_a_merge():
    out = diagnose.spell_outcomes(codes(("H", 20), ("D", 4), ("H", 20)), codes(("H", 44)))
    assert out["matched"] == 0 and out["missed"] == {"merged": 2}
    assert out["spurious"] == {"piece of one": 1}


def test_a_shifted_possession_has_its_edges_off():
    out = diagnose.spell_outcomes(codes(("D", 10), ("H", 20), ("D", 10)),
                                  codes(("D", 20), ("H", 20)))
    assert out["missed"] == {"edges off": 1}


def random_labels(rng, n=1500):
    runs = []
    while sum(len(run) for run in runs) < n:
        runs.append(np.full(rng.integers(1, 60), rng.choice(list("HAD"))))
    return np.concatenate(runs)[:n]


def test_pairs_match_the_organisers_greedy_matching():
    rng = np.random.default_rng(0)
    for _ in range(10):
        gt, pred = practice.codes(random_labels(rng)), practice.codes(random_labels(rng))
        runs = pm.valid_runs(gt)
        true_spells, pred_spells = pm.possession_spells(gt, runs), pm.possession_spells(pred, runs)
        overlaps = pm.spell_overlaps(true_spells, pred_spells)
        for threshold in pm.SPELL_IOUS:
            flags = np.zeros(len(true_spells), dtype=bool)
            flags[[i for i, _ in diagnose.matched_pairs(overlaps, threshold)]] = True
            assert np.array_equal(flags, pm.matched_gt_flags(overlaps, len(true_spells),
                                                             threshold))


def test_ball_states_and_the_state_table():
    features = np.zeros((6, len(NAMES)), dtype=np.float32)
    features[0, NAMES.index("ball_seen")] = 1
    features[1, NAMES.index("ball_bridged")] = 1
    features[2:, NAMES.index("gap_length")] = np.log1p([3.0, 10.0, 20.0, 0.5])
    states = diagnose.ball_states(features)
    assert states.tolist() == [0, 1, 2, 3, 4, 2]

    pred = np.array([pm.H, pm.A, pm.D, pm.D, pm.D, pm.H])
    gt = np.array([pm.H, pm.H, pm.D, pm.A, pm.D, pm.D])
    rows = {row["state"]: row for row in diagnose.state_table(states, pred, gt)}
    assert abs(rows["all"]["H"] + rows["all"]["A"] + rows["all"]["D"] - 1.0) < 1e-12
    assert rows["all"]["right"] == 0.5 and rows["gap <=5 s"]["true D"] == 1.0
    lines = diagnose.format_state_table(list(rows.values()))
    assert "true D" in lines[0] and len(lines) == len(diagnose.STATES) + 2

    speed = np.array([1.0, 2.0, 3.0, 4.0, 5.0, np.nan])
    rows = {row["state"]: row for row in diagnose.state_table(states, pred, gt, speed)}
    assert rows["all"]["m/s"] == 3.0  # the median of 1..5; the missing value is left out
    assert rows["gap <=5 s"]["m/s D"] == 3.0 and np.isnan(rows["gap <=5 s"]["m/s live"])
    assert "m/s live" in diagnose.format_state_table(list(rows.values()))[0]


def test_player_speed_undoes_the_design_log_scale():
    features = np.zeros((2, len(NAMES)))
    features[:, NAMES.index("speed_all")] = [0.0, 2.0]
    assert diagnose.player_speed(features).tolist() == [0.0, 2.0]
    logged = np.log1p(features)
    assert np.allclose(diagnose.player_speed(logged, logged=True), [0.0, 2.0])
