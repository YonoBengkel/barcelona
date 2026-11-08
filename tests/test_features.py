"""Per-frame features on small synthetic halves."""

import numpy as np
import pytest

from possession.features import (NAMES, attack_directions, bridged_ball, compute, gap_frames,
                                 rolling_mean, swapped)

PITCH = (105.0, 68.0)


def two_teams(n=200, home_keeper=-50.0):
    """Home keeper on one goal line, away keeper on the other, outfielders near the centre."""
    rng = np.random.default_rng(0)
    x = np.concatenate([np.full((n, 1), home_keeper), rng.uniform(-20, 20, (n, 10)),
                        np.full((n, 1), -home_keeper), rng.uniform(-20, 20, (n, 10))], axis=1)
    y = rng.uniform(-30, 30, (n, 22))
    team = np.array([0] * 11 + [1] * 11, dtype=np.int8)
    return {"ball_x": np.zeros(n, dtype=np.float32), "ball_y": np.zeros(n, dtype=np.float32),
            "x": x.astype(np.float32), "y": y.astype(np.float32), "team": team}


def test_each_team_attacks_away_from_its_keeper():
    assert attack_directions(two_teams(home_keeper=-50.0)).tolist() == [1.0, -1.0]
    assert attack_directions(two_teams(home_keeper=50.0)).tolist() == [-1.0, 1.0]


def test_gap_frames_count_both_ways():
    since, until, length, last, following = gap_frames(np.array([1, 0, 0, 1, 0], dtype=bool))
    assert since.tolist() == [0, 1, 2, 0, 1]
    assert until.tolist() == [0, 2, 1, 0, 1]
    assert length.tolist() == [0, 2, 2, 0, 1]


def test_short_gaps_are_bridged_and_long_ones_are_not():
    half = two_teams(n=10)
    half["ball_x"] = np.array([0, np.nan, np.nan, 3, np.nan, np.nan, np.nan, np.nan, np.nan, 9],
                              dtype=np.float32)
    x, _, seen, bridged = bridged_ball(half, bridge_max=2)
    assert x[1] == pytest.approx(1.0) and x[2] == pytest.approx(2.0)
    assert np.isnan(x[5]) and bridged.tolist()[1:3] == [True, True] and not bridged[5]


def test_rolling_mean_ignores_missing_values():
    out = rolling_mean(np.array([1.0, np.nan, 3.0, np.nan]), window=1)
    assert out.tolist() == [1.0, 2.0, 3.0, 3.0]


def test_nearest_player_and_shape():
    half = two_teams()
    half["x"][:, 3] = 0.5    # a home outfielder next to the ball at the centre spot
    half["y"][:, 3] = 0.0
    features, names = compute(half, PITCH)
    assert features.shape == (200, len(NAMES)) and names == NAMES
    column = dict(zip(NAMES, features.T))
    assert np.all(column["near_home"] < column["near_away"])
    assert np.all(column["crowd2_home"] >= 1)
    assert not np.isinf(features).any()


def test_swapping_mirrors_the_team_features():
    half = two_teams()
    half["x"][:, 3], half["y"][:, 3] = 0.5, 0.0
    original = dict(zip(NAMES, compute(half, PITCH)[0].T))
    mirrored = dict(zip(NAMES, compute(swapped(half), PITCH)[0].T))
    assert np.allclose(original["near_home"], mirrored["near_away"])
    assert np.allclose(original["near_diff"], -mirrored["near_diff"])
