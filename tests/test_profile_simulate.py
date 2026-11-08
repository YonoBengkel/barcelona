"""Profile statistics and the degradation simulator on synthetic halves."""

import numpy as np
import pytest

from possession.profile import gap_starts, profile, run_lengths
from possession.simulate import DegradeParams, degrade


def make_half(n=20000, players=22, seed=1):
    """A random-walk ball with 22 players scattered around it, ball height mostly low."""
    rng = np.random.default_rng(seed)
    ball = np.cumsum(rng.normal(0, 0.3, (n, 2)), axis=0)
    return {
        "ball_x": ball[:, 0].astype(np.float32), "ball_y": ball[:, 1].astype(np.float32),
        "ball_z": np.abs(rng.normal(0, 1.0, n)).astype(np.float32),
        "x": (ball[:, :1] + rng.normal(0, 8, (n, players))).astype(np.float32),
        "y": (ball[:, 1:] + rng.normal(0, 8, (n, players))).astype(np.float32),
        "team": np.repeat(np.array([0, 1], dtype=np.int8), players // 2),
        "alive": np.ones(n, dtype=bool), "owner": np.zeros(n, dtype=np.int8),
    }


def test_run_lengths_split_true_and_false():
    seen, gaps = run_lengths(np.array([1, 1, 0, 0, 0, 1, 0], dtype=bool))
    assert seen.tolist() == [2, 1] and gaps.tolist() == [3, 1]


def test_gap_starts_mark_the_last_seen_frame_of_long_gaps():
    present = np.array([1, 1, 0, 0, 0, 0, 0, 1, 0, 1], dtype=bool)
    assert gap_starts(present, min_gap=5).tolist() == [1]


def test_profile_of_clean_tracking():
    result = profile([make_half(n=2000)], pitch=(105.0, 68.0))
    assert result["ball_seen_share"] == 1.0
    assert result["objects_per_frame"]["mean"] == 22


def test_ball_missing_share_hits_the_target():
    out = degrade(make_half(), DegradeParams(ball_missing_share=0.4), seed=0)
    assert np.isnan(out["ball_x"]).mean() == pytest.approx(0.4, abs=0.02)


def test_players_tracked_hits_the_target():
    out = degrade(make_half(), DegradeParams(players_tracked=20.5), seed=0)
    assert (~np.isnan(out["x"])).sum(axis=1).mean() == pytest.approx(20.5, abs=0.3)


def test_height_is_dropped_and_labels_are_kept():
    half = make_half(n=2000)
    out = degrade(half, seed=0)
    assert "ball_z" not in out
    assert out["alive"] is half["alive"] and out["owner"] is half["owner"]


def test_degradation_is_reproducible_from_the_seed():
    half = make_half(n=5000)
    first, again, other = degrade(half, seed=3), degrade(half, seed=3), degrade(half, seed=4)
    assert np.array_equal(np.isnan(first["ball_x"]), np.isnan(again["ball_x"]))
    assert not np.array_equal(np.isnan(first["ball_x"]), np.isnan(other["ball_x"]))


def test_outside_stretches_can_be_hidden():
    half = make_half(n=4000)
    half["ball_x"][1000:1500] = 60.0  # outside a 105 m pitch
    out = degrade(half, DegradeParams(outside_visible=0.0, ball_missing_share=0.2), seed=0)
    assert np.isnan(out["ball_x"][1000:1500]).all()


def test_noise_drifts_instead_of_jumping():
    half = make_half(n=5000)
    out = degrade(half, DegradeParams(ball_missing_share=0.0, ball_noise_m=0.3), seed=0)
    step = np.abs(np.diff(out["ball_x"] - half["ball_x"]))
    assert np.nanmedian(step) < 0.1  # independent 0.3 m noise per frame would step about 0.33 m


def test_risk_puts_gaps_where_the_ball_is_high():
    half = make_half()
    out = degrade(half, DegradeParams(crowd_weight=0.0, height_weight=3.0), seed=0)
    hidden = np.isnan(out["ball_x"])
    high = half["ball_z"] > 2.0
    starts = np.flatnonzero(~hidden[:-1] & hidden[1:]) + 1
    assert high[starts].mean() > 2 * high.mean()
