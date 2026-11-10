"""Profile statistics and the degradation simulator on synthetic halves."""

from dataclasses import replace

import numpy as np
import pytest

from possession.features import NAMES, compute
from possession.profile import gap_starts, player_speeds, profile, run_lengths
from possession.simulate import DegradeParams, degrade, frames_until_restart


def make_half(n=20000, players=22, seed=1, period=3000):
    """A ball looping inside the pitch with 22 players scattered around it."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    ball = np.stack([40 * np.sin(2 * np.pi * t / period),
                     25 * np.sin(2 * np.pi * t / (0.7 * period))], axis=1)
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


def test_player_speeds_are_the_features_speed_all():
    half = make_half(n=600)
    half["x"][100:140, 3] = np.nan  # a player lost for a while
    mean, each = player_speeds(half)
    speed_all = compute(half, (105.0, 68.0))[0][:, NAMES.index("speed_all")]
    assert np.allclose(mean, speed_all, rtol=1e-5, equal_nan=True)
    assert each.shape == half["x"].shape
    assert np.isnan(each[112:129, 3]).all()  # every second around them lost the player
    assert np.isfinite(each[112:129, 4]).all()


def test_position_noise_shows_in_the_players_speeds():
    half = make_half(n=4000)
    half["x"] = np.repeat(half["x"][:1], 4000, axis=0)  # every player stands still
    half["y"] = np.repeat(half["y"][:1], 4000, axis=0)
    quiet = profile([degrade(half, DegradeParams(player_noise_m=0.0), seed=0)], (105.0, 68.0))
    noisy = profile([degrade(half, DegradeParams(player_noise_m=0.15), seed=0)], (105.0, 68.0))
    assert quiet["player_speed_each"]["p50"] < 0.01 and quiet["player_speed"]["p90"] < 0.01
    # 0.15 m drifting over a second steps about 1.06 m/s per axis: about 1.3 m/s standing still.
    assert 1.0 < noisy["player_speed_each"]["p50"] < 1.7


def test_player_speeds_by_ball_state_follow_the_gaps():
    half = make_half(n=3000)
    still = np.arange(3000)[:, None] < 1000  # the players stand still for the first 40 s
    half["x"] = np.where(still, half["x"][:1], half["x"]).astype(np.float32)
    half["y"] = np.where(still, half["y"][:1], half["y"]).astype(np.float32)
    half["ball_x"][:600] = np.nan  # and the ball is missing for the first 24 s of it
    half["ball_y"][:600] = np.nan
    by_state = profile([half], (105.0, 68.0))["player_speed_by_state"]
    assert by_state["gap >15 s"]["p50"] < 0.01
    assert by_state["seen"]["p50"] > 1.0
    assert by_state["gap 5-15 s"]["n"] == 0


def test_frames_until_restart_counts_to_the_next_frame_in_play():
    dead = np.array([0, 1, 1, 0, 1, 1, 1], dtype=bool)
    assert frames_until_restart(dead).tolist() == [0, 2, 1, 0, 3, 2, 1]


def test_gaps_that_last_until_the_restart_stay_in_the_stoppage():
    half = make_half(n=40000)
    half["alive"] = np.arange(40000) % 4000 >= 2000  # 80 s stopped, then 80 s of play, in turn
    params = DegradeParams(dead_weight=4.0)
    drawn = np.isnan(degrade(half, params, seed=0)["ball_x"])
    to_restart = np.isnan(degrade(half, replace(params, until_restart=1.0), seed=0)["ball_x"])
    dead = ~half["alive"]
    assert dead[to_restart].mean() > dead[drawn].mean()
    assert to_restart.mean() == pytest.approx(0.407, abs=0.03)  # the share still hits the target


def test_ball_missing_share_hits_the_target():
    # Gaps last about 12 s by default, so a long half keeps the share's granularity fine.
    out = degrade(make_half(n=80000), DegradeParams(ball_missing_share=0.4), seed=0)
    assert np.isnan(out["ball_x"]).mean() == pytest.approx(0.4, abs=0.03)


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


def test_risk_puts_gaps_where_the_ball_is_slow():
    slow_half, fast_half = make_half(n=40000, period=40000), make_half(n=40000, period=400)
    half = {key: np.concatenate([slow_half[key], fast_half[key]]) if key != "team"
            else slow_half[key] for key in slow_half}
    params = DegradeParams(line_weight=0.0, box_weight=0.0, crowd_weight=0.0, slow_weight=3.0,
                           short_gap_share=1.0, ball_missing_share=0.2)
    hidden = np.isnan(degrade(half, params, seed=0)["ball_x"])
    starts = np.flatnonzero(~hidden[:-1] & hidden[1:]) + 1
    assert np.mean(starts < 40000) > 0.75  # the slow first part is half the frames
