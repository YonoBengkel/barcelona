"""scripts/calibrate_simulator.py's player-noise fit (M11g), on a synthetic half."""

import importlib.util
import os
from dataclasses import replace

import numpy as np
import pytest

from possession.simulate import DegradeParams

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "calibrate_simulator", os.path.join(ROOT, "scripts", "calibrate_simulator.py"))
calibrate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(calibrate)


def standing_half(n=3000, players=22, seed=2):
    """Players who never move and a ball that does: every player speed is position noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return {"ball_x": (30 * np.sin(t / 200)).astype(np.float32),
            "ball_y": np.zeros(n, dtype=np.float32),
            "x": np.repeat(rng.uniform(-40, 40, (1, players)), n, axis=0).astype(np.float32),
            "y": np.repeat(rng.uniform(-25, 25, (1, players)), n, axis=0).astype(np.float32),
            "team": np.repeat(np.array([0, 1], dtype=np.int8), players // 2)}


def test_the_fit_recovers_the_noise_that_made_the_target(capsys):
    halves = [standing_half()]
    target = calibrate.player_summary(halves, replace(DegradeParams(), player_noise_m=0.075))
    assert calibrate.fit_players(halves, target) == (0.075, 0.0)
    assert "best player_noise_m 0.075" in capsys.readouterr().out


def test_the_ball_fit_weighs_the_players_speed_by_ball_state():
    exact = {k: dict(v) if isinstance(v, dict) else v for k, v in calibrate.TARGET.items()}
    assert calibrate.loss(exact) == 0.0
    slower = dict(exact, players_gap_15_p50=1.3 * calibrate.TARGET["players_gap_15_p50"])
    assert calibrate.loss(slower) == pytest.approx(calibrate.PLAYER_STATE_WEIGHT * np.log(1.3) ** 2)
    got = calibrate.summary([standing_half()], DegradeParams())
    assert set(got) == set(calibrate.TARGET)  # every target has its degraded side


def test_dead_weight_puts_the_long_gaps_at_stoppages():
    half = standing_half(n=40000)
    half["alive"] = np.arange(40000) % 4000 >= 2000  # 80 s stopped, then 80 s of play, in turn
    shares = []
    for weight in (0.0, 4.0):
        params = replace(DegradeParams(), dead_weight=weight)
        shares.append(calibrate.long_gap_dead_share(
            [half], calibrate.degraded_halves([half], params)))
    assert 0.0 <= shares[0] < shares[1] <= 1.0


def test_without_a_target_the_fit_only_reports(capsys):
    best, value = calibrate.fit_players([standing_half(n=1000)])
    assert best is None and value == float("inf")
    assert "no TARGET_PLAYERS yet" in capsys.readouterr().out
