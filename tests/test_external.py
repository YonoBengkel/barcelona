"""Label alignment on synthetic halves: which DFL possession changes survive."""

import numpy as np

from possession.external import AWAY, HOME, controlled_labels, flip_offsets


def labels(text):
    return np.array(list(text))


def test_touch_without_control_is_merged():
    raw = labels("HHHHAAHHHH")          # away touches the ball for two frames, no action
    out = controlled_labels(raw, np.array([1]), np.array([HOME]), tolerance=0)
    assert "".join(out) == "HHHHHHHHHH"


def test_controlled_change_is_kept():
    raw = labels("HHHHAAAAAA")
    out = controlled_labels(raw, np.array([1, 6]), np.array([HOME, AWAY]), tolerance=0)
    assert "".join(out) == "HHHHAAAAAA"


def test_restart_after_dead_ball_is_kept_without_actions():
    raw = labels("HHDDAAAA")
    out = controlled_labels(raw, np.array([], dtype=int), np.array([], dtype=np.int8),
                            tolerance=0)
    assert "".join(out) == "HHDDAAAA"


def test_tolerance_reaches_an_action_just_outside_the_run():
    raw = labels("HHHHAAHHHH")  # the away run covers frames 4 and 5
    # Home acts on frames 1 and 8; away is recorded one frame after its run's last frame.
    actions = (np.array([1, 6, 8]), np.array([HOME, AWAY, HOME]))
    assert "".join(controlled_labels(raw, *actions, tolerance=0)) == "HHHHHHHHHH"
    assert "".join(controlled_labels(raw, *actions, tolerance=1)) == "HHHHAAHHHH"


def test_flip_offsets_measure_from_the_change():
    raw = labels("HHHHAAAAAA")
    offsets = flip_offsets(raw, np.array([2, 7]), np.array([HOME, AWAY]))
    assert offsets.tolist() == [3]
