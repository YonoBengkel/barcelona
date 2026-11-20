"""possession.lab_labels: the strict merge, the three definitions on a synthetic match, and the swap
that lets the practice ground and the decoder see one definition at a time."""

import numpy as np
import pytest

from possession import external as ex
from possession import lab_labels


def letters(text):
    return np.array(list(text))


def test_a_short_run_with_one_action_is_merged_into_the_possession_it_interrupts():
    labels = letters("HHHHHHAAAHHHHHH")
    frames, teams = np.array([1, 7, 12]), np.array([ex.HOME, ex.AWAY, ex.HOME])
    out = lab_labels.strict_labels(labels, frames, teams, 0, 5, 2)
    assert "".join(out) == "H" * 15


def test_a_run_is_kept_when_long_or_controlled_twice_or_after_a_dead_ball():
    labels = letters("HHHHHHAAAHHHHHH")
    twice = lab_labels.strict_labels(labels, np.array([1, 6, 8]),
                                     np.array([ex.HOME, ex.AWAY, ex.AWAY]), 0, 5, 2)
    assert "".join(twice) == "HHHHHHAAAHHHHHH"
    long_run = letters("HHHAAAAAAHHHHH")
    kept = lab_labels.strict_labels(long_run, np.array([4]), np.array([ex.AWAY]), 0, 5, 2)
    assert "".join(kept) == "HHHAAAAAAHHHHH"
    restart = letters("DDDAAHHHHH")
    after = lab_labels.strict_labels(restart, np.array([], dtype=int), np.array([], dtype=int),
                                     0, 5, 2)
    assert "".join(after) == "DDDAAHHHHH"


def synthetic_match():
    """Two halves of 200 live frames: H, then a 10-frame A run with one controlled A action, then
    H again; H's controlled actions spread through each half."""
    halves, times, teams = {}, [], []
    for number, section in enumerate(ex.SECTIONS):
        t0 = 1000.0 * (number + 1)
        owner = np.full(200, ex.HOME, dtype=np.int8)
        owner[100:110] = ex.AWAY
        halves[section] = {"alive": np.ones(200, dtype=bool), "owner": owner, "t0": t0}
        for frame, team in ((10, ex.HOME), (50, ex.HOME), (105, ex.AWAY), (120, ex.HOME),
                            (150, ex.HOME)):
            times.append(t0 + frame / ex.FPS)
            teams.append(team)
    events = (np.array(times), np.array(teams, dtype=np.int8), np.ones(len(times), dtype=bool))
    return halves, events


def test_the_three_definitions_on_a_synthetic_match():
    halves, events = synthetic_match()
    raw, _, _ = lab_labels.aligned("raw")(halves, events, 1.0)
    controlled, shift, _ = lab_labels.aligned("controlled")(halves, events, 1.0)
    strict, strict_shift, _ = lab_labels.aligned("strict")(halves, events, 1.0)
    expected, _, _ = ex.aligned_labels(halves, events, 1.0)
    for section in ex.SECTIONS:
        assert (raw[section][100:110] == "A").all() and (raw[section][:100] == "H").all()
        assert (controlled[section] == expected[section]).all()
        assert (controlled[section][100:110] == "A").all()  # one controlled action keeps it
        assert (strict[section] == "H").all()  # but it is short and thinly controlled
    assert strict_shift == shift
    with pytest.raises(ValueError):
        lab_labels.aligned("loose")


def test_active_swaps_the_definition_in_and_out():
    halves, events = synthetic_match()
    original = ex.aligned_labels
    with lab_labels.active("strict"):
        assert ex.aligned_labels is not original
        labels, _, _ = ex.aligned_labels(halves, events, 1.0)
        assert (labels[ex.SECTIONS[0]] == "H").all()
    assert ex.aligned_labels is original
    with pytest.raises(RuntimeError):
        with lab_labels.active("raw"):
            raise RuntimeError("the block fails")
    assert ex.aligned_labels is original
