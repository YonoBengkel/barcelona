"""Predictors: a half of tracking in, one H/A/D/X label per frame out.

The first is the organisers' baseline on our arrays: the team of the tracked player nearest the
ball, then a majority vote over a centred window (starter/baseline.py), so that the practice
ground can be checked against the baseline scores the organisers published for the match.
"""

import os
import sys

import numpy as np

from possession.external import ROOT

STARTER_DIR = os.path.join(ROOT, "starter")
if STARTER_DIR not in sys.path:
    sys.path.insert(0, STARTER_DIR)
from baseline import NOBODY, smooth  # noqa: E402  (the organisers' smoothing, reused as is)


def nearest_team(half):
    """Per frame: team (0 home, 1 away) of the tracked player nearest the ball, NOBODY without one."""
    dx = half["x"] - half["ball_x"][:, None]
    dy = half["y"] - half["ball_y"][:, None]
    distance = np.where(np.isnan(dx) | np.isnan(dy), np.inf, dx * dx + dy * dy)
    nearest = np.argmin(distance, axis=1)
    seen = ~np.isnan(half["ball_x"]) & np.isfinite(distance.min(axis=1))
    return np.where(seen, half["team"][nearest], NOBODY).astype(np.int8)


def baseline(window):
    """The organisers' baseline with a +/- window majority vote; window 0 is the raw variant."""
    def predict(half):
        return smooth(nearest_team(half), window)
    predict.__name__ = "baseline_raw" if not window else "baseline_smooth{}".format(window)
    return predict
