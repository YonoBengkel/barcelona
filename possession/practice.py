"""The practice ground: degraded IDSSE halves, labels aligned with the challenge, official scoring.

Every model and parameter choice is made here, never on the match (AGENTS.md). A case is one half
of one IDSSE match, degraded by possession.simulate to the match's tracking quality, with labels
from DFL's ball status and possession aligned to the challenge's controlled-possession definition
by possession.external.controlled_labels. Scores come from the organisers' own
evaluation/possession_metrics.py, so a number here means what a number on the board means.
"""

import os
import sys

import numpy as np

from possession import external as ex
from possession.simulate import DegradeParams, degrade

EVALUATION_DIR = os.path.join(ex.ROOT, "evaluation")
if EVALUATION_DIR not in sys.path:
    sys.path.insert(0, EVALUATION_DIR)
import possession_metrics as pm  # noqa: E402  (the organisers' metric library)


def cases(match_ids, params=DegradeParams(), tolerance_s=1.0, seed=0):
    """[{'match', 'section', 'half', 'labels'}], one per half, degraded reproducibly from seed."""
    out = []
    for i, match_id in enumerate(match_ids):
        _, halves, events = ex.load_match(match_id)
        labels, _, _ = ex.aligned_labels(halves, events, tolerance_s)
        for j, section in enumerate(ex.SECTIONS):
            out.append({"match": match_id, "section": section, "labels": labels[section],
                        "half": degrade(halves[section], params, seed=seed * 1000 + 10 * i + j)})
    return out


def joined(arrays):
    """One array from per-half label arrays, with an X between halves.

    The scorer never grades X and never lets a possession cross it, which is exactly how it
    treats the halftime gap of the match.
    """
    out = []
    for array in arrays:
        if out:
            out.append(np.array(["X"]))
        out.append(np.asarray(array))
    return np.concatenate(out)


def score(cases_, predictions):
    """(pooled result, {match: result}) from possession_metrics.evaluate.

    pooled scores every half of every match as one long submission, so each possession counts once
    across the whole practice ground; per match is the spread.
    """
    per_match = {}
    for match in dict.fromkeys(c["match"] for c in cases_):
        index = [i for i, c in enumerate(cases_) if c["match"] == match]
        gt = joined([cases_[i]["labels"] for i in index])
        pred = joined([predictions[i] for i in index])
        per_match[match] = pm.evaluate(pm.label_codes(gt), pm.label_codes(pred))
    gt = joined([c["labels"] for c in cases_])
    pred = joined(predictions)
    return pm.evaluate(pm.label_codes(gt), pm.label_codes(pred)), per_match
