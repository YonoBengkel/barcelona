"""Lab, after the competition: other definitions of possession for the IDSSE practice labels.

The competition's pipeline learned 'controlled' labels (possession.external.controlled_labels): a
team's run counts as a possession when the team makes a controlled action inside it. Two other
readings of the same DFL data, fixed before any run:

- 'raw': DFL's ball possession as it is, which changes on any touch;
- 'strict': the controlled labels, then every team run shorter than STRICT_SECONDS that holds fewer
  than STRICT_ACTIONS controlled actions of its team (widened by the tolerance) is merged into the
  possession it interrupted, unless it follows a dead ball or opens the half. A reading of the
  challenge's own proxies for control: a player keeping the ball for a few seconds, or a team
  completing more than two consecutive passes.

aligned(name) has the signature of possession.external.aligned_labels, and active(name) swaps it in
for a block, so the practice ground (possession.practice.cases) and the decoder's scoring
(scripts/train_models.labels_at) both see that definition. Dead frames are the DFL's in all three.
"""

import contextlib

import numpy as np

from possession import external as ex

DEFINITIONS = ("raw", "controlled", "strict")
STRICT_SECONDS = 3.0
STRICT_ACTIONS = 2
ORIGINAL = ex.aligned_labels  # the competition's labels, kept before any swap


def strict_labels(labels, frames, teams, tolerance, min_frames, min_actions):
    """Labels with short, thinly controlled team runs merged into the possession they interrupt.

    labels: one half's controlled labels. frames, teams: that half's controlled actions on its
    frame index, offset-corrected. A run is kept when it opens the half or follows a dead ball,
    lasts min_frames or more, or holds min_actions or more of its team's actions within tolerance
    frames of it; otherwise it takes the label of the possession before it.
    """
    out = np.array(labels, copy=True)
    owner = None
    by_team = {"H": np.sort(frames[teams == ex.HOME]), "A": np.sort(frames[teams == ex.AWAY])}
    for start, stop, label in ex.runs(np.asarray(labels)):
        if label == "D":
            owner = None
            continue
        actions = by_team[label]
        low, high = np.searchsorted(actions, [start - tolerance, stop + tolerance])
        if owner is None or stop - start >= min_frames or high - low >= min_actions:
            owner = label
        else:
            out[start:stop] = owner
    return out


def aligned(name, min_seconds=STRICT_SECONDS, min_actions=STRICT_ACTIONS):
    """A function with the signature and return of possession.external.aligned_labels that gives
    the labels of one definition: ({section: labels}, shift, offsets)."""
    if name not in DEFINITIONS:
        raise ValueError("unknown definition {!r}; one of {}".format(name, DEFINITIONS))

    def labels_for(halves, events, tolerance_s):
        if name == "raw":
            return {s: ex.raw_labels(halves[s]) for s in ex.SECTIONS}, 0, np.array([], dtype=int)
        labels, shift, offsets = ORIGINAL(halves, events, tolerance_s)
        if name == "controlled":
            return labels, shift, offsets
        times, teams, controlled = events
        times, teams = times[controlled], teams[controlled]
        tolerance = int(round(tolerance_s * ex.FPS))
        out = {}
        for section in ex.SECTIONS:
            half = halves[section]
            frames = np.round((times - half["t0"]) * ex.FPS).astype(int)
            inside = (frames >= 0) & (frames < len(half["alive"]))
            out[section] = strict_labels(labels[section], frames[inside] - shift, teams[inside],
                                         tolerance, int(round(min_seconds * ex.FPS)), min_actions)
        return out, shift, offsets

    return labels_for


@contextlib.contextmanager
def active(name):
    """Within the block, possession.external.aligned_labels gives definition name's labels."""
    ex.aligned_labels = aligned(name)
    try:
        yield
    finally:
        ex.aligned_labels = ORIGINAL
