"""Stage two of a stacked model (M11l): trees that read the first stage's log-odds around each
frame and give corrected dead-ball and team log-odds.

The perfect-evidence check (M11c) showed that the team call holds most of the score still
missing, yet four changes (one-second context, M11a; last-touch features, M11i; minimum spell
lengths in the decoder, M11j; twice the training copies, M11k) each moved held-out spellF1 by less
than 0.002. The first stage judges every frame from features of its neighbourhood, but it
never sees how its own confidence moves around the frame. Stage two does: a brief dip of the team
log-odds inside a long, confident possession looks different from a real change, after which the
log-odds stay on the other side for seconds. At every frame it reads the two log-odds of the first
stage (dead and home, as possession.models.score gives them), their values 0.2 to 2 s earlier and
later, their centred means over about 1 to 10 s, and the lowest and highest team log-odds within
about 2 and 5 s. It gives (dead, home) again, which possession.models turns into the decoder's
evidence as before.

The settings below were fixed before any run and are never tuned on a held-out match.
scripts/train_stacking.py fits stage two on first-stage log-odds of matches the first stage never
saw, nested inside the leave-one-match-out loop, so it learns from evidence as uncertain as the
evidence it will correct.

features(dead, -home) is exactly the home-away mirror of features(dead, home): every column made
from home is negated, its minimum and maximum swap, and the dead-ball columns stay. Every training
row is used both ways (rows), as the first stage's team model sees every half both ways, so stage
two treats both sides alike.

scikit-learn fits; apply() needs numpy alone. features() is made of shifts, cumulative sums
(possession.features.rolling_mean) and exact minima and maxima, and the trees are walked by
possession.models.score_trees in a fixed order: no matrix products, so the numbers are the same on
any machine.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from possession.features import rolling_mean

# Fixed before any run (M11l), never tuned on a held-out match. In frames, at 25 fps.
OFFSETS = (5, 12, 25, 50)  # lags and leads: 0.2, 0.48, 1 and 2 s
WINDOWS = (12, 25, 62, 125)  # half-widths of the centred means: windows of about 1, 2, 5 and 10 s
EXTREMES = (25, 62)  # half-widths of the team log-odds' centred minimum and maximum: 2 and 5 s
TREES = {"max_iter": 200, "max_leaf_nodes": 31, "learning_rate": 0.05, "min_samples_leaf": 200}
STEP = 4  # every 4th frame of a training half becomes a row


def names():
    """The columns features() makes, in order: the two log-odds, their lags and leads, their
    centred means, then the team log-odds' centred minimum and maximum."""
    out = ["dead", "home"]
    for offset in OFFSETS:
        out += ["dead_lag{}".format(offset), "dead_lead{}".format(offset),
                "home_lag{}".format(offset), "home_lead{}".format(offset)]
    for window in WINDOWS:
        out += ["dead_mean{}".format(window), "home_mean{}".format(window)]
    for window in EXTREMES:
        out += ["home_min{}".format(window), "home_max{}".format(window)]
    return tuple(out)


def lag(values, offset):
    """The value offset frames earlier; NaN on the first offset frames, before the half began."""
    out = np.full(len(values), np.nan)
    if offset < len(values):
        out[offset:] = values[:len(values) - offset]
    return out


def lead(values, offset):
    """The value offset frames later; NaN on the last offset frames, after the half has ended."""
    out = np.full(len(values), np.nan)
    if offset < len(values):
        out[:len(values) - offset] = values[offset:]
    return out


def rolling_min(values, window):
    """Centred minimum over 2 * window + 1 frames, the window clipped at the half's edges: the
    half is padded with +inf, which no minimum takes. A minimum is exact, whatever the order."""
    values = np.asarray(values, dtype=np.float64)
    edge = np.full(window, np.inf)
    padded = np.concatenate([edge, values, edge])
    return sliding_window_view(padded, 2 * window + 1).min(axis=1)


def rolling_max(values, window):
    """Centred maximum, as rolling_min: the negated minimum of the negated values, so that the
    maximum of a series is exactly the negated minimum of its mirror."""
    return -rolling_min(-np.asarray(values, dtype=np.float64), window)


def features(dead, home):
    """(frames, len(names())) float32: one half's stage-two features, in names() order, from its
    first-stage log-odds dead and home.

    Computed in float64 and cast once at the end, because the trees are fitted on float32 rows
    and must be applied to the same numbers. Lags and leads are NaN where they reach beyond the
    half, which the trees route like any missing value; means, minima and maxima are taken over
    the part of their window inside the half.
    """
    dead = np.asarray(dead, dtype=np.float64)
    home = np.asarray(home, dtype=np.float64)
    columns = {"dead": dead, "home": home}
    for offset in OFFSETS:
        for name, values in (("dead", dead), ("home", home)):
            columns["{}_lag{}".format(name, offset)] = lag(values, offset)
            columns["{}_lead{}".format(name, offset)] = lead(values, offset)
    for window in WINDOWS:
        columns["dead_mean{}".format(window)] = rolling_mean(dead, window)
        columns["home_mean{}".format(window)] = rolling_mean(home, window)
    for window in EXTREMES:
        columns["home_min{}".format(window)] = rolling_min(home, window)
        columns["home_max{}".format(window)] = rolling_max(home, window)
    return np.column_stack([columns[name] for name in names()]).astype(np.float32)


def rows(halves, step=STEP):
    """Stage two's training rows: ((X, y) of the dead-ball call, (X, y) of the team call), X
    float32 and y bool.

    halves: (dead, home, labels) per half, its first-stage log-odds and its 'H', 'A', 'D' labels.
    Frames 0, step, 2 * step, ... of each half are kept, each twice: as it is, features(dead,
    home), and mirrored, features(dead, -home). A dead-ball row is labelled dead or not, the same
    both ways. A team row, on live frames only, is labelled home as it is and away mirrored.
    """
    dead_X, dead_y, team_X, team_y = [], [], [], []
    for dead, home, labels in halves:
        labels = np.asarray(labels)
        home = np.asarray(home, dtype=np.float64)
        kept = np.arange(0, len(labels), step)
        live = kept[labels[kept] != "D"]
        as_is, mirrored = features(dead, home), features(dead, -home)
        dead_X += [as_is[kept], mirrored[kept]]
        dead_y += [labels[kept] == "D"] * 2
        team_X += [as_is[live], mirrored[live]]
        team_y += [labels[live] == "H", labels[live] == "A"]
    return ((np.concatenate(dead_X), np.concatenate(dead_y)),
            (np.concatenate(team_X), np.concatenate(team_y)))


def settings():
    """The fixed settings above, as a model file records them."""
    return {"offsets": list(OFFSETS), "windows": list(WINDOWS), "extremes": list(EXTREMES),
            "trees": dict(TREES), "step": STEP}


def fit(halves):
    """The JSON-ready stage two, fitted on halves as rows() takes them: {'kind': 'stack', 'names',
    'settings', 'dead', 'team'}, each call a possession.models.fit_trees model with TREES, its
    'names' set to names() (fit_trees would write the first stage's design)."""
    from possession.models import fit_trees

    dead_rows, team_rows = rows(halves, STEP)
    stack = {"kind": "stack", "names": list(names()), "settings": settings()}
    for call, (X, y) in (("dead", dead_rows), ("team", team_rows)):
        stack[call] = dict(fit_trees(X, y, **TREES), names=list(names()))
    return stack


def apply(stack, dead, home):
    """(dead, home): one half's first-stage log-odds corrected by stage two, float64, with numpy
    alone (possession.models.score_trees)."""
    from possession.models import score_trees

    X = features(dead, home)
    return score_trees(stack["dead"], X), score_trees(stack["team"], X)
