"""Turn per-frame evidence into whole possessions: a three-state hidden Markov model and Viterbi.

The states are H, A and D, coded 0, 1 and 2 exactly as evaluation/possession_metrics.py codes
them. A frame's evidence is the log-probability of each state from possession.models. A path
collects the evidence of the states it passes through and pays a cost each time it changes state:

- team_switch when the ball passes from one team to the other without a stoppage;
- dead_switch for a stoppage, half of it when play stops and half when it restarts. A stoppage
  inside a half pays both halves, so only their sum matters;
- dead_bias is added to the evidence for D on every frame, and moves how readily a path calls a
  stoppage.

Viterbi finds the path with the highest total. A short run of contrary frame calls cannot pay for
the two switches it would take, so the flicker goes and each possession gets one beginning and one
end. The three numbers are chosen on the practice ground (scripts/train_models.py), never on the
match. There is no X: an abstention inside a graded half is scored as a miss.

Two more numbers can hold every spell to a minimum length (M11j): team_min frames for a spell of
H or A, dead_min for a spell of D. Only a spell cut by the start or the end of the stretch may be
shorter. A run of contrary calls shorter than the minimum then cannot appear at all, whatever the
costs, so lower costs can let short possessions through without letting the flicker back. With
both at 1 the model is the plain one above, decoded by the same code as before.

viterbi() decodes many parameter sets in one pass over the frames, which is what makes a grid
search affordable. The arithmetic is float64 in a fixed order and ties go to the lower state (H,
then A, then D; with minima, to staying in the current spell first), so the same evidence gives the
same path on every machine.
"""

import numpy as np

STATES = ("H", "A", "D")
H, A, D = 0, 1, 2


def costs(team_switch, dead_switch):
    """(K, 3, 3) cost of moving from state i (axis 1) to state j (axis 2), for K parameter sets."""
    team_switch, dead_switch = np.broadcast_arrays(
        np.atleast_1d(np.asarray(team_switch, dtype=np.float64)),
        np.atleast_1d(np.asarray(dead_switch, dtype=np.float64)))
    out = np.zeros((len(team_switch), 3, 3))
    out[:, H, A] = out[:, A, H] = team_switch
    out[:, [H, A], D] = dead_switch[:, None] / 2
    out[:, D, [H, A]] = dead_switch[:, None] / 2
    return out


def viterbi(log_emission, team_switch, dead_switch, dead_bias, team_min=1, dead_min=1):
    """(frames, K) int8 best state path for each of K parameter sets, one column each.

    log_emission: (frames, 3) log-probabilities of H, A and D. team_switch, dead_switch and
    dead_bias: scalars, or arrays of one value per parameter set. team_min and dead_min: the
    fewest frames a spell of H or A, and of D, may last, one whole number each for the batch.
    """
    emission = np.asarray(log_emission, dtype=np.float64)
    if emission.ndim != 2 or emission.shape[1] != 3:
        raise ValueError("evidence must be (frames, 3)")
    if not np.isfinite(emission).all():
        raise ValueError("evidence must be finite")
    minimum = np.array([team_min, team_min, dead_min])
    if minimum.dtype.kind not in "iu" or (minimum < 1).any():
        raise ValueError("team_min and dead_min must be whole numbers of frames, 1 or more")
    cost = costs(team_switch, dead_switch)
    k, n = len(cost), len(emission)
    bias = np.zeros((k, 3))
    bias[:, D] = np.broadcast_to(np.asarray(dead_bias, dtype=np.float64), (k,))
    if n == 0:
        return np.zeros((0, k), dtype=np.int8)
    if (minimum == 1).all():
        return _plain(emission, cost, bias)
    return _held(emission, cost, bias, minimum)


def _plain(emission, cost, bias):
    """The plain model's paths: any spell may be as short as one frame."""
    k, n = len(cost), len(emission)
    path = np.zeros((n, k), dtype=np.int8)
    back = np.empty((n, k, 3), dtype=np.int8)
    score = emission[0] + bias
    for t in range(1, n):
        candidates = score[:, :, None] - cost  # (K, from, to)
        back[t] = candidates.argmax(axis=1)
        score = candidates.max(axis=1) + (emission[t] + bias)
        score -= score.max(axis=1, keepdims=True)  # keep the numbers small; the path is unchanged

    state = score.argmax(axis=1)
    path[n - 1] = state
    rows = np.arange(k)
    for t in range(n - 1, 0, -1):
        state = back[t, rows, state]
        path[t - 1] = state
    return path


def _held(emission, cost, bias, minimum):
    """The paths when a spell of state s lasts at least minimum[s] frames (an int array of 3).

    An explicit-duration model with a geometric tail. entries[s] is the best score of a path whose
    spell of s begins at frame t, held[s] that of a path whose spell of s has lasted its minimum by
    t, or began the stretch. A spell that reaches its minimum at t began at t - minimum[s] + 1, and
    its evidence in between comes from running sums, so a frame costs the same at any minimum. A
    spell follows only a spell of another state. The scores are not renormalised: float64 holds a
    match's worth of evidence.
    """
    k, n = len(cost), len(emission)
    rows, states = np.arange(k), np.arange(3)
    span = int(minimum.max())
    running = np.cumsum(emission, axis=0)  # running[t, s]: the evidence of s over frames 0..t
    tail = (minimum - 1) * bias  # the bias on the frames of a spell after its first
    entries = np.full((span, k, 3), -np.inf)  # spells begun at the last span frames, by t % span
    entered_from = np.zeros((n, k, 3), dtype=np.int8)
    completed = np.zeros((n, k, 3), dtype=bool)
    own = np.where(np.eye(3, dtype=bool), -np.inf, 0.0)  # a spell never follows its own state
    held = emission[0] + bias
    for t in range(1, n):
        frame = emission[t] + bias
        candidates = held[:, :, None] - cost + own  # (K, from, to)
        entered_from[t] = candidates.argmax(axis=1)
        entries[t % span] = candidates.max(axis=1) + frame
        began = t - minimum + 1
        reached = entries[began % span, :, states].T + (
            running[t] - running[np.maximum(began, 0), states]) + tail
        if t < span:
            reached[:, began < 1] = -np.inf  # no spell begins before the second frame
        stay = held + frame
        completed[t] = reached > stay
        held = np.where(completed[t], reached, stay)

    # The best ending: a spell that has lasted its minimum, or a last spell cut short by the end.
    best, cut = held.copy(), np.full((k, 3), -1)  # cut: where that last spell began, -1 if none
    for s in states:
        for frame_no in range(max(1, n - int(minimum[s]) + 1), n):
            score = entries[frame_no % span, :, s] \
                + (running[n - 1, s] - running[frame_no, s]) + (n - 1 - frame_no) * bias[:, s]
            better = score > best[:, s]
            best[better, s] = score[better]
            cut[better, s] = frame_no
    state = best.argmax(axis=1)
    last_began = cut[rows, state]

    path = np.empty((n, k), dtype=np.int8)
    inside = last_began >= 0  # in a spell short of its minimum, begun this many frames before:
    left = np.where(inside, n - 1 - last_began, 0)
    for t in range(n - 1, -1, -1):
        path[t] = state
        if t == 0:
            break
        # A spell that reached its minimum at t began minimum - 1 frames before.
        done = ~inside & completed[t, rows, state]
        left = np.where(done, minimum[state] - 1, left)
        inside = inside | done
        # At the frame a spell began, the frame before belongs to the spell it followed.
        first = inside & (left == 0)
        state = np.where(first, entered_from[t, rows, state], state)
        inside = inside & ~first
        left = np.where(inside, left - 1, left)
    return path


def decode(log_emission, params):
    """'H', 'A' or 'D' per frame for one parameter set: team_switch, dead_switch, dead_bias and,
    when the model file holds them, team_min and dead_min."""
    path = viterbi(log_emission, params["team_switch"], params["dead_switch"],
                   params["dead_bias"], params.get("team_min", 1), params.get("dead_min", 1))[:, 0]
    return np.array(STATES)[path]
