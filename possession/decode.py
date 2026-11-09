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

viterbi() decodes many parameter sets in one pass over the frames, which is what makes a grid
search affordable. The arithmetic is float64 in a fixed order and ties go to the lower state (H,
then A, then D), so the same evidence gives the same path on every machine.
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


def viterbi(log_emission, team_switch, dead_switch, dead_bias):
    """(frames, K) int8 best state path for each of K parameter sets, one column each.

    log_emission: (frames, 3) log-probabilities of H, A and D. team_switch, dead_switch and
    dead_bias: scalars, or arrays of one value per parameter set.
    """
    emission = np.asarray(log_emission, dtype=np.float64)
    if emission.ndim != 2 or emission.shape[1] != 3:
        raise ValueError("evidence must be (frames, 3)")
    if not np.isfinite(emission).all():
        raise ValueError("evidence must be finite")
    cost = costs(team_switch, dead_switch)
    k, n = len(cost), len(emission)
    bias = np.zeros((k, 3))
    bias[:, D] = np.broadcast_to(np.asarray(dead_bias, dtype=np.float64), (k,))
    path = np.zeros((n, k), dtype=np.int8)
    if n == 0:
        return path

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


def decode(log_emission, params):
    """'H', 'A' or 'D' per frame for one parameter set: team_switch, dead_switch, dead_bias."""
    path = viterbi(log_emission, params["team_switch"], params["dead_switch"],
                   params["dead_bias"])[:, 0]
    return np.array(STATES)[path]
