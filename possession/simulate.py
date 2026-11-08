"""Degrade complete tracking until it looks like the match's automatic tracking.

IDSSE's tracking is corrected: the ball and every player are on every frame, with the ball's
height. The match's automatic tracking loses the ball on about 41% of graded frames, mostly in
crowds and in the air (docs/challenge/DATA.md), tracks about 20.5 players instead of 22, has no
ball height, and is noisier. A model chosen on clean tracking would count on a ball it will not
get, so every choice is made on degraded copies instead.

The ball disappears in gaps. A gap starts on a frame with a probability proportional to a risk
that rises with the players crowding the ball and with its height, and lasts a log-normally
distributed time. The overall rate is solved for, so the hidden share matches the target whatever
the risk weights are. While the ball is outside the lines (62-74% of dead-ball frames in IDSSE),
whole stretches can be hidden as well, in case the match's tracker does not follow it there.
Players disappear in gaps too, without the risk term. Noise drifts over about a second rather
than jumping every frame: at 25 fps, 0.3 m of independent noise per frame alone would read as a
10 m/s ball. Every number lives in DegradeParams, so calibrating against the match's profile
(scripts/atd_profile.py) never touches the code.
"""

from dataclasses import dataclass

import numpy as np

FPS = 25


@dataclass(frozen=True)
class DegradeParams:
    # Share of frames without a ball: 41% of graded frames per docs/challenge/DATA.md.
    ball_missing_share: float = 0.407
    # Gap length, log-normal: median and spread. Placeholders until the ATD profile is in.
    gap_median_s: float = 0.6
    gap_sigma: float = 1.3
    # Log-risk added per player within crowd_radius_m of the ball, and when the ball is above
    # height_m (in the air, where the match's tracking loses it most).
    crowd_radius_m: float = 2.0
    crowd_weight: float = 0.3
    height_m: float = 2.0
    height_weight: float = 1.5
    # About 20.5 of 22 players tracked on an average graded frame (docs/challenge/DATA.md).
    players_tracked: float = 20.5
    player_gap_median_s: float = 2.0
    player_gap_sigma: float = 1.0
    # Chance that a stretch with the ball outside the lines stays visible. 1.0 until the match's
    # profile says how often its tracker sees the ball there.
    outside_visible: float = 1.0
    pitch: tuple = (105.0, 68.0)
    # Position noise in metres, drifting over noise_window_s.
    ball_noise_m: float = 0.2
    player_noise_m: float = 0.15
    noise_window_s: float = 1.0


def _smooth_noise(rng, shape, sigma, window):
    """Noise of standard deviation sigma that drifts over `window` frames instead of jumping.

    A moving sum of `window` white samples, divided by sqrt(window) to keep unit variance.
    """
    if sigma <= 0:
        return np.zeros(shape, dtype=np.float32)
    white = np.cumsum(rng.standard_normal((shape[0] + window,) + tuple(shape[1:])), axis=0)
    return (sigma * (white[window:] - white[:-window]) / np.sqrt(window)).astype(np.float32)


def _outside_mask(half, params, rng):
    """Whole stretches with the ball outside the lines, each hidden with 1 - outside_visible."""
    length, width = params.pitch
    outside = (np.abs(half["ball_x"]) > length / 2) | (np.abs(half["ball_y"]) > width / 2)
    hidden = np.zeros(len(outside), dtype=bool)
    if params.outside_visible >= 1.0:
        return hidden
    breaks = np.flatnonzero(outside[1:] != outside[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [len(outside)]))
    for start, stop in zip(bounds[:-1], bounds[1:]):
        if outside[start] and rng.random() >= params.outside_visible:
            hidden[start:stop] = True
    return hidden


def _gap_mask(n, weights, share, median, sigma, rng, columns=None, base=None):
    """Boolean (n,) or (n, columns) mask of gap frames covering about `share` of the frames.

    Gap starts are Bernoulli draws with probability rate * weights, lengths log-normal. The rate
    is found by bisection on fixed random numbers, so coverage rises monotonically with it and
    the result is reproducible from the seed. Frames already hidden by `base` count towards the
    share, and are part of the returned mask.
    """
    shape = (n,) if columns is None else (n, columns)
    u = rng.random(shape)
    lengths = np.maximum(1, np.round(median * np.exp(sigma * rng.standard_normal(shape))))
    weights = np.broadcast_to(weights if columns is None else weights[:, None], shape)

    already = np.zeros(shape, dtype=bool) if base is None else base

    def coverage(rate):
        starts = u < np.minimum(1.0, rate * weights)
        cover = np.zeros((n + 1,) + shape[1:], dtype=np.int32)
        index = np.nonzero(starts)
        np.add.at(cover, index, 1)
        end = (np.minimum(index[0] + lengths[index].astype(int), n),) + index[1:]
        np.add.at(cover, end, -1)
        return (np.cumsum(cover, axis=0)[:n] > 0) | already

    if already.mean() >= share:
        return already

    low, high = 0.0, 1.0 / max(float(weights.min()), 1e-9)
    for _ in range(40):
        middle = (low + high) / 2
        if coverage(middle).mean() < share:
            low = middle
        else:
            high = middle
    return coverage(high)


def degrade(half, params=DegradeParams(), seed=0):
    """A degraded copy of one half (dict of arrays as possession.external returns it).

    The ball and the players are hidden in gaps and jittered; 'ball_z' is dropped, because the
    match's tracking has no height. Labels ('alive', 'owner') are untouched: they are the truth.
    """
    rng = np.random.default_rng(seed)
    n = len(half["ball_x"])
    out = {key: value for key, value in half.items() if key != "ball_z"}

    dx = half["x"] - half["ball_x"][:, None]
    dy = half["y"] - half["ball_y"][:, None]
    near = np.sum(dx * dx + dy * dy <= params.crowd_radius_m ** 2, axis=1)
    high = np.nan_to_num(half.get("ball_z", np.zeros(n)), nan=0.0) > params.height_m
    risk = np.exp(params.crowd_weight * near + params.height_weight * high)
    hidden = _gap_mask(n, risk, params.ball_missing_share, params.gap_median_s * FPS,
                       params.gap_sigma, rng, base=_outside_mask(half, params, rng))

    window = max(1, int(round(params.noise_window_s * FPS)))
    noise = _smooth_noise(rng, (n, 2), params.ball_noise_m, window)
    out["ball_x"] = np.where(hidden, np.nan, half["ball_x"] + noise[:, 0]).astype(np.float32)
    out["ball_y"] = np.where(hidden, np.nan, half["ball_y"] + noise[:, 1]).astype(np.float32)

    on_pitch = ~np.isnan(half["x"])
    on_pitch_count = on_pitch.sum(axis=1).mean() if on_pitch.size else 0.0
    share = max(0.0, 1.0 - params.players_tracked / max(on_pitch_count, 1e-9))
    lost = _gap_mask(n, np.ones(n), share, params.player_gap_median_s * FPS,
                     params.player_gap_sigma, rng, columns=half["x"].shape[1]) \
        if share > 0 else np.zeros_like(on_pitch)
    jitter_x = _smooth_noise(rng, half["x"].shape, params.player_noise_m, window)
    jitter_y = _smooth_noise(rng, half["x"].shape, params.player_noise_m, window)
    out["x"] = np.where(lost, np.nan, half["x"] + jitter_x).astype(np.float32)
    out["y"] = np.where(lost, np.nan, half["y"] + jitter_y).astype(np.float32)
    return out
