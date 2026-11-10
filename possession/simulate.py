"""Degrade complete tracking until it looks like the match's automatic tracking.

IDSSE's tracking is corrected: the ball and every player are on every frame, with the ball's
height. The match's automatic tracking (ATD) loses the ball on 40.7% of graded frames, tracks
about 20.5 players instead of 22, has no ball height, and is noisier. A model chosen on clean
tracking would count on a ball it will not get, so every choice is made on degraded copies.

How the ATD loses the ball, from its profile (scripts/atd_profile.py, aggregates only): rarely
and for long -- 150 gaps in 106 minutes, median 10.7 s, 85% of missing frames inside gaps of 10 s
or more -- and mostly where play stops. A gap is 3.4 times likelier to start within 2 m of a line
than the ball is to be seen there, 2.3 times likelier in a penalty box, and starts on a slow ball
(median 1.5 m/s against 5.1 m/s when seen); the ball is seen outside the lines on only 4.2% of
seen frames, against 30% of IDSSE's frames.

So a gap starts on a frame with a probability proportional to a risk built from exactly those
covariates, plus whether IDSSE marks the ball dead there (the slow, near-the-line pattern above is
what dead balls look like), and lasts a time drawn from a mixture of short flickers and long
losses. How much each covariate weighs is fitted to the profile, never to the match's labels. The overall
rate is solved for, so the hidden share matches the target whatever the weights. Stretches with the
ball outside the lines can be hidden whole. Players vanish in gaps too, without the risk term.
Noise drifts over about a second rather than jumping every frame: at 25 fps, 0.3 m of
independent noise per frame alone would read as a 10 m/s ball. Only the ball gets it: fitted to
the match's player speeds, the players' noise came out at none (M11g), because IDSSE's own
tracking already moves about as the match's does.

Every number lives in DegradeParams; scripts/calibrate_simulator.py fits them to the profile. The
defaults below are its fit of 2026-11-08 (loss 16.1 -> 3.1): gap lengths, seen-run lengths,
penalty-box share and crowding match the match's tracking closely. Two do not, and no risk weight
fixes them: the ball before a gap moves at a median 4.3 m/s against the match's 1.5 m/s (its
tracker may hold the last position briefly before giving up, which a risk cannot imitate), and
the seen ball is within 2 m of a line on 18% of frames against 9%.
"""

from dataclasses import dataclass

import numpy as np

FPS = 25


@dataclass(frozen=True)
class DegradeParams:
    # Share of frames without a ball: 40.7% of graded frames, docs/challenge/DATA.md and profile.
    ball_missing_share: float = 0.407
    # Gap lengths: a log-normal mixture of short flickers and long losses.
    short_gap_share: float = 0.167
    short_gap_median_s: float = 0.4
    short_gap_sigma: float = 0.7
    long_gap_median_s: float = 8.0
    long_gap_sigma: float = 0.630
    # Log-risk terms for a gap starting on a frame.
    line_m: float = 2.0
    line_weight: float = 0.038       # ball within line_m of a touchline or goal line
    box_weight: float = 0.801        # ball inside a penalty box
    slow_mps: float = 2.0
    slow_weight: float = 0.405       # ball slower than slow_mps
    crowd_radius_m: float = 2.0
    crowd_weight: float = 0.716      # per player within crowd_radius_m of the ball
    # Ball out of play (IDSSE's own label). The fit chose none, and so did M11h's scan against the
    # players' speeds by ball state: a gap's length ignores the stoppage, so long gaps outlast it.
    dead_weight: float = 0.0
    # Chance that a gap opening while the ball is out of play lasts until play restarts instead of
    # a drawn length (M11h). At none the random draws are exactly those of the simulator before.
    until_restart: float = 0.0
    # Chance that a stretch with the ball outside the lines stays visible.
    outside_visible: float = 0.187
    pitch: tuple = (105.0, 68.0)
    # About 20.5 of 22 players tracked on an average graded frame.
    players_tracked: float = 20.5
    player_gap_median_s: float = 2.0
    player_gap_sigma: float = 1.0
    # Position noise in metres, drifting over noise_window_s. The players get none since M11g:
    # fitted to the players' speed percentiles of the match's tracking
    # (scripts/calibrate_simulator.py --players), any added noise only moved IDSSE further from
    # them; 0.15 m, the value before, read as about 1.3 m/s for a player standing still.
    ball_noise_m: float = 0.2
    player_noise_m: float = 0.0
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


def _cover(starts, lengths, n):
    """Union of [start, start + length) intervals, along axis 0, as a boolean mask."""
    shape = starts.shape
    cover = np.zeros((n + 1,) + shape[1:], dtype=np.int32)
    index = np.nonzero(starts)
    np.add.at(cover, index, 1)
    end = (np.minimum(index[0] + lengths[index].astype(int), n),) + index[1:]
    np.add.at(cover, end, -1)
    return np.cumsum(cover, axis=0)[:n] > 0


def _gap_mask(weights, share, lengths, rng, base=None):
    """Mask covering about `share` of the frames with gaps starting where weights are high.

    Gap starts are Bernoulli draws with probability rate * weights; `lengths` holds the length a
    gap would have if it started on each frame. The rate is found by bisection on fixed random
    numbers, so coverage rises monotonically with it and the result is reproducible from the
    seed. Frames already hidden by `base` count towards the share and stay hidden.
    """
    n = weights.shape[0]
    u = rng.random(weights.shape)
    already = np.zeros(weights.shape, dtype=bool) if base is None else base
    if already.mean() >= share:
        return already

    def coverage(rate):
        return _cover(u < np.minimum(1.0, rate * weights), lengths, n) | already

    low, high = 0.0, 1.0 / max(float(weights.min()), 1e-12)
    for _ in range(40):
        middle = (low + high) / 2
        if coverage(middle).mean() < share:
            low = middle
        else:
            high = middle
    return coverage(high)


def ball_risk(half, params):
    """Relative chance, per frame, that the tracking loses the ball there."""
    length, width = params.pitch
    bx, by = half["ball_x"], half["ball_y"]
    near_line = (np.abs(bx) > length / 2 - params.line_m) | (np.abs(by) > width / 2 - params.line_m)
    in_box = (np.abs(bx) > length / 2 - 16.5) & (np.abs(bx) <= length / 2) & (np.abs(by) < 20.16)
    step = np.hypot(np.diff(bx, prepend=bx[:1]), np.diff(by, prepend=by[:1])) * FPS
    speed = np.convolve(step, np.ones(5) / 5, mode="same")  # smoothed over 0.2 s
    dx = half["x"] - bx[:, None]
    dy = half["y"] - by[:, None]
    crowd = np.sum(dx * dx + dy * dy <= params.crowd_radius_m ** 2, axis=1)
    dead = ~half["alive"] if "alive" in half else np.zeros(len(bx), dtype=bool)
    return np.exp(params.line_weight * near_line + params.box_weight * in_box
                  + params.slow_weight * (speed < params.slow_mps) + params.crowd_weight * crowd
                  + params.dead_weight * dead)


def gap_lengths(n, params, rng):
    """Frames each gap would last, one draw per frame, from the short/long mixture."""
    short = rng.random(n) < params.short_gap_share
    median = np.where(short, params.short_gap_median_s, params.long_gap_median_s) * FPS
    sigma = np.where(short, params.short_gap_sigma, params.long_gap_sigma)
    return np.maximum(1, np.round(median * np.exp(sigma * rng.standard_normal(n))))


def frames_until_restart(dead):
    """(frames,) frames from each frame to the next one in play: 0 where the ball is in play, and
    up to the half's end where play does not restart."""
    n = len(dead)
    index = np.arange(n)
    following = np.minimum.accumulate(np.where(dead, n, index)[::-1])[::-1]
    return following - index


def hide_ball(half, params, rng):
    """Boolean (frames,) mask of the frames on which the degraded tracking has no ball.

    A gap that opens while the ball is out of play lasts until the restart with chance
    until_restart; any other gap lasts a drawn length. The draws come in the same order as before
    until_restart existed, so at none the mask is the same.
    """
    n = len(half["ball_x"])
    lengths = gap_lengths(n, params, rng)
    if params.until_restart > 0 and "alive" in half:
        dead = ~np.asarray(half["alive"], dtype=bool)
        take = dead & (rng.random(n) < params.until_restart)
        lengths = np.where(take, np.maximum(frames_until_restart(dead), 1), lengths)
    return _gap_mask(ball_risk(half, params), params.ball_missing_share, lengths, rng,
                     base=_outside_mask(half, params, rng))


def degrade(half, params=DegradeParams(), seed=0):
    """A degraded copy of one half (dict of arrays as possession.external returns it).

    The ball and the players are hidden in gaps and jittered; 'ball_z' is dropped, because the
    match's tracking has no height. Labels ('alive', 'owner') are untouched: they are the truth.
    """
    rng = np.random.default_rng(seed)
    n = len(half["ball_x"])
    out = {key: value for key, value in half.items() if key != "ball_z"}

    hidden = hide_ball(half, params, rng)
    window = max(1, int(round(params.noise_window_s * FPS)))
    noise = _smooth_noise(rng, (n, 2), params.ball_noise_m, window)
    out["ball_x"] = np.where(hidden, np.nan, half["ball_x"] + noise[:, 0]).astype(np.float32)
    out["ball_y"] = np.where(hidden, np.nan, half["ball_y"] + noise[:, 1]).astype(np.float32)

    on_pitch = ~np.isnan(half["x"])
    on_pitch_count = on_pitch.sum(axis=1).mean() if on_pitch.size else 0.0
    share = max(0.0, 1.0 - params.players_tracked / max(on_pitch_count, 1e-9))
    columns = half["x"].shape[1]
    if share > 0:
        median = params.player_gap_median_s * FPS
        lengths = np.maximum(1, np.round(
            median * np.exp(params.player_gap_sigma * rng.standard_normal((n, columns)))))
        lost = _gap_mask(np.ones((n, columns)), share, lengths, rng)
    else:
        lost = np.zeros_like(on_pitch)
    jitter_x = _smooth_noise(rng, half["x"].shape, params.player_noise_m, window)
    jitter_y = _smooth_noise(rng, half["x"].shape, params.player_noise_m, window)
    out["x"] = np.where(lost, np.nan, half["x"] + jitter_x).astype(np.float32)
    out["y"] = np.where(lost, np.nan, half["y"] + jitter_y).astype(np.float32)
    return out
