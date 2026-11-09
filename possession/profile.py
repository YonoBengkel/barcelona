"""Aggregate statistics of a tracking signal: how the ball and the players go missing, and how
fast the players seem to move.

The same profile runs on the match's automatic tracking (scripts/atd_profile.py, which prints only
these aggregates) and on the degraded practice matches (possession.simulate), so the simulator can
be calibrated against the first and checked against it afterwards. Every number is a statistic
over whole halves; nothing here is a label or a frame.

The player speeds are measured exactly as possession.features measures speed_all (frame-to-frame
steps, averaged over a centred second), because the models read that feature. Position noise
inflates it most where players stand still, so its low percentiles show how noisy a tracking is
(M11g).
"""

import numpy as np

from possession.features import SMOOTH_S, rolling_mean, row_mean

FPS = 25
PERCENTILES = (10, 25, 50, 75, 90, 95, 99)
# Gaps shorter than this are flicker, not occlusions; context is measured on the longer ones.
CONTEXT_MIN_GAP = 5
CROWD_RADII_M = (2.0, 5.0)


def run_lengths(flags):
    """(lengths of True runs, lengths of False runs) of a boolean array."""
    if not len(flags):
        return np.array([], dtype=int), np.array([], dtype=int)
    breaks = np.flatnonzero(flags[1:] != flags[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [len(flags)]))
    lengths = np.diff(bounds)
    values = flags[bounds[:-1]]
    return lengths[values], lengths[~values]


def gap_starts(present, min_gap=CONTEXT_MIN_GAP):
    """Indices of the last visible frame before each gap of at least min_gap frames."""
    breaks = np.flatnonzero(present[1:] != present[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [len(present)]))
    out = []
    for i in range(len(bounds) - 1):
        start, stop = bounds[i], bounds[i + 1]
        if not present[start] and stop - start >= min_gap and start > 0:
            out.append(start - 1)
    return np.array(out, dtype=int)


def describe(values, scale=1.0):
    """{'n', 'mean', 'p10', ...} of an array, scaled (e.g. frames to seconds)."""
    values = np.asarray(values, dtype=float) * scale
    if not len(values):
        return {"n": 0}
    out = {"n": int(len(values)), "mean": float(values.mean())}
    out.update({"p{}".format(p): float(v) for p, v in zip(PERCENTILES,
                                                          np.percentile(values, PERCENTILES))})
    return out


def crowding(half, frames, radius):
    """Mean number of tracked players within radius metres of the ball on the given frames."""
    if not len(frames):
        return float("nan")
    dx = half["x"][frames] - half["ball_x"][frames, None]
    dy = half["y"][frames] - half["ball_y"][frames, None]
    return float(np.nanmean(np.sum(dx * dx + dy * dy <= radius * radius, axis=1)))


def ball_speed(half):
    """m/s between consecutive frames where the ball is seen on both; NaN elsewhere."""
    speed = np.full(len(half["ball_x"]), np.nan)
    step = np.hypot(np.diff(half["ball_x"]), np.diff(half["ball_y"])) * FPS
    speed[1:] = step
    return speed


def player_speeds(half):
    """(mean over the tracked players, each player's) speed over a centred second, m/s, per frame:
    the first is possession.features' speed_all, the second the same per player slot, (frames,
    slots). NaN where nothing was tracked in the second."""
    window = int(SMOOTH_S * FPS) // 2
    x, y = half["x"].astype(np.float64), half["y"].astype(np.float64)
    first = np.full((1, x.shape[1]), np.nan)
    step = np.hypot(np.diff(x, axis=0, prepend=first), np.diff(y, axis=0, prepend=first)) * FPS
    each = np.column_stack([rolling_mean(step[:, j], window) for j in range(step.shape[1])]) \
        if step.shape[1] else np.empty((len(step), 0))
    return rolling_mean(row_mean(step)[0], window), each


def profile(halves, pitch):
    """Aggregate profile over a list of halves (dicts with ball_x, ball_y, x, y, team)."""
    length, width = pitch
    seen_runs, gap_runs, objects, home, away, segments = [], [], [], [], [], []
    starts_speed, all_speed, near = [], [], {r: ([], []) for r in CROWD_RADII_M}
    players_mean, players_each, players_seen = [], [], []
    zones = {"outer_third": [0, 0], "penalty_box": [0, 0], "near_line": [0, 0]}
    outside, frames_seen, frames_total = 0, 0, 0

    for half in halves:
        present = ~np.isnan(half["ball_x"])
        seen, gaps = run_lengths(present)
        seen_runs.append(seen)
        gap_runs.append(gaps)
        frames_seen += int(present.sum())
        frames_total += len(present)

        tracked = ~np.isnan(half["x"])
        objects.append(tracked.sum(axis=1))
        home.append(tracked[:, half["team"] == 0].sum(axis=1))
        away.append(tracked[:, half["team"] == 1].sum(axis=1))
        for slot in range(tracked.shape[1]):
            segments.append(run_lengths(tracked[:, slot])[0])

        mean, each = player_speeds(half)
        players_mean.append(mean)
        players_each.append(each.ravel())
        players_seen.append(mean[present])

        speed = ball_speed(half)
        starts = gap_starts(present)
        visible = np.flatnonzero(present)
        starts_speed.append(speed[starts])
        all_speed.append(speed[visible])
        for radius in CROWD_RADII_M:
            near[radius][0].append(crowding(half, starts, radius))
            near[radius][1].append(crowding(half, visible, radius))

        bx, by = half["ball_x"], half["ball_y"]
        masks = {"outer_third": np.abs(bx) > length / 6,
                 "penalty_box": (np.abs(bx) > length / 2 - 16.5) & (np.abs(by) < 20.16),
                 "near_line": (np.abs(bx) > length / 2 - 2) | (np.abs(by) > width / 2 - 2)}
        for name, mask in masks.items():
            zones[name][0] += int(mask[starts].sum())
            zones[name][1] += int(mask[visible].sum())
        outside += int(((np.abs(bx) > length / 2) | (np.abs(by) > width / 2))[visible].sum())

    n_starts = sum(len(s) for s in starts_speed)
    return {
        "ball_seen_share": frames_seen / frames_total if frames_total else float("nan"),
        "seen_run_s": describe(np.concatenate(seen_runs), 1 / FPS),
        "gap_run_s": describe(np.concatenate(gap_runs), 1 / FPS),
        "gap_frames_in_gaps_of_at_least": {
            "1s": _share_in_long(gap_runs, FPS), "5s": _share_in_long(gap_runs, 5 * FPS),
            "10s": _share_in_long(gap_runs, 10 * FPS)},
        "objects_per_frame": describe(np.concatenate(objects)),
        "objects_over_22_share": float(np.mean(np.concatenate(objects) > 22)),
        "home_per_frame": float(np.mean(np.concatenate(home))),
        "away_per_frame": float(np.mean(np.concatenate(away))),
        "slot_segment_s": describe(np.concatenate(segments), 1 / FPS),
        "gap_contexts": n_starts,
        "ball_speed_before_gap": describe(_finite(starts_speed)),
        "ball_speed_when_seen": describe(_finite(all_speed)),
        "crowding_before_gap": {r: _nanmean(near[r][0]) for r in CROWD_RADII_M},
        "crowding_when_seen": {r: _nanmean(near[r][1]) for r in CROWD_RADII_M},
        "zone_share_before_gap": {k: v[0] / n_starts if n_starts else float("nan")
                                  for k, v in zones.items()},
        "zone_share_when_seen": {k: v[1] / frames_seen if frames_seen else float("nan")
                                 for k, v in zones.items()},
        "ball_outside_pitch_when_seen": outside / frames_seen if frames_seen else float("nan"),
        "player_speed": describe(_finite(players_mean)),
        "player_speed_ball_seen": describe(_finite(players_seen)),
        "player_speed_each": describe(_finite(players_each)),
    }


def _share_in_long(gap_runs, threshold):
    gaps = np.concatenate(gap_runs)
    return float(gaps[gaps >= threshold].sum() / gaps.sum()) if gaps.sum() else float("nan")


def _nanmean(values):
    """Mean ignoring NaN, and NaN (not a warning) when there is nothing to average."""
    values = np.asarray(values, dtype=float)
    return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")


def _finite(parts):
    values = np.concatenate(parts)
    return values[np.isfinite(values)]


def format_profile(result):
    """Human-readable lines, aggregates only."""
    def dist(d, unit):
        if not d.get("n"):
            return "n 0"
        return "n {}  mean {:.2f}{u}  p10 {:.2f}  p25 {:.2f}  p50 {:.2f}  p75 {:.2f}  " \
               "p90 {:.2f}  p95 {:.2f}  p99 {:.2f}{u}".format(
                   d["n"], d["mean"], d["p10"], d["p25"], d["p50"], d["p75"], d["p90"],
                   d["p95"], d["p99"], u=unit)
    r = result
    lines = [
        "ball seen on {:.1%} of frames".format(r["ball_seen_share"]),
        "seen runs:  " + dist(r["seen_run_s"], " s"),
        "gap runs:   " + dist(r["gap_run_s"], " s"),
        "share of missing frames inside gaps >=1 s {:.1%}, >=5 s {:.1%}, >=10 s {:.1%}".format(
            *(r["gap_frames_in_gaps_of_at_least"][k] for k in ("1s", "5s", "10s"))),
        "tracked objects per frame: " + dist(r["objects_per_frame"], ""),
        "frames with more than 22 objects {:.2%}; home {:.2f}, away {:.2f} per frame".format(
            r["objects_over_22_share"], r["home_per_frame"], r["away_per_frame"]),
        "slot segments: " + dist(r["slot_segment_s"], " s"),
        "gaps of >=0.2 s measured for context: {}".format(r["gap_contexts"]),
        "ball speed before a gap: " + dist(r["ball_speed_before_gap"], " m/s"),
        "ball speed when seen:    " + dist(r["ball_speed_when_seen"], " m/s"),
        "players within 2 m / 5 m of the ball: before a gap {:.2f} / {:.2f}, when seen "
        "{:.2f} / {:.2f}".format(r["crowding_before_gap"][2.0], r["crowding_before_gap"][5.0],
                                 r["crowding_when_seen"][2.0], r["crowding_when_seen"][5.0]),
        "zones before a gap vs when seen: outer thirds {:.1%} vs {:.1%}, penalty boxes {:.1%} vs "
        "{:.1%}, within 2 m of a line {:.1%} vs {:.1%}".format(
            r["zone_share_before_gap"]["outer_third"], r["zone_share_when_seen"]["outer_third"],
            r["zone_share_before_gap"]["penalty_box"], r["zone_share_when_seen"]["penalty_box"],
            r["zone_share_before_gap"]["near_line"], r["zone_share_when_seen"]["near_line"]),
        "ball seen outside the pitch lines on {:.2%} of seen frames".format(
            r["ball_outside_pitch_when_seen"]),
        "players' mean speed over 1 s:           " + dist(r["player_speed"], " m/s"),
        "  the same while the ball is seen:      " + dist(r["player_speed_ball_seen"], " m/s"),
        "each player's speed over 1 s:           " + dist(r["player_speed_each"], " m/s"),
    ]
    return lines
