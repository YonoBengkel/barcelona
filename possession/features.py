"""Per-frame features from positions alone, identical on IDSSE and on the match's tracking.

The match's tracking carries positions and nothing else to rely on: no ball height, no ball state,
no identities, and a speed channel whose unit the documents disagree on and which is missing on
89.5% of the ball's frames (scripts/data_properties.py). So every feature here is built from x and
y, team by team, and means the same on a degraded IDSSE half as on a half of the match. That is
what lets a model chosen on the first be applied to the second.

Two families, for the two halves of the label:

- whether the ball is in play: how long the ball has been missing and will stay missing, where it
  was last seen, how fast the players move (they slow down when play stops), whether both teams
  stand in their own halves (a kickoff), how many players crowd the penalty boxes;
- which team has it: the nearest player of each team and the gap between them, how many of each
  team are around the ball, where the ball is and where it is heading along the home side's
  attacking direction, and how each team is shaped and moving (the side in possession spreads out
  and advances; the other compacts and retreats).

A short ball gap is bridged by linear interpolation, so the possession features survive a flicker.
A long one is left unknown and flagged as such: guessing where the ball was across ten seconds
would invent information the decoder should not trust.
"""

import numpy as np

FPS = 25
HOME, AWAY = 0, 1
BRIDGE_MAX_S = 1.0
SMOOTH_S = 1.0
NEAR_LINE_M = 2.0
BOX_DEPTH_M, BOX_HALF_WIDTH_M = 16.5, 20.16
CROWD_RADII_M = (2.0, 5.0)

NAMES = (
    "ball_seen", "ball_bridged", "gap_since", "gap_until", "gap_length",
    "ball_near_line", "ball_in_box", "ball_outside", "ball_speed",
    "ball_progress_home", "ball_velocity_home",
    "near_home", "near_away", "near_diff",
    "crowd2_home", "crowd2_away", "crowd5_home", "crowd5_away",
    "speed_all", "speed_home", "speed_away", "forward_home", "forward_away",
    "advance_home", "advance_away", "width_home", "width_away", "length_home", "length_away",
    "tracked_home", "tracked_away", "own_half_share", "players_in_boxes",
)


def rolling_mean(values, window):
    """Centred mean over 2 * window + 1 frames, ignoring NaN; NaN where a window holds nothing."""
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    total = np.concatenate(([0.0], np.cumsum(np.where(valid, values, 0.0))))
    count = np.concatenate(([0], np.cumsum(valid)))
    index = np.arange(len(values))
    low, high = np.maximum(0, index - window), np.minimum(len(values), index + window + 1)
    n = count[high] - count[low]
    return np.where(n > 0, (total[high] - total[low]) / np.maximum(n, 1), np.nan)


def row_mean(values):
    """Mean over columns ignoring NaN, and the number of finite values per row."""
    valid = np.isfinite(values)
    count = valid.sum(axis=1)
    total = np.where(valid, values, 0.0).sum(axis=1)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan), count


def row_std(values):
    mean, count = row_mean(values)
    square, _ = row_mean(values * values)
    return np.where(count > 1, np.sqrt(np.maximum(square - mean * mean, 0.0)), np.nan)


def gap_frames(visible):
    """(since, until, length, last, next) per frame for the ball's gaps, in frames.

    since: frames since the ball was last seen (0 where it is seen); until: frames until it is
    seen again; length: the length of the gap the frame sits in; last and next: the frames it was
    last and will next be seen on (-1 and n where there is none).
    """
    n = len(visible)
    index = np.arange(n)
    last = np.maximum.accumulate(np.where(visible, index, -1))
    following = np.minimum.accumulate(np.where(visible, index, n)[::-1])[::-1]
    since = np.where(visible, 0, index - last)
    until = np.where(visible, 0, following - index)
    length = np.where(visible, 0, following - last - 1)
    return since, until, length, last, following


def bridged_ball(half, bridge_max=int(BRIDGE_MAX_S * FPS)):
    """(x, y, seen, bridged): the ball with gaps up to bridge_max frames linearly interpolated."""
    bx, by = half["ball_x"].astype(np.float64), half["ball_y"].astype(np.float64)
    seen = ~np.isnan(bx)
    x, y = bx.copy(), by.copy()
    _, _, length, last, following = gap_frames(seen)
    bridged = ~seen & (length <= bridge_max) & (last >= 0) & (following < len(bx))
    if seen.sum() >= 2 and bridged.any():
        visible = np.flatnonzero(seen)
        x[bridged] = np.interp(np.flatnonzero(bridged), visible, bx[visible])
        y[bridged] = np.interp(np.flatnonzero(bridged), visible, by[visible])
    return x, y, seen, bridged


def _deepest_side(x):
    """Median sign of the x of the player farthest from the centre line, frame by frame."""
    if x.shape[1] == 0:
        return 0.0
    distance = np.where(np.isnan(x), -1.0, np.abs(x))
    deepest = x[np.arange(len(x)), np.argmax(distance, axis=1)]
    deepest = deepest[np.isfinite(deepest)]
    return float(np.sign(np.median(deepest))) if len(deepest) else 0.0


def attack_directions(half):
    """(home, away), each +1 or -1: the x direction each team attacks in this half.

    A team's own goal is on the side of its deepest player, frame by frame (nearly always its
    goalkeeper), taken as the median over the half, so a centre-back pushed up or a keeper the
    tracking misses for a while does not flip it. If both teams come out on the same side, the
    team whose players sit lower on average defends the negative goal.
    """
    sides = [_deepest_side(half["x"][:, half["team"] == team]) for team in (HOME, AWAY)]
    if sides[0] == 0 or sides[0] == sides[1]:
        means = [row_mean(half["x"][:, half["team"] == team])[0] for team in (HOME, AWAY)]
        lower = np.nanmean(means[0]) < np.nanmean(means[1])
        sides = [-1.0, 1.0] if lower else [1.0, -1.0]
    return np.array([-sides[0], -sides[1]])


def compute(half, pitch):
    """(features, NAMES): a (frames, len(NAMES)) float32 matrix, NaN where a feature is unknown."""
    length, width = pitch
    n = len(half["ball_x"])
    window = int(SMOOTH_S * FPS) // 2
    directions = attack_directions(half)
    team = half["team"]
    x, y = half["x"].astype(np.float64), half["y"].astype(np.float64)

    bx, by, seen, bridged = bridged_ball(half)
    known = np.isfinite(bx)
    since, until, gap, last, _ = gap_frames(seen)

    # Where the ball is, or where it was last seen while it is missing.
    anchor = np.where(known, np.arange(n), np.maximum(last, 0))
    ax, ay = bx[anchor], by[anchor]
    anchored = known | (last >= 0)
    near_line = (np.abs(ax) > length / 2 - NEAR_LINE_M) | (np.abs(ay) > width / 2 - NEAR_LINE_M)
    in_box = (np.abs(ax) > length / 2 - BOX_DEPTH_M) & (np.abs(ax) <= length / 2) \
        & (np.abs(ay) < BOX_HALF_WIDTH_M)
    outside = (np.abs(ax) > length / 2) | (np.abs(ay) > width / 2)

    step = np.hypot(np.diff(bx, prepend=np.nan), np.diff(by, prepend=np.nan)) * FPS
    velocity = np.diff(bx, prepend=np.nan) * FPS * directions[HOME]

    dx, dy = x - bx[:, None], y - by[:, None]
    squared = dx * dx + dy * dy
    columns = {HOME: team == HOME, AWAY: team == AWAY}
    near, crowd = {}, {}
    for side, mask in columns.items():
        part = np.where(np.isnan(squared[:, mask]), np.inf, squared[:, mask])
        nearest = np.sqrt(part.min(axis=1)) if part.shape[1] else np.full(n, np.inf)
        near[side] = np.where(np.isfinite(nearest) & known, nearest, np.nan)
        for radius in CROWD_RADII_M:
            count = np.sum(part <= radius * radius, axis=1).astype(np.float64)
            crowd[(side, radius)] = np.where(known, count, np.nan)

    vx = np.diff(x, axis=0, prepend=np.full((1, x.shape[1]), np.nan)) * FPS
    vy = np.diff(y, axis=0, prepend=np.full((1, x.shape[1]), np.nan)) * FPS
    speed = np.hypot(vx, vy)
    per_team = {}
    for side, mask in columns.items():
        along = x[:, mask] * directions[side]
        per_team[side] = {
            "speed": rolling_mean(row_mean(speed[:, mask])[0], window),
            "forward": rolling_mean(row_mean(vx[:, mask] * directions[side])[0], window),
            "advance": row_mean(along)[0],
            "width": row_std(y[:, mask]),
            "length": row_std(x[:, mask]),
            "tracked": np.isfinite(x[:, mask]).sum(axis=1).astype(np.float64),
            "own_half": row_mean(np.where(np.isfinite(along), (along < 0).astype(float),
                                          np.nan))[0],
        }
    in_boxes = (np.abs(x) > length / 2 - BOX_DEPTH_M) & (np.abs(y) < BOX_HALF_WIDTH_M)

    columns_out = [
        seen.astype(float), bridged.astype(float),
        np.log1p(since / FPS), np.log1p(until / FPS), np.log1p(gap / FPS),
        np.where(anchored, near_line, np.nan), np.where(anchored, in_box, np.nan),
        np.where(anchored, outside, np.nan),
        rolling_mean(step, window),
        np.where(known, bx * directions[HOME] / (length / 2), np.nan),
        rolling_mean(velocity, window),
        near[HOME], near[AWAY], near[HOME] - near[AWAY],
        crowd[(HOME, 2.0)], crowd[(AWAY, 2.0)], crowd[(HOME, 5.0)], crowd[(AWAY, 5.0)],
        rolling_mean(row_mean(speed)[0], window),
        per_team[HOME]["speed"], per_team[AWAY]["speed"],
        per_team[HOME]["forward"], per_team[AWAY]["forward"],
        per_team[HOME]["advance"], per_team[AWAY]["advance"],
        per_team[HOME]["width"], per_team[AWAY]["width"],
        per_team[HOME]["length"], per_team[AWAY]["length"],
        per_team[HOME]["tracked"], per_team[AWAY]["tracked"],
        np.fmin(per_team[HOME]["own_half"], per_team[AWAY]["own_half"]),
        in_boxes.sum(axis=1).astype(np.float64),
    ]
    return np.stack(columns_out, axis=1).astype(np.float32), NAMES


def swapped(half):
    """The same half with home and away exchanged, for training a model that treats both alike."""
    out = dict(half)
    out["team"] = (1 - half["team"]).astype(half["team"].dtype)
    if "owner" in half:
        out["owner"] = (1 - half["owner"]).astype(half["owner"].dtype)
    return out
