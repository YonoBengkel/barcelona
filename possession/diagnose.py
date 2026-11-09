"""Where the pipeline goes wrong, in two views: by ball visibility, and possession by possession.

- By ball state: how often each label is called while the ball is seen, bridged over a short gap,
  or missing for up to 5 s, 5-15 s or longer, and, where labels exist, how often the call is right.
  The same table on the match, which has no labels, shows where the match's calls differ from the
  practice ground's, from aggregates alone (AGENTS.md).
- Possession by possession: every true possession the matching at IoU 0.5 misses is put down to
  one cause. It went to the wrong team; it was called dead; it was split into pieces; it was
  merged with a neighbour into one predicted possession; or one predicted possession covers it but
  with its edges too far off. Every predicted possession left unmatched is put down to one kind:
  inside a stoppage, given to the wrong team, or a piece of a true possession.

The matching is possession_metrics' own, so the counts agree with its numbers.
"""

import collections

import numpy as np

from possession.features import FPS, NAMES
from possession.practice import pm

STATES = ("seen", "bridged", "gap <=5 s", "gap 5-15 s", "gap >15 s")
CAUSES = ("wrong team", "called dead", "split", "merged", "edges off")
SPURIOUS = ("in a stoppage", "wrong team", "piece of one")
LENGTHS = ("<2 s", "2-5 s", "5-15 s", ">=15 s")
LENGTH_EDGES_S = (2.0, 5.0, 15.0)


def ball_states(features):
    """(frames,) index into STATES, from a feature matrix: possession.features.compute's, or
    possession.models.design's, which starts with the same columns."""
    seen = features[:, NAMES.index("ball_seen")] > 0.5
    bridged = features[:, NAMES.index("ball_bridged")] > 0.5
    gap_s = np.expm1(np.asarray(features[:, NAMES.index("gap_length")], dtype=np.float64))
    return np.select([seen, bridged, gap_s <= 5.0, gap_s <= 15.0], [0, 1, 2, 3], 4).astype(np.int8)


def player_speed(features, logged=False):
    """(frames,) mean speed of the tracked players over the last second, m/s: the speed_all
    feature of possession.features.compute, or of possession.models.design (logged=True), which
    holds it on a log scale. Players slow down when play stops, so its median per ball state shows
    whether the ball tends to go missing in play or in stoppages, without a label."""
    speed = np.asarray(features[:, NAMES.index("speed_all")], dtype=np.float64)
    return np.expm1(speed) if logged else speed


def _median(values):
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else float("nan")


def state_table(states, pred, gt=None, speed=None):
    """Rows per ball state, then 'all': share of frames, predicted H, A and D shares and, with
    labels, the true D share and the share of frames called right. With speed (player_speed), the
    median player speed, and with labels too, its median over the truly dead and the live frames."""
    groups = [(name, states == index) for index, name in enumerate(STATES)]
    rows = []
    for name, mask in groups + [("all", np.ones(len(states), dtype=bool))]:
        n = int(mask.sum())
        row = {"state": name, "frames": n / max(len(states), 1)}
        for label, code in (("H", pm.H), ("A", pm.A), ("D", pm.D)):
            row[label] = float(np.mean(pred[mask] == code)) if n else 0.0
        if gt is not None:
            row["true D"] = float(np.mean(gt[mask] == pm.D)) if n else 0.0
            row["right"] = float(np.mean(gt[mask] == pred[mask])) if n else 0.0
        if speed is not None:
            row["m/s"] = _median(speed[mask])
            if gt is not None:
                row["m/s D"] = _median(speed[mask & (gt == pm.D)])
                row["m/s live"] = _median(speed[mask & (gt != pm.D)])
        rows.append(row)
    return rows


SHARES = ("frames", "H", "A", "D", "true D", "right")
SPEEDS = ("m/s", "m/s D", "m/s live")


def format_state_table(rows):
    """Printable lines of state_table()."""
    columns = [key for key in SHARES + SPEEDS if key in rows[0]]
    lines = ["{:<11}".format("ball") + "".join("{:>9}".format(c) for c in columns)]
    for row in rows:
        lines.append("{:<11}".format(row["state"]) + "".join(
            "{:>9.1%}".format(row[c]) if c in SHARES else "{:>9.2f}".format(row[c])
            for c in columns))
    return lines


def format_confusion(gt, pred):
    """Printable lines: true H, A, D against called H, A, D, as shares of the frames."""
    total = max(len(gt), 1)
    lines = ["{:<8}".format("") + "".join("{:>10}".format("called " + s) for s in "HAD")]
    for label, code in (("H", pm.H), ("A", pm.A), ("D", pm.D)):
        lines.append("{:<8}".format("true " + label) + "".join(
            "{:>10.1%}".format(np.sum((gt == code) & (pred == other)) / total)
            for other in (pm.H, pm.A, pm.D)))
    return lines


def matched_pairs(overlaps, threshold):
    """[(true index, predicted index)], paired exactly as possession_metrics.matched_gt_flags."""
    pairs, true_used, pred_used = [], set(), set()
    for iou, i, j in overlaps:
        if iou < threshold:
            break
        if i in true_used or j in pred_used:
            continue
        true_used.add(i)
        pred_used.add(j)
        pairs.append((i, j))
    return pairs


def _owner(spells, n):
    """(n,) index of the spell each frame belongs to, -1 outside every spell."""
    out = np.full(n, -1, dtype=np.int64)
    for index, (start, stop, _) in enumerate(spells):
        out[start:stop] = index
    return out


def _dominant(frames, code):
    """'same', 'other' or 'dead', whichever the frames hold most of; ties go to 'same'."""
    other = pm.A if code == pm.H else pm.H
    counts = {"same": np.sum(frames == code), "other": np.sum(frames == other),
              "dead": np.sum(frames == pm.D)}
    return max(("same", "other", "dead"), key=lambda kind: counts[kind])


def spell_outcomes(gt, pred, threshold=pm.SPELL_IOU_HEADLINE):
    """How each true and each predicted possession of one stretch fared at one IoU threshold."""
    runs = pm.valid_runs(gt)
    pred = pm.fill_abstentions(pred, runs)
    true_spells = pm.possession_spells(gt, runs)
    pred_spells = pm.possession_spells(pred, runs)
    pairs = matched_pairs(pm.spell_overlaps(true_spells, pred_spells), threshold)
    true_hit, pred_hit = {i for i, _ in pairs}, {j for _, j in pairs}
    pred_of, true_of = _owner(pred_spells, len(gt)), _owner(true_spells, len(gt))

    missed, spurious = collections.Counter(), collections.Counter()
    by_length = {name: [0, 0] for name in LENGTHS}
    for i, (start, stop, code) in enumerate(true_spells):
        length = LENGTHS[int(np.searchsorted(LENGTH_EDGES_S, (stop - start) / FPS, side="right"))]
        by_length[length][1] += 1
        if i in true_hit:
            by_length[length][0] += 1
            continue
        kind = _dominant(pred[start:stop], code)
        if kind == "other":
            missed["wrong team"] += 1
        elif kind == "dead":
            missed["called dead"] += 1
        else:
            pieces = {int(j) for j in np.unique(pred_of[start:stop])
                      if j >= 0 and pred_spells[j][2] == code}
            if len(pieces) > 1:
                missed["split"] += 1
            else:
                first, last, _ = pred_spells[pieces.pop()]
                neighbours = {int(k) for k in np.unique(true_of[first:last])
                              if k >= 0 and k != i and true_spells[k][2] == code}
                missed["merged" if neighbours else "edges off"] += 1
    for j, (start, stop, code) in enumerate(pred_spells):
        if j not in pred_hit:
            kind = _dominant(gt[start:stop], code)
            spurious[{"dead": "in a stoppage", "other": "wrong team",
                      "same": "piece of one"}[kind]] += 1
    return {"true": len(true_spells), "predicted": len(pred_spells), "matched": len(pairs),
            "missed": missed, "spurious": spurious, "by_length": by_length}


def pool(outcomes):
    """Several spell_outcomes() taken together."""
    out = {"true": 0, "predicted": 0, "matched": 0, "missed": collections.Counter(),
           "spurious": collections.Counter(), "by_length": {name: [0, 0] for name in LENGTHS}}
    for part in outcomes:
        for key in ("true", "predicted", "matched"):
            out[key] += part[key]
        out["missed"].update(part["missed"])
        out["spurious"].update(part["spurious"])
        for name in LENGTHS:
            out["by_length"][name][0] += part["by_length"][name][0]
            out["by_length"][name][1] += part["by_length"][name][1]
    return out


def format_outcomes(outcome, threshold=pm.SPELL_IOU_HEADLINE):
    """Printable lines of a spell_outcomes() or pool() result."""
    missed = outcome["true"] - outcome["matched"]
    unmatched = outcome["predicted"] - outcome["matched"]
    return [
        "possessions at IoU {:g}: {} true, {} matched ({:.1%}); {} predicted, {} unmatched".format(
            threshold, outcome["true"], outcome["matched"],
            outcome["matched"] / max(outcome["true"], 1), outcome["predicted"], unmatched),
        "  missed, by cause: " + ", ".join(
            "{} {} ({:.0%})".format(cause, outcome["missed"][cause],
                                    outcome["missed"][cause] / max(missed, 1))
            for cause in CAUSES),
        "  matched, by true length: " + ", ".join(
            "{} {:.0%} of {}".format(name, hit / max(total, 1), total)
            for name, (hit, total) in outcome["by_length"].items()),
        "  unmatched predictions, by kind: " + ", ".join(
            "{} {} ({:.0%})".format(kind, outcome["spurious"][kind],
                                    outcome["spurious"][kind] / max(unmatched, 1))
            for kind in SPURIOUS),
    ]
