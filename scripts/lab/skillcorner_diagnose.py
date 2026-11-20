"""Lab: where the submitted pipeline and SkillCorner's labels part ways, over the 20 matches.

Diagnostics for the first reading of scripts/lab/cross_provider.py (tracking as given, pipeline as
submitted), pooled over SkillCorner's 20 matches:

- the confusion of frame labels, on every frame and on tracked frames only;
- what each side says on the frames that carry no tracking at all (replays, close-ups);
- the team call where both say the ball is alive;
- the pooled spellF1 with SkillCorner's labels moved earlier or later, which shows whether the two
  place the changes of possession at different times.

Public data only (SkillCorner, MIT). CPU only; a job for Kaggle.

    python scripts/lab/skillcorner_diagnose.py [--workers N]
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cross_provider as cp  # noqa: E402  (puts scripts/ and the repository on the path)
import train_tcn_ensemble as te  # noqa: E402

from possession import practice  # noqa: E402
from possession import skillcorner as sc  # noqa: E402

pm = practice.pm
SHIFTS_S = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
READING = "as given"


def shifted(codes, frames):
    """codes moved later by frames (earlier when negative), X where nothing moved in."""
    out = np.full_like(codes, pm.X)
    if frames >= 0:
        out[frames:] = codes[:len(codes) - frames]
    else:
        out[:frames] = codes[-frames:]
    return out


def truth_shifted(halves, frames):
    return practice.joined([shifted(practice.codes(h["labels"]), frames) for h in halves],
                           gap=pm.X)


def tracked_flags(halves):
    """Per frame of the joined match, whether it carries tracking (False on the gap)."""
    return practice.joined([np.asarray(h["tracked"], dtype=bool) for h in halves], gap=False)


def confusion(gt, pred, mask):
    """3 x 3 counts, truth H, A, D by row and the prediction by column, over frames in mask."""
    return np.array([[np.sum(mask & (gt == t) & (pred == p)) for p in range(3)]
                     for t in range(3)])


def shares(table):
    return table / np.maximum(table.sum(axis=1, keepdims=True), 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()
    start = time.time()
    matches = list(sc.MATCHES)
    fetched = sc.fetch(matches)
    print("SkillCorner open data at {}: {} files fetched ({:.0f} s)".format(
        sc.COMMIT[:12], fetched, time.time() - start))
    pool = te.worker_pool(args.workers)
    try:
        parts = dict((pool.map if pool else map)(cp.match_parts, matches))
    finally:
        if pool is not None:
            pool.shutdown()
    print("parsed and weighed {} matches ({:.0f} s so far)".format(len(parts),
                                                                  time.time() - start))

    every = np.zeros((3, 3), dtype=np.int64)
    tracked_only = np.zeros((3, 3), dtype=np.int64)
    untracked = {"frames": 0, "truth D": 0, "ours D": 0}
    team = {"frames": 0, "right": 0}
    per_match = []
    for m in matches:
        halves = parts[m][READING]
        gt = cp.truth(halves, "every frame")
        pred = cp.predicted(halves, "submitted")
        flags = tracked_flags(halves)
        graded = gt != pm.X
        every += confusion(gt, pred, graded)
        tracked_only += confusion(gt, pred, graded & flags)
        gap = graded & ~flags
        untracked["frames"] += int(gap.sum())
        untracked["truth D"] += int(np.sum(gap & (gt == pm.D)))
        untracked["ours D"] += int(np.sum(gap & (pred == pm.D)))
        live = graded & flags & (gt != pm.D) & (pred != pm.D)
        team["frames"] += int(live.sum())
        team["right"] += int(np.sum(live & (gt == pred)))
        per_match.append((m, float(np.mean(~flags[graded])),
                          float(np.mean(gt[live] == pred[live])) if live.any() else float("nan")))

    print("\n===== the submitted pipeline against SkillCorner's labels, 20 matches pooled =====")
    for name, table in (("every frame", every), ("tracked frames", tracked_only)):
        print("confusion on {} (rows: truth H, A, D; columns: ours H, A, D; share of the row):"
              .format(name))
        for label, row, count in zip("HAD", shares(table), table.sum(axis=1)):
            print("  {}  {}   ({} frames)".format(label, "  ".join("{:.3f}".format(v)
                                                                  for v in row), count))
    print("frames without tracking: {} ({:.1%} of graded frames); SkillCorner says D on {:.1%} of "
          "them, ours on {:.1%}".format(
              untracked["frames"], untracked["frames"] / every.sum(),
              untracked["truth D"] / max(untracked["frames"], 1),
              untracked["ours D"] / max(untracked["frames"], 1)))
    print("team call on tracked frames where both say the ball is alive: {:.1%} right over {} "
          "frames".format(team["right"] / max(team["frames"], 1), team["frames"]))
    print("per match, share of frames without tracking / team call right: " + ", ".join(
        "{} {:.0%}/{:.1%}".format(m, gap, right) for m, gap, right in per_match))

    print("pooled spellF1 with SkillCorner's labels moved later by s seconds (earlier when "
          "negative), every frame:")
    for seconds in SHIFTS_S:
        frames = int(round(seconds * sc.FPS))
        counts = [practice.counts(truth_shifted(parts[m][READING], frames),
                                  cp.predicted(parts[m][READING], "submitted"))
                  for m in matches]
        base = [practice.counts(truth_shifted(parts[m][READING], frames),
                                cp.predicted(parts[m][READING], "baseline"))
                for m in matches]
        print("  {:+.1f} s: ours {:.4f}, baseline_smooth50 {:.4f}".format(
            seconds, practice.pooled(counts)[0], practice.pooled(base)[0]))
    print("\ndone in {:.0f} s".format(time.time() - start))
    return 0


if __name__ == "__main__":
    sys.exit(main())
