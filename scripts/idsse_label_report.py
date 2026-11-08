"""How much the control rule changes IDSSE's possession labels, match by match.

Raw DFL labels against controlled_labels() at a few tolerances: possession count, duration and
changes per minute of live ball, beside the public window's figures for reference only -- the
public window chooses nothing here (AGENTS.md). Public data only; reads no match data.

    python scripts/idsse_label_report.py [MATCH_ID ...]
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from possession import external as ex  # noqa: E402

TOLERANCES_S = (0.5, 1.0, 2.0)
# The public window, for orientation only: 17 possessions over 6031 live frames, mean 14.2 s.
PUBLIC = {"per_live_min": 17 / (6031 / ex.FPS / 60), "mean_s": 14.2, "under_1s": 1 / 17}


def stats(labels):
    spells = np.array([stop - start for start, stop, label in ex.runs(labels) if label != "D"])
    live_min = np.sum(labels != "D") / ex.FPS / 60
    return {"n": len(spells), "per_live_min": len(spells) / live_min,
            "mean_s": spells.mean() / ex.FPS, "median_s": np.median(spells) / ex.FPS,
            "under_1s": np.mean(spells < ex.FPS)}


def line(name, s):
    return "  {:<16} {:>4} poss  {:4.1f}/live min  mean {:4.1f} s  median {:4.1f} s  <1 s {:4.0%}".format(
        name, s["n"], s["per_live_min"], s["mean_s"], s["median_s"], s["under_1s"])


def report(match_id):
    files = ex.paths(match_id)
    if not os.path.isfile(files["positions"]):
        print("{}: positions not downloaded, skipped".format(match_id))
        return None
    info = ex.read_info(files["info"])
    halves = ex.read_positions(files["positions"], info["home"], info["away"])
    times, teams, controlled = ex.read_events(files["events"], info["home"], info["away"])
    times, teams = times[controlled], teams[controlled]

    per_half = {}
    for section in ex.SECTIONS:
        half = halves[section]
        frames = np.round((times - half["t0"]) * ex.FPS).astype(int)
        inside = (frames >= 0) & (frames < len(half["alive"]))
        per_half[section] = (ex.raw_labels(half), frames[inside], teams[inside])

    offsets = np.concatenate([ex.flip_offsets(*per_half[s]) for s in ex.SECTIONS])
    shift = int(np.median(offsets))
    spread = np.percentile(np.abs(offsets - shift), [50, 80, 90]) / ex.FPS
    ball = np.concatenate([halves[s]["ball_x"] for s in ex.SECTIONS])
    print("\n{}  ({} controlled actions, offset median {:+.2f} s, |dev| p50/p80/p90 "
          "{:.2f}/{:.2f}/{:.2f} s, ball x range {:.1f}..{:.1f} m)".format(
              match_id, len(times), shift / ex.FPS, *spread, np.nanmin(ball), np.nanmax(ball)))

    raw = np.concatenate([per_half[s][0] for s in ex.SECTIONS])
    print(line("raw DFL", stats(raw)))
    results = {"raw": stats(raw)}
    for tolerance in TOLERANCES_S:
        aligned = np.concatenate([
            ex.controlled_labels(labels, frames - shift, half_teams, int(tolerance * ex.FPS))
            for labels, frames, half_teams in (per_half[s] for s in ex.SECTIONS)])
        results[tolerance] = stats(aligned)
        print(line("control ±{:.1f} s".format(tolerance), results[tolerance]))
    return results


def main(match_ids):
    print("public window (orientation only): {:.1f}/live min, mean {:.1f} s, <1 s {:.0%}".format(
        PUBLIC["per_live_min"], PUBLIC["mean_s"], PUBLIC["under_1s"]))
    for match_id in match_ids:
        report(match_id)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or sorted(ex.MATCHES)))
