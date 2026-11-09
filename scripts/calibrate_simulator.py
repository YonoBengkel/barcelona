"""Fit the simulator's ball parameters so degraded IDSSE profiles like the match's tracking.

TARGET holds the aggregates scripts/atd_profile.py printed for the match on 2026-11-08: no label,
no frame, only statistics of when and how long its tracking loses the ball, which is unsupervised
use of the match's own tracking. A seeded search over DegradeParams -- random samples, then
local perturbations around the best with a shrinking scale -- scores each candidate by how far the
degraded IDSSE profile lands from TARGET, and prints the best one, to be copied into
DegradeParams' defaults. Public data only; reads no match data.

--players fits the players' position noise instead (M11g). The ball search above runs with that
noise off, so it was never set against the match: 0.15 m drifting over a second reads as about
1.3 m/s for a player who stands still. Each level of NOISE_GRID degrades every half of the matches
named (all seven by default), and the players' speed percentiles (possession.profile, measured as
the models' speed_all) are set against TARGET_PLAYERS by squared log-ratio; the level with the
lowest loss is printed, to be copied into DegradeParams.player_noise_m.

    python scripts/calibrate_simulator.py [--samples 80] [--local 120] [MATCH_ID ...]
    python scripts/calibrate_simulator.py --players [MATCH_ID ...]
"""

import argparse
import os
import sys
from dataclasses import replace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from possession import external as ex  # noqa: E402
from possession.profile import profile  # noqa: E402
from possession.simulate import DegradeParams, degrade  # noqa: E402

TARGET = {
    "gap_s": {"p10": 0.47, "p25": 6.09, "p50": 10.72, "p75": 21.06, "p90": 33.81},
    "seen_s": {"p10": 2.18, "p50": 13.80, "p90": 61.68},
    "speed_before_gap_p50": 1.51, "speed_seen_p50": 5.08,
    "near_line_before": 0.307, "near_line_seen": 0.091,
    "box_before": 0.314, "box_seen": 0.136,
    "crowd2_before": 0.77, "crowd2_seen": 0.58,
    "outside_seen": 0.0419,
}
SEARCH = {  # parameter: (low, high)
    "line_weight": (0.0, 3.0), "box_weight": (0.0, 3.0), "slow_weight": (0.0, 3.0),
    "crowd_weight": (0.0, 0.8), "dead_weight": (0.0, 4.0), "outside_visible": (0.0, 0.5),
    "short_gap_share": (0.05, 0.3), "long_gap_median_s": (8.0, 18.0),
    "long_gap_sigma": (0.6, 1.2),
}
# The players' speed percentiles of the match's tracking, in m/s: {statistic: {percentile: value}}
# for 'player_speed' and 'player_speed_each' of possession.profile. None until
# scripts/atd_profile.py has printed them; --players then prints the degraded side alone.
TARGET_PLAYERS = None
PLAYER_STATISTICS = ("player_speed", "player_speed_each")
PLAYER_PERCENTILES = ("p10", "p25", "p50", "p75", "p90")
NOISE_GRID = (0.0, 0.025, 0.05, 0.075, 0.1, 0.15)


def summary(halves, params):
    degraded = [degrade(h, replace(params, players_tracked=99.0, player_noise_m=0.0), seed=i)
                for i, h in enumerate(halves)]
    p = profile(degraded, params.pitch)
    return {
        "gap_s": {k: p["gap_run_s"].get(k, np.nan) for k in TARGET["gap_s"]},
        "seen_s": {k: p["seen_run_s"].get(k, np.nan) for k in TARGET["seen_s"]},
        "speed_before_gap_p50": p["ball_speed_before_gap"].get("p50", np.nan),
        "speed_seen_p50": p["ball_speed_when_seen"].get("p50", np.nan),
        "near_line_before": p["zone_share_before_gap"]["near_line"],
        "near_line_seen": p["zone_share_when_seen"]["near_line"],
        "box_before": p["zone_share_before_gap"]["penalty_box"],
        "box_seen": p["zone_share_when_seen"]["penalty_box"],
        "crowd2_before": p["crowding_before_gap"][2.0],
        "crowd2_seen": p["crowding_when_seen"][2.0],
        "outside_seen": p["ball_outside_pitch_when_seen"],
    }


def loss(got):
    """Squared log-ratio for durations and speeds, squared differences for shares."""
    total = 0.0
    for group in ("gap_s", "seen_s"):
        for key, want in TARGET[group].items():
            total += np.log(max(got[group][key], 1e-3) / want) ** 2
    for key in ("speed_before_gap_p50", "speed_seen_p50"):
        total += np.log(max(got[key], 1e-3) / TARGET[key]) ** 2
    for key in ("near_line_before", "near_line_seen", "box_before", "box_seen", "outside_seen"):
        total += (10 * (got[key] - TARGET[key])) ** 2
    for key in ("crowd2_before", "crowd2_seen"):
        total += (2 * (got[key] - TARGET[key])) ** 2
    return float(total) if np.isfinite(total) else float("inf")


def player_summary(halves, params):
    """{statistic: {percentile: m/s}} of the players' speeds on the halves degraded with params."""
    degraded = [degrade(h, params, seed=i) for i, h in enumerate(halves)]
    p = profile(degraded, params.pitch)
    return {name: {k: p[name].get(k, np.nan) for k in PLAYER_PERCENTILES}
            for name in PLAYER_STATISTICS}


def player_loss(got, target):
    """Squared log-ratio of every percentile to the target's."""
    total = sum(np.log(max(got[name][k], 1e-3) / want) ** 2
                for name, wants in target.items() for k, want in wants.items())
    return float(total) if np.isfinite(total) else float("inf")


def fit_players(halves, target=None):
    """(best player_noise_m, its loss) over NOISE_GRID, printing every level's percentiles; the
    best is None without a target."""
    def cells(summary):
        return "  ".join(" ".join("{:5.2f}".format(summary[name][k]) for k in PLAYER_PERCENTILES)
                         for name in PLAYER_STATISTICS)

    print("players' speeds over 1 s, {} in m/s: mean over the players, then each player".format(
        "/".join(PLAYER_PERCENTILES)))
    best = (None, float("inf"))
    for noise in NOISE_GRID:
        got = player_summary(halves, replace(DegradeParams(), player_noise_m=noise))
        value = player_loss(got, target) if target else float("nan")
        print("  noise {:.3f} m  {}  loss {:.3f}".format(noise, cells(got), value))
        if target and value < best[1]:
            best = (noise, value)
    if target:
        print("  match          {}".format(cells(target)))
        print("best player_noise_m {:.3f} (loss {:.3f})".format(*best))
    else:
        print("no TARGET_PLAYERS yet: run scripts/atd_profile.py on the laptop first")
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--samples", type=int, default=80)
    parser.add_argument("--local", type=int, default=120)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--players", action="store_true",
                        help="fit the players' position noise to TARGET_PLAYERS instead (M11g)")
    parser.add_argument("matches", nargs="*")
    args = parser.parse_args()
    matches = args.matches or (sorted(ex.MATCHES) if args.players
                               else ["J03WOH", "J03WMX", "J03WQQ"])

    halves = []
    for match_id in matches:
        _, match_halves, _ = ex.load_match(match_id)
        halves += [match_halves[s] for s in ex.SECTIONS]
    if args.players:
        print("{} halves of {}".format(len(halves), ", ".join(matches)))
        fit_players(halves, TARGET_PLAYERS)
        return 0

    rng = np.random.default_rng(args.seed)
    best = (loss(summary(halves, DegradeParams())), DegradeParams())
    print("defaults: loss {:.3f}".format(best[0]))
    for i in range(args.samples):
        candidate = replace(DegradeParams(), **{k: float(rng.uniform(*b)) for k, b in SEARCH.items()})
        value = loss(summary(halves, candidate))
        if value < best[0]:
            best = (value, candidate)
            print("sample {:>3}: loss {:.3f}".format(i, value))
    for i in range(args.local):
        scale = 0.25 * (1 - i / max(args.local, 1)) + 0.03
        moved = {k: float(np.clip(getattr(best[1], k) + rng.normal(0, scale * (b[1] - b[0])), *b))
                 for k, b in SEARCH.items()}
        candidate = replace(best[1], **moved)
        value = loss(summary(halves, candidate))
        if value < best[0]:
            best = (value, candidate)
            print("local  {:>3}: loss {:.3f}".format(i, value))

    value, params = best
    print("\nbest loss {:.3f}".format(value))
    print("  " + ", ".join("{}={:.3f}".format(k, getattr(params, k)) for k in SEARCH))
    got = summary(halves, params)
    print("\n{:<22} {:>10} {:>10}".format("statistic", "degraded", "match"))
    for key, want in TARGET.items():
        if isinstance(want, dict):
            for sub, w in want.items():
                print("{:<22} {:>10.2f} {:>10.2f}".format(key + "." + sub, got[key][sub], w))
        else:
            print("{:<22} {:>10.3f} {:>10.3f}".format(key, got[key], want))
    return 0


if __name__ == "__main__":
    sys.exit(main())
