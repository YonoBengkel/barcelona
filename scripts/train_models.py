"""Fit the dead-ball and team models and the decoder on the practice ground, match by match.

Each IDSSE match is held out in turn: the models are fitted on the other six (every half degraded
with --seeds seeds, every --step-th frame kept) and judged on the held-out match, degraded with a
seed no training copy used. For each model the C with the lowest pooled log loss wins. The
baseline's own rules are scored beside them on the same frames: "no ball means dead" and "the
nearest player's team has it".

The decoder (possession.decode) then turns those held-out probabilities into labels for every
setting of a grid of its three costs, and each setting is scored on each match with the official
metrics: possessions matched by overlap (spellF1) and frame macroF1. Each match's setting is chosen
on the other six matches alone, so the score reported for the whole pipeline is a held-out score
as well, except for C, which is chosen once on all seven (it barely matters, see the log loss
lines); the setting written out is the one that does best on all seven. The organisers' baseline
(nearest player, majority over +/- 2 s) is scored on the same frames, at label tolerances of 0.5,
1 and 2 s.

Last, the models are refitted on all seven matches and written, with the chosen decoder, to
models/possession_v1.json. Public data only; reads no match data. A job for Kaggle
(kaggle/README.md), not for the team's laptop.

    python scripts/train_models.py [--seeds 2] [--step 8] [--out models/possession_v1.json]
"""

import argparse
import collections
import itertools
import os
import sys
import time

import numpy as np
from sklearn import __version__ as sklearn_version
from sklearn.metrics import log_loss, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from possession import external as ex  # noqa: E402
from possession import practice  # noqa: E402
from possession.decode import viterbi  # noqa: E402
from possession.features import FPS, HOME, compute, swapped  # noqa: E402
from possession.models import design, fit, log_state_probabilities, probability, save  # noqa: E402
from possession.predictors import NOBODY, nearest_team, smooth  # noqa: E402

pm = practice.pm
PITCH = (105.0, 68.0)
GRID = (0.01, 0.1, 1.0)
EVAL_SEED = 99
TOLERANCES = (0.5, 1.0, 2.0)
BASELINE_WINDOW = 50
# The decoder's costs in nats of evidence, and the bias on the D evidence per frame.
DECODER_AXES = {"team_switch": (8, 16, 32, 64, 128, 256, 512),
                "dead_switch": (8, 16, 32, 64, 128, 256, 512),
                "dead_bias": (-1.0, -0.5, 0.0, 0.5, 1.0)}
DECODER_GRID = tuple(itertools.product(*DECODER_AXES.values()))


def build(match_ids, seeds, step):
    """Training rows per match, and the full held-out halves per match for evaluation."""
    train = {m: {"dead": ([], []), "team": ([], [])} for m in match_ids}
    for seed in range(seeds):
        for case in practice.cases(match_ids, tolerance_s=1.0, seed=seed):
            labels = case["labels"]
            X = design(compute(case["half"], PITCH)[0]).astype(np.float32)
            mirrored = design(compute(swapped(case["half"]), PITCH)[0]).astype(np.float32)
            rows = train[case["match"]]
            every = np.arange(0, len(labels), step)
            rows["dead"][0].append(X[every])
            rows["dead"][1].append(labels[every] == "D")
            live = np.flatnonzero(labels != "D")[::step]
            rows["team"][0].extend([X[live], mirrored[live]])
            rows["team"][1].extend([labels[live] == "H", labels[live] == "A"])
    held_out = {}
    for case in practice.cases(match_ids, tolerance_s=1.0, seed=EVAL_SEED):
        X = design(compute(case["half"], PITCH)[0]).astype(np.float32)
        part = held_out.setdefault(case["match"], {"X": [], "labels": [], "nearest": [],
                                                   "ball_known": [], "lengths": []})
        part["X"].append(X)
        part["labels"].append(case["labels"])
        part["nearest"].append(nearest_team(case["half"]))
        part["ball_known"].append(X[:, 0] + X[:, 1] > 0)  # ball_seen or ball_bridged
        part["lengths"].append(len(X))
    for part in held_out.values():
        for key in ("X", "labels", "nearest", "ball_known"):
            part[key] = np.concatenate(part[key])
    return train, held_out


def halves_of(part, key):
    """A held-out match's array split back into its halves."""
    return np.split(part[key], np.cumsum(part["lengths"])[:-1])


def stack(train, matches, model):
    X = np.concatenate([x for m in matches for x in train[m][model][0]])
    y = np.concatenate([y for m in matches for y in train[m][model][1]])
    return X, y


def evaluate(train, held_out, matches, C):
    """Pooled held-out predictions over all folds for one C, and each fold's models."""
    pooled = {"dead_p": [], "dead_y": [], "team_p": [], "team_y": [], "nearest": [],
              "known": [], "no_ball": []}
    folds = {}
    for match in matches:
        others = [m for m in matches if m != match]
        dead_model = fit(*stack(train, others, "dead"), C=C)
        team_model = fit(*stack(train, others, "team"), C=C)
        folds[match] = {"dead": dead_model, "team": team_model}
        part = held_out[match]
        live = part["labels"] != "D"
        pooled["dead_p"].append(probability(dead_model, part["X"]))
        pooled["dead_y"].append(part["labels"] == "D")
        pooled["no_ball"].append(~part["ball_known"])
        pooled["team_p"].append(probability(team_model, part["X"][live]))
        pooled["team_y"].append(part["labels"][live] == "H")
        pooled["nearest"].append(part["nearest"][live])
        pooled["known"].append(part["ball_known"][live])
    return {key: np.concatenate(value) for key, value in pooled.items()}, folds


def report(C, r):
    dead_acc = np.mean((r["dead_p"] > 0.5) == r["dead_y"])
    rule_acc = np.mean(r["no_ball"] == r["dead_y"])
    team_acc = np.mean((r["team_p"] > 0.5) == r["team_y"])
    has_nearest = r["nearest"] != NOBODY
    nearest_acc = np.mean((r["nearest"][has_nearest] == HOME) == r["team_y"][has_nearest])
    ours_same = np.mean((r["team_p"][has_nearest] > 0.5) == r["team_y"][has_nearest])
    ours_blind = np.mean((r["team_p"][~has_nearest] > 0.5) == r["team_y"][~has_nearest])
    print("C={:<5} dead: logloss {:.4f} AUC {:.4f} acc {:.4f} (no-ball rule {:.4f}) | "
          "team: logloss {:.4f} AUC {:.4f} acc {:.4f}; where a nearest player exists ours {:.4f} "
          "vs nearest {:.4f}; without one ours {:.4f} ({:.0%} of live frames)".format(
              C, log_loss(r["dead_y"], r["dead_p"]), roc_auc_score(r["dead_y"], r["dead_p"]),
              dead_acc, rule_acc, log_loss(r["team_y"], r["team_p"]),
              roc_auc_score(r["team_y"], r["team_p"]), team_acc, ours_same, nearest_acc,
              ours_blind, np.mean(~has_nearest)))
    return log_loss(r["dead_y"], r["dead_p"]), log_loss(r["team_y"], r["team_p"])


def decoded(log_ps, settings):
    """Per setting in turn, the match's decoded codes with an X between halves."""
    params = [np.array([s[i] for s in settings], dtype=np.float64) for i in range(3)]
    paths = [viterbi(log_p, *params) for log_p in log_ps]
    for k in range(len(settings)):
        yield practice.joined([path[:, k] for path in paths], gap=pm.X)


def best(table, matches):
    """Index of the setting with the best pooled (spellF1, macroF1) on matches; ties to the first."""
    scores = [practice.pooled([row[m] for m in matches]) for row in table]
    return max(range(len(table)), key=lambda k: (scores[k], -k))


def labels_at(match, tolerances):
    """{tolerance: the match's aligned labels per half}. Degradation leaves labels alone, so they
    line up with the held-out halves frame by frame."""
    _, halves, events = ex.load_match(match)
    out = {}
    for tolerance in tolerances:
        labels, _, _ = ex.aligned_labels(halves, events, tolerance)
        out[tolerance] = [labels[section] for section in ex.SECTIONS]
    return out


def spell_rate(codes, gt_codes):
    """(possessions per 10 minutes of graded play, their mean length in s) in codes."""
    runs = pm.valid_runs(gt_codes)
    stats = pm.possession_stats(np.where(gt_codes == pm.X, pm.X, codes), runs)
    minutes = sum(stop - start for start, stop in runs) / FPS / 60.0
    return stats["n_spells"] / minutes * 10.0, stats["mean_duration_s"]


def fit_decoder(held_out, oof, matches):
    """Choose the decoder setting, nested by match; report it, and return (setting, range)."""
    start = time.time()
    table = [dict() for _ in DECODER_GRID]
    for match in matches:
        gt = practice.joined([practice.codes(l) for l in halves_of(held_out[match], "labels")],
                             gap=pm.X)
        for k, pred in enumerate(decoded(oof[match], DECODER_GRID)):
            table[k][match] = practice.counts(gt, pred)
        print("  decoder grid on {}: {:.0f} s so far".format(match, time.time() - start))
    chosen = {m: best(table, [o for o in matches if o != m]) for m in matches}
    final = best(table, matches)

    rows = collections.defaultdict(list)
    rates = collections.defaultdict(list)
    for match in matches:
        nested, overall = decoded(oof[match], [DECODER_GRID[chosen[match]], DECODER_GRID[final]])
        nearest = halves_of(held_out[match], "nearest")
        base = practice.joined([practice.codes(smooth(n, BASELINE_WINDOW)) for n in nearest],
                               gap=pm.X)
        other = labels_at(match, [t for t in TOLERANCES if t != 1.0])
        for tolerance in TOLERANCES:
            labels = halves_of(held_out[match], "labels") if tolerance == 1.0 else other[tolerance]
            gt = practice.joined([practice.codes(l) for l in labels], gap=pm.X)
            for name, pred in (("nested", nested), ("all seven", overall), ("baseline", base)):
                rows[(name, tolerance)].append(practice.counts(gt, pred))
            if tolerance == 1.0:
                rates["labels"].append(spell_rate(gt, gt))
                rates["ours"].append(spell_rate(nested, gt))

    print("\ndecoder: {} settings of {}, each match's chosen on the other six; C was chosen once "
          "on all seven ({:.0f} s)".format(len(DECODER_GRID), ", ".join(DECODER_AXES),
                                           time.time() - start))
    print("{:<10} {:>28} {:>28} {:>28}".format("labels", "ours, held out",
                                               "ours, set on all seven", "baseline_smooth50"))
    for tolerance in TOLERANCES:
        cells = ["spellF1 {:.4f}  mF1 {:.4f}".format(*practice.pooled(rows[(name, tolerance)]))
                 for name in ("nested", "all seven", "baseline")]
        print("+/-{:<7} {:>28} {:>28} {:>28}".format("{:g} s".format(tolerance), *cells))
    per_match = [practice.pooled([part])[0] for part in rows[("nested", 1.0)]]
    print("held-out spellF1 per match: {:.3f}-{:.3f}".format(min(per_match), max(per_match)))
    picks = collections.Counter(DECODER_GRID[k] for k in chosen.values())
    print("settings chosen per match: " + "; ".join(
        "{} x{}".format(dict(zip(DECODER_AXES, setting)), n) for setting, n in picks.most_common()))
    setting = dict(zip(DECODER_AXES, DECODER_GRID[final]))
    edges = ["{} {}".format(name, "upper" if value == DECODER_AXES[name][-1] else "lower")
             for name, value in setting.items()
             if value in (DECODER_AXES[name][0], DECODER_AXES[name][-1])]
    print("chosen on all seven: {}{}".format(setting, "  AT THE EDGE OF THE GRID: " + ", ".join(
        edges) + "; widen it" if edges else ""))

    practice_range = {}
    for name, values in rates.items():
        rate, length = np.array(values).T
        practice_range[name] = {"spells_per_10min": [round(float(rate.min()), 2),
                                                     round(float(rate.max()), 2)],
                                "mean_spell_s": [round(float(length.min()), 2),
                                                 round(float(length.max()), 2)]}
        print("possessions, {}: {:.1f}-{:.1f} per 10 min, mean {:.1f}-{:.1f} s".format(
            name, rate.min(), rate.max(), length.min(), length.max()))
    return setting, practice_range


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--step", type=int, default=8)
    parser.add_argument("--out", default=os.path.join(ROOT, "models", "possession_v1.json"))
    args = parser.parse_args()
    matches = sorted(ex.MATCHES)

    start = time.time()
    train, held_out = build(matches, args.seeds, args.step)
    print("built rows in {:.0f} s: dead {}, team {}".format(
        time.time() - start, len(stack(train, matches, "dead")[1]),
        len(stack(train, matches, "team")[1])))

    losses, folds = {}, {}
    for C in GRID:
        result, folds[C] = evaluate(train, held_out, matches, C)
        losses[C] = report(C, result)
    best_dead = min(GRID, key=lambda c: losses[c][0])
    best_team = min(GRID, key=lambda c: losses[c][1])
    print("chosen C: dead {}, team {} ({:.0f} s)".format(best_dead, best_team,
                                                          time.time() - start))

    oof = {m: [log_state_probabilities({"dead": folds[best_dead][m]["dead"],
                                        "team": folds[best_team][m]["team"]}, X)
               for X in halves_of(held_out[m], "X")] for m in matches}
    decoder, practice_range = fit_decoder(held_out, oof, matches)

    models = {"dead": fit(*stack(train, matches, "dead"), C=best_dead),
              "team": fit(*stack(train, matches, "team"), C=best_team),
              "decoder": decoder,
              "fitted_on": {"matches": matches, "seeds": args.seeds, "step": args.step,
                            "label_tolerance_s": 1.0, "pitch": list(PITCH),
                            "scikit_learn": sklearn_version,
                            "decoder_grid": {k: list(v) for k, v in DECODER_AXES.items()},
                            "practice_range": practice_range}}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save(models, args.out)
    print("wrote {} ({:.0f} s in all)".format(os.path.relpath(args.out, ROOT),
                                              time.time() - start))
    return 0


if __name__ == "__main__":
    sys.exit(main())
