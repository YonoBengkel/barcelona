"""Fit the dead-ball and team models and the decoder on the practice ground, match by match.

Each IDSSE match is held out in turn: the models are fitted on the other six (every half degraded
with --seeds seeds, every --step-th frame kept) and judged on the held-out match, degraded with a
seed no training copy used. The dead-ball model is a logistic regression whose C is chosen by
pooled held-out log loss. The team model comes in the kinds named by --team: a logistic
regression (C chosen the same way), and gradient-boosted trees (M11d), whose settings in
TREE_GRID are chosen the same way. Each kind becomes its own variant: 'v1' with the logistic
team model, 'v1+trees' with the trees. The baseline's own rules are scored beside them on the
same frames: "no ball means dead" and "the nearest player's team has it".

The decoder (possession.decode) then turns those held-out probabilities into labels for every
setting of a grid of its three costs, and each setting is scored on each match with the official
metrics: possessions matched by overlap (spellF1) and frame macroF1. Each match's setting is chosen
on the other six matches alone, so the score reported for the whole pipeline is a held-out score
as well, except for C, which is chosen once on all seven (it barely matters, see the log loss
lines); the setting written out is the one that does best on all seven. The organisers' baseline
(nearest player, majority over +/- 2 s) is scored on the same frames, at label tolerances of 0.5,
1 and 2 s.

It then prints where the held-out pipeline goes wrong (possession.diagnose): its calls and
errors by ball state, and the cause of every possession it misses. With --oracle it also decodes
the held-out evidence with its team call, its dead-ball call or both turned to the truth, at the
model's own confidence, which bounds what fixing each part could gain.

Last, the models are refitted on all seven matches and written, with the chosen decoder, to
models/possession_<variant>.json.

All of this runs for every design of the features named (possession.models.DESIGNS) on the same
degraded halves. Each variant other than v1 is then set against v1 match by match, and the
adoption rule fixed before the run is applied: a variant replaces v1 only if its held-out
spellF1 at +/-1 s is at least 0.01 higher, it is better on at least 5 of the 7 matches, it loses
nothing at +/-0.5 s or +/-2 s, and its macroF1 falls by at most 0.005.

Public data only; reads no match data. A job for Kaggle (kaggle/README.md), not for the team's
laptop.

    python scripts/train_models.py [--designs v1 [v2 ...]] [--team logistic [trees]]
                                   [--seeds 2] [--step 8] [--out-dir models] [--oracle]
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

from possession import diagnose  # noqa: E402
from possession import external as ex  # noqa: E402
from possession import practice  # noqa: E402
from possession.decode import viterbi  # noqa: E402
from possession.features import FPS, HOME, compute, swapped  # noqa: E402
from possession.models import (DESIGNS, design, design_names, fit, fit_trees,  # noqa: E402
                               log_state_probabilities, probability, save)
from possession.predictors import NOBODY, nearest_team, smooth  # noqa: E402

pm = practice.pm
PITCH = (105.0, 68.0)
GRID = (0.01, 0.1, 1.0)
TEAM_KINDS = ("logistic", "trees")
# M11d: gradient-boosted team models, chosen among by held-out log loss like C.
TREE_GRID = ({"max_iter": 200, "max_leaf_nodes": 31, "learning_rate": 0.1, "min_samples_leaf": 20},
             {"max_iter": 300, "max_leaf_nodes": 63, "learning_rate": 0.05,
              "min_samples_leaf": 100})
EVAL_SEED = 99
TOLERANCES = (0.5, 1.0, 2.0)
BASELINE_WINDOW = 50
# The decoder's costs in nats of evidence, and the bias on the D evidence per frame.
DECODER_AXES = {"team_switch": (8, 16, 32, 64, 128, 256, 512),
                "dead_switch": (8, 16, 32, 64, 128, 256, 512),
                "dead_bias": (-1.0, -0.5, 0.0, 0.5, 1.0)}
DECODER_GRID = tuple(itertools.product(*DECODER_AXES.values()))


def build(match_ids, seeds, step, designs=("v1",)):
    """Training rows per design and match, and the held-out halves per match for evaluation,
    with one feature matrix per design under 'X'. The features are computed once per half and
    only the design differs, so every design sees exactly the same frames."""
    train = {v: {m: {"dead": ([], []), "team": ([], [])} for m in match_ids} for v in designs}
    for seed in range(seeds):
        for case in practice.cases(match_ids, tolerance_s=1.0, seed=seed):
            labels = case["labels"]
            features = compute(case["half"], PITCH)[0]
            mirrored_features = compute(swapped(case["half"]), PITCH)[0]
            every = np.arange(0, len(labels), step)
            live = np.flatnonzero(labels != "D")[::step]
            for version in designs:
                X = design(features, version).astype(np.float32)
                mirrored = design(mirrored_features, version).astype(np.float32)
                rows = train[version][case["match"]]
                rows["dead"][0].append(X[every])
                rows["dead"][1].append(labels[every] == "D")
                rows["team"][0].extend([X[live], mirrored[live]])
                rows["team"][1].extend([labels[live] == "H", labels[live] == "A"])
    held_out = {}
    for case in practice.cases(match_ids, tolerance_s=1.0, seed=EVAL_SEED):
        features = compute(case["half"], PITCH)[0]
        part = held_out.setdefault(case["match"], {"X": {v: [] for v in designs}, "labels": [],
                                                   "nearest": [], "ball_known": [],
                                                   "lengths": []})
        for version in designs:
            part["X"][version].append(design(features, version).astype(np.float32))
        part["labels"].append(case["labels"])
        part["nearest"].append(nearest_team(case["half"]))
        part["ball_known"].append(features[:, 0] + features[:, 1] > 0)  # seen or bridged
        part["lengths"].append(len(features))
    for part in held_out.values():
        for key in ("labels", "nearest", "ball_known"):
            part[key] = np.concatenate(part[key])
        part["X"] = {version: np.concatenate(X) for version, X in part["X"].items()}
    return train, held_out


def halves_of(part, key):
    """A held-out match's array split back into its halves."""
    return np.split(part[key], np.cumsum(part["lengths"])[:-1])


def stack(train, matches, model):
    X = np.concatenate([x for m in matches for x in train[m][model][0]])
    y = np.concatenate([y for m in matches for y in train[m][model][1]])
    return X, y


def lomo(train, held_out, matches, model, fitter):
    """Each match held out in turn: fitter on the other six matches' rows of model ('dead' or
    'team'). Returns ({match: fitted}, pooled held-out probabilities, their labels); the team
    model is judged on live frames only."""
    folds, p, y = {}, [], []
    for match in matches:
        others = [m for m in matches if m != match]
        folds[match] = fitter(*stack(train, others, model))
        part = held_out[match]
        rows = slice(None) if model == "dead" else part["labels"] != "D"
        p.append(probability(folds[match], part["X"][rows]))
        y.append(part["labels"][rows] == ("D" if model == "dead" else "H"))
    return folds, np.concatenate(p), np.concatenate(y)


def rules(held_out, matches):
    """The baseline's own rules on the held-out frames, to score the models against."""
    live = [held_out[m]["labels"] != "D" for m in matches]
    dead_y = np.concatenate([held_out[m]["labels"] == "D" for m in matches])
    no_ball = np.concatenate([~held_out[m]["ball_known"] for m in matches])
    return {"no_ball_acc": float(np.mean(no_ball == dead_y)),
            "nearest": np.concatenate([held_out[m]["nearest"][l] for m, l in zip(matches, live)])}


def report_dead(label, p, y, rule):
    print("dead {:<34} logloss {:.4f} AUC {:.4f} acc {:.4f} (no-ball rule {:.4f})".format(
        label, log_loss(y, p), roc_auc_score(y, p), np.mean((p > 0.5) == y), rule["no_ball_acc"]))
    return log_loss(y, p)


def report_team(label, p, y, rule):
    has_nearest = rule["nearest"] != NOBODY
    print("team {:<34} logloss {:.4f} AUC {:.4f} acc {:.4f}; where a nearest player exists ours "
          "{:.4f} vs nearest {:.4f}; without one ours {:.4f} ({:.0%} of live frames)".format(
              label, log_loss(y, p), roc_auc_score(y, p), np.mean((p > 0.5) == y),
              np.mean((p[has_nearest] > 0.5) == y[has_nearest]),
              np.mean((rule["nearest"][has_nearest] == HOME) == y[has_nearest]),
              np.mean((p[~has_nearest] > 0.5) == y[~has_nearest]), np.mean(~has_nearest)))
    return log_loss(y, p)


def team_candidates(kind, version):
    """[(label, fitter)]: the team models of one kind, chosen among by held-out log loss."""
    if kind == "logistic":
        return [("logistic C={:g}".format(C),
                 lambda X, y, C=C: fit(X, y, C=C, version=version)) for C in GRID]
    return [("trees {max_iter}x{max_leaf_nodes} lr {learning_rate:g} leaf>={min_samples_leaf}"
             .format(**params), lambda X, y, params=params: fit_trees(X, y, version=version,
                                                                      **params))
            for params in TREE_GRID]


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
    """Choose the decoder setting, nested by match, and report it.

    Returns (setting, practice range, held-out codes per match, summary), where summary holds
    the held-out (spellF1, macroF1) per label tolerance and the held-out spellF1 per match.
    """
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
    held_out_codes = {}
    for match in matches:
        nested, overall = decoded(oof[match], [DECODER_GRID[chosen[match]], DECODER_GRID[final]])
        held_out_codes[match] = nested
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
    per_match = {match: practice.pooled([part])[0]
                 for match, part in zip(matches, rows[("nested", 1.0)])}
    print("held-out spellF1 per match: " + ", ".join(
        "{} {:.3f}".format(match, value) for match, value in per_match.items()))
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
    summary = {"pooled": {t: practice.pooled(rows[("nested", t)]) for t in TOLERANCES},
               "per_match": per_match}
    return setting, practice_range, held_out_codes, summary


def report_errors(held_out, predicted, matches):
    """Where the held-out pipeline goes wrong (M10): by ball state, and possession by possession.

    predicted: per match, the codes the nested decoder gave, X between halves. The same ball-state
    table, without the truth columns, is what scripts/make_submission.py prints for the match.
    """
    states, preds, truths, speeds, outcomes, shares = [], [], [], [], [], []
    for match in matches:
        part = held_out[match]
        gt = practice.joined([practice.codes(l) for l in halves_of(part, "labels")], gap=pm.X)
        state = practice.joined([diagnose.ball_states(X) for X in halves_of(part, "X")], gap=-1)
        speed = practice.joined([diagnose.player_speed(X, logged=True)
                                 for X in halves_of(part, "X")], gap=np.nan)
        graded = gt != pm.X
        states.append(state[graded])
        preds.append(predicted[match][graded])
        truths.append(gt[graded])
        speeds.append(speed[graded])
        outcomes.append(diagnose.spell_outcomes(gt, predicted[match]))
        shares.append("{} {:.1%}/{:.1%}".format(match, np.mean(gt[graded] == pm.D),
                                                np.mean(predicted[match][graded] == pm.D)))
    states, preds, truths, speeds = (np.concatenate(a) for a in (states, preds, truths, speeds))
    print("\nerrors of the held-out pipeline (M10), labels at +/-1 s")
    for line in diagnose.format_state_table(diagnose.state_table(states, preds, truths, speeds)):
        print(line)
    for line in diagnose.format_confusion(truths, preds):
        print(line)
    for line in diagnose.format_outcomes(diagnose.pool(outcomes)):
        print(line)
    print("dead-ball share per match, true/ours: " + ", ".join(shares))


ORACLES = ("model", "true team", "true dead", "true both")


def oracle_evidence(log_p, truth, variant):
    """The held-out evidence with its team call, its dead-ball call or both turned to the truth,
    each kept at the model's own confidence: a frame that said 70% home on a true away frame now
    says 70% away. What is fixed is which way the evidence points, never how sure it is.

    log_p: (frames, 3) log-probabilities of H, A, D; truth: the frames' codes.
    """
    log_alive = np.logaddexp(log_p[:, 0], log_p[:, 1])
    team = log_p[:, :2] - log_alive[:, None]  # log P(H | alive), log P(A | alive)
    dead = np.column_stack([log_alive, log_p[:, 2]])  # log P(alive), log P(D)
    if variant in ("true team", "true both"):
        sure, unsure = team.max(axis=1), team.min(axis=1)
        team = np.where((truth == pm.H)[:, None], np.column_stack([sure, unsure]),
                        np.where((truth == pm.A)[:, None], np.column_stack([unsure, sure]), team))
    if variant in ("true dead", "true both"):
        sure, unsure = dead.max(axis=1), dead.min(axis=1)
        dead = np.where((truth == pm.D)[:, None], np.column_stack([unsure, sure]),
                        np.column_stack([sure, unsure]))
    return np.column_stack([dead[:, 0] + team[:, 0], dead[:, 0] + team[:, 1], dead[:, 1]])


def oracle_headroom(held_out, oof, matches):
    """How far the pipeline is from what its decoder could do with right evidence (M11c).

    For each variant of oracle_evidence, the best pooled spellF1 over the decoder grid on all
    seven matches, labels at +/-1 s. It is an upper bound chosen in sample, not a held-out score:
    the gap between rows says which part, team or dead ball, holds the remaining score.
    """
    start = time.time()
    print("\nperfect-evidence check (M11c), best setting on all seven, labels at +/-1 s; the "
          "evidence keeps the model's confidence and only its direction is corrected")
    truths = {m: [practice.codes(l) for l in halves_of(held_out[m], "labels")] for m in matches}
    gts = {m: practice.joined(truths[m], gap=pm.X) for m in matches}
    for variant in ORACLES:
        table = [dict() for _ in DECODER_GRID]
        for match in matches:
            evidence = [oracle_evidence(log_p, truth, variant)
                        for log_p, truth in zip(oof[match], truths[match])]
            for k, pred in enumerate(decoded(evidence, DECODER_GRID)):
                table[k][match] = practice.counts(gts[match], pred)
        k = best(table, matches)
        spell, macro = practice.pooled([table[k][m] for m in matches])
        print("  {:<10} spellF1 {:.4f}  mF1 {:.4f}  at {} ({:.0f} s)".format(
            variant, spell, macro, dict(zip(DECODER_AXES, DECODER_GRID[k])),
            time.time() - start))


def variant_name(version, kind):
    """'v1' for the logistic team model, 'v1+trees' for the gradient-boosted one."""
    return version if kind == "logistic" else "{}+{}".format(version, kind)


def fit_design(version, team_kinds, train, held_out, matches, args, start):
    """One design: the dead-ball model once, then for each kind of team model its decoder, error
    report and file models/possession_<variant>.json.

    held_out carries this design's feature matrix under 'X'. Returns {variant: fit_decoder's
    summary}.
    """
    print("\n===== design {}: {} columns =====".format(version, len(design_names(version))))
    rule = rules(held_out, matches)
    dead = {}
    for C in GRID:
        folds, p, y = lomo(train, held_out, matches, "dead",
                           lambda X, y, C=C: fit(X, y, C=C, version=version))
        dead[C] = (report_dead("logistic C={:g}".format(C), p, y, rule), folds)
    best_dead = min(GRID, key=lambda c: dead[c][0])
    dead_model = fit(*stack(train, matches, "dead"), C=best_dead, version=version)
    print("chosen dead C={:g} ({:.0f} s)".format(best_dead, time.time() - start))

    summaries = {}
    for kind in team_kinds:
        team = {}
        for label, fitter in team_candidates(kind, version):
            folds, p, y = lomo(train, held_out, matches, "team", fitter)
            team[label] = (report_team(label, p, y, rule), folds, fitter)
        chosen = min(team, key=lambda label: team[label][0])
        variant = variant_name(version, kind)
        print("\n----- {}: team {} ({:.0f} s) -----".format(variant, chosen, time.time() - start))

        oof = {m: [log_state_probabilities({"dead": dead[best_dead][1][m],
                                            "team": team[chosen][1][m]}, X)
                   for X in halves_of(held_out[m], "X")] for m in matches}
        decoder, practice_range, held_out_codes, summary = fit_decoder(held_out, oof, matches)
        report_errors(held_out, held_out_codes, matches)
        if args.oracle:
            oracle_headroom(held_out, oof, matches)

        models = {"design": version, "dead": dead_model,
                  "team": team[chosen][2](*stack(train, matches, "team")),
                  "decoder": decoder,
                  "fitted_on": {"matches": matches, "seeds": args.seeds, "step": args.step,
                                "label_tolerance_s": 1.0, "pitch": list(PITCH),
                                "scikit_learn": sklearn_version, "team_model": chosen,
                                "decoder_grid": {k: list(v) for k, v in DECODER_AXES.items()},
                                "practice_range": practice_range}}
        out = os.path.join(args.out_dir, "possession_{}.json".format(variant))
        os.makedirs(args.out_dir, exist_ok=True)
        save(models, out)
        print("wrote {} ({:.0f} s so far)".format(os.path.relpath(out, ROOT),
                                                  time.time() - start))
        summaries[variant] = summary
    return summaries


def compare(summaries, matches):
    """Each variant (a design, a kind of team model) against v1, match by match, with the
    adoption rule fixed before the run."""
    reference = summaries["v1"]
    for version, summary in summaries.items():
        if version == "v1":
            continue
        gains = {m: summary["per_match"][m] - reference["per_match"][m] for m in matches}
        better = sum(gain > 0 for gain in gains.values())
        spell = {t: summary["pooled"][t][0] - reference["pooled"][t][0] for t in TOLERANCES}
        macro = summary["pooled"][1.0][1] - reference["pooled"][1.0][1]
        adopt = (spell[1.0] >= 0.01 and better >= 5 and spell[0.5] >= 0 and spell[2.0] >= 0
                 and macro >= -0.005)
        print("\n{} against v1, held out: spellF1 {:+.4f} at +/-1 s ({:+.4f} at 0.5 s, {:+.4f} "
              "at 2 s), macroF1 {:+.4f}; better on {} of {} matches".format(
                  version, spell[1.0], spell[0.5], spell[2.0], macro, better, len(matches)))
        print("  per match: " + ", ".join("{} {:+.3f}".format(m, g) for m, g in gains.items()))
        print("  adoption rule (spellF1 +0.01 or more at 1 s, better on 5 or more matches, no "
              "loss at 0.5 s or 2 s, macroF1 down 0.005 at most): {}".format(
                  "ADOPT " + version if adopt else "keep v1"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # v2 lost to v1 on 2026-11-09 (runs/20261109T0833Z-train_models); run --designs v1 v2 to see it.
    parser.add_argument("--designs", nargs="+", default=["v1"], choices=sorted(DESIGNS))
    # M11d sets the gradient-boosted team model against the logistic one on the same frames.
    parser.add_argument("--team", nargs="+", default=["logistic", "trees"], choices=TEAM_KINDS)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--step", type=int, default=8)
    parser.add_argument("--out-dir", default=os.path.join(ROOT, "models"))
    # Run once for M11c (runs/20261109T0856Z-train_models): the team model holds most of the
    # missing score. Off by default; it adds about 9 minutes.
    parser.add_argument("--oracle", action=argparse.BooleanOptionalAction, default=False,
                        help="also print the perfect-evidence check (about 9 minutes)")
    args = parser.parse_args()
    matches = sorted(ex.MATCHES)

    start = time.time()
    train, held_out = build(matches, args.seeds, args.step, args.designs)
    first = args.designs[0]
    print("built rows in {:.0f} s: dead {}, team {}".format(
        time.time() - start, len(stack(train[first], matches, "dead")[1]),
        len(stack(train[first], matches, "team")[1])))

    summaries = {}
    for version in args.designs:
        view = {m: dict(part, X=part["X"][version]) for m, part in held_out.items()}
        summaries.update(fit_design(version, args.team, train[version], view, matches, args,
                                    start))
    if "v1" in summaries and len(summaries) > 1:
        compare(summaries, matches)
    print("\ndone in {:.0f} s".format(time.time() - start))
    return 0


if __name__ == "__main__":
    sys.exit(main())
