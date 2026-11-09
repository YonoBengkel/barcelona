"""Fit the dead-ball and team models and the decoder on the practice ground, match by match.

Each IDSSE match is held out in turn: the models are fitted on the other six (every half degraded
with --seeds seeds, every --step-th frame kept) and judged on the held-out match, degraded with a
seed no training copy used. Both models come in kinds, named by --dead and --team: a logistic
regression (C chosen by pooled held-out log loss) and gradient-boosted trees (settings in TREES,
chosen the same way); the team model also has 'trees-wide', a larger tree model. Every pair of a
dead-ball kind and a team kind is a variant: 'v1' for two logistic models, 'v1+trees' for a tree
team model (adopted in M11d), '+deadtrees' for a tree dead-ball model ('v1+trees-wide+deadtrees'
was adopted in M11e). The baseline's own rules are scored beside them on the same frames: "no ball
means dead" and "the nearest player's team has it".

The decoder (possession.decode) then turns those held-out probabilities into labels for every
setting of a grid of its three costs, and each setting is scored on each match with the official
metrics: possessions matched by overlap (spellF1) and frame macroF1. Each match's setting is chosen
on the other six matches alone, so the score reported for the whole pipeline is a held-out score
as well, except for C, which is chosen once on all seven (it barely matters, see the log loss
lines); the setting written out is the one that does best on all seven. The organisers' baseline
(nearest player, majority over +/- 2 s) is scored on the same frames, at label tolerances of 0.5,
1 and 2 s. Each grid named by --grids makes a variant of its own: 'base', the grid of M9 to M11e,
and 'wide', which holds every base setting and reaches further up in dead_bias ('+widebias'; M11e
chose 1.0, the top of the base grid).

It then prints where the held-out pipeline goes wrong (possession.diagnose): its calls and
errors by ball state, and the cause of every possession it misses. With --oracle it also decodes
the held-out evidence with its team call, its dead-ball call or both turned to the truth, at the
model's own confidence, which bounds what fixing each part could gain.

The speed check (M11f, --speed-factors) chooses nothing. The match's tracking reports slower
players than the practice ground in every ball state (the ball-state table of
scripts/make_submission.py), and the models read player speed. So the held-out matches are scored
again with every player speed scaled by each factor, each match decoded with its own nested
setting, and the score and the share of frames called dead are printed per factor.

Last, the models are refitted on all seven matches and written, with the chosen decoder, to
models/possession_<variant>.json.

All of this runs for every design of the features named (possession.models.DESIGNS) on the same
degraded halves. Each variant is then set against the reference (--reference; the adopted
v1+trees-wide+deadtrees by default) match by match, and the adoption rule fixed before the run is
applied: a variant replaces the reference only if its held-out spellF1 at +/-1 s is at least 0.01
higher, it is better on at least 5 of the 7 matches, it loses nothing at +/-0.5 s or +/-2 s, and
its macroF1 falls by at most 0.005. If several pass, the one with the highest held-out spellF1 at
+/-1 s is adopted.

Public data only; reads no match data. A job for Kaggle (kaggle/README.md), not for the team's
laptop.

    python scripts/train_models.py [--designs v1 [v2]] [--dead trees [logistic]]
                                   [--team trees-wide [trees logistic]] [--grids base [wide]]
                                   [--reference v1+trees-wide+deadtrees]
                                   [--speed-factors [0.9 0.8]] [--seeds 2] [--step 8]
                                   [--out-dir models] [--oracle]
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
from possession.features import FPS, HOME, NAMES, compute, swapped  # noqa: E402
from possession.models import (DESIGNS, design, design_names, fit, fit_trees,  # noqa: E402
                               log_state_probabilities, probability, save)
from possession.predictors import NOBODY, nearest_team, smooth  # noqa: E402

pm = practice.pm
PITCH = (105.0, 68.0)
GRID = (0.01, 0.1, 1.0)
DEAD_KINDS = ("logistic", "trees")
TEAM_KINDS = ("logistic", "trees", "trees-wide")
# Gradient-boosted settings per (model, kind), chosen among by held-out log loss like C.
# ('team', 'trees') is the setting M11d chose from two (runs/20261109T0931Z-train_models) and
# adopted; M11e adopted the larger team model with trees for the dead-ball model
# (runs/20261109T1004Z-train_models).
TREES = {("team", "trees"): ({"max_iter": 300, "max_leaf_nodes": 63, "learning_rate": 0.05,
                              "min_samples_leaf": 100},),
         ("team", "trees-wide"): ({"max_iter": 600, "max_leaf_nodes": 127, "learning_rate": 0.05,
                                   "min_samples_leaf": 100},),
         ("dead", "trees"): ({"max_iter": 300, "max_leaf_nodes": 63, "learning_rate": 0.05,
                              "min_samples_leaf": 100},)}
EVAL_SEED = 99
TOLERANCES = (0.5, 1.0, 2.0)
BASELINE_WINDOW = 50
# The decoder's costs in nats of evidence, and the bias on the D evidence per frame.
DECODER_AXES = {"team_switch": (8, 16, 32, 64, 128, 256, 512),
                "dead_switch": (8, 16, 32, 64, 128, 256, 512),
                "dead_bias": (-1.0, -0.5, 0.0, 0.5, 1.0)}
DECODER_GRID = tuple(itertools.product(*DECODER_AXES.values()))
GRIDS = ("base", "wide")
# M11e chose dead_bias 1.0, the top of the base grid, in six folds of seven
# (runs/20261109T1004Z-train_models). The wide grid adds these above it (M11f).
WIDE_DEAD_BIAS = (1.5, 2.0, 2.5, 3.0)
# The speed check (M11f) scales the players' speeds of the held-out matches by these factors.
SPEED_FACTORS = (0.9, 0.8)
PLAYER_SPEEDS = ("speed_all", "speed_home", "speed_away")


def decoder_axes(grid="base"):
    """The decoder's axes for one of GRIDS: 'base' is DECODER_AXES; 'wide' extends dead_bias
    upward by WIDE_DEAD_BIAS and keeps every base setting, in the same axis order."""
    if grid == "base":
        return DECODER_AXES
    return dict(DECODER_AXES, dead_bias=tuple(DECODER_AXES["dead_bias"]) + tuple(WIDE_DEAD_BIAS))


def slowed(features, factor):
    """A copy of possession.features.compute's matrix with the players' speeds (PLAYER_SPEEDS)
    scaled by factor, as a tracker that reports slower players would give it."""
    out = np.array(features, copy=True)
    for name in PLAYER_SPEEDS:
        out[:, NAMES.index(name)] *= factor
    return out


def build(match_ids, seeds, step, designs=("v1",), speed_factors=()):
    """Training rows per design and match, and the held-out halves per match for evaluation,
    with one feature matrix per design under 'X' and, for the speed check, one per design and
    speed factor under 'slow'. The features are computed once per half and only the design
    differs, so every design sees exactly the same frames."""
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
        part = held_out.setdefault(case["match"], {
            "X": {v: [] for v in designs},
            "slow": {v: {f: [] for f in speed_factors} for v in designs},
            "labels": [], "nearest": [], "ball_known": [], "lengths": []})
        for version in designs:
            part["X"][version].append(design(features, version).astype(np.float32))
            for factor in speed_factors:
                part["slow"][version][factor].append(
                    design(slowed(features, factor), version).astype(np.float32))
        part["labels"].append(case["labels"])
        part["nearest"].append(nearest_team(case["half"]))
        part["ball_known"].append(features[:, 0] + features[:, 1] > 0)  # seen or bridged
        part["lengths"].append(len(features))
    for part in held_out.values():
        for key in ("labels", "nearest", "ball_known"):
            part[key] = np.concatenate(part[key])
        part["X"] = {version: np.concatenate(X) for version, X in part["X"].items()}
        part["slow"] = {version: {factor: np.concatenate(X) for factor, X in by_factor.items()}
                        for version, by_factor in part["slow"].items()}
    return train, held_out


def split(part, array):
    """An array over a held-out match's frames, split back into its halves."""
    return np.split(array, np.cumsum(part["lengths"])[:-1])


def halves_of(part, key):
    """A held-out match's array split back into its halves."""
    return split(part, part[key])


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


def _share_right(p, y):
    return float(np.mean((p > 0.5) == y)) if len(y) else float("nan")


def report_team(label, p, y, rule):
    has_nearest = rule["nearest"] != NOBODY
    print("team {:<34} logloss {:.4f} AUC {:.4f} acc {:.4f}; where a nearest player exists ours "
          "{:.4f} vs nearest {:.4f}; without one ours {:.4f} ({:.0%} of live frames)".format(
              label, log_loss(y, p), roc_auc_score(y, p), _share_right(p, y),
              _share_right(p[has_nearest], y[has_nearest]),
              np.mean((rule["nearest"][has_nearest] == HOME) == y[has_nearest]),
              _share_right(p[~has_nearest], y[~has_nearest]), np.mean(~has_nearest)))
    return log_loss(y, p)


def candidates(model, kind, version):
    """[(label, fitter)]: the fits of one kind for model ('dead' or 'team'), chosen among by
    held-out log loss."""
    if kind == "logistic":
        return [("logistic C={:g}".format(C),
                 lambda X, y, C=C: fit(X, y, C=C, version=version)) for C in GRID]
    return [("{} {max_iter}x{max_leaf_nodes} lr {learning_rate:g} leaf>={min_samples_leaf}"
             .format(kind, **params), lambda X, y, params=params: fit_trees(
                 X, y, version=version, **params))
            for params in TREES[(model, kind)]]


def choose(model, kind, train, held_out, matches, version, rule):
    """(label, held-out folds, final fit on all matches) of the best candidate of one kind."""
    report = report_dead if model == "dead" else report_team
    results = {}
    for label, fitter in candidates(model, kind, version):
        folds, p, y = lomo(train, held_out, matches, model, fitter)
        results[label] = (report(label, p, y, rule), folds, fitter)
    chosen = min(results, key=lambda label: results[label][0])
    return chosen, results[chosen][1], results[chosen][2](*stack(train, matches, model))


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


def fit_decoder(held_out, oof, matches, axes=None):
    """Choose the decoder setting on the grid of axes (DECODER_AXES unless given), nested by
    match, and report it.

    Returns (setting, practice range, held-out codes per match, summary), where summary holds
    the held-out (spellF1, macroF1) per label tolerance, the held-out spellF1 per match, and
    under 'nested' each match's setting chosen on the other six, as a (team_switch, dead_switch,
    dead_bias) tuple.
    """
    axes = DECODER_AXES if axes is None else axes
    grid = tuple(itertools.product(*axes.values()))
    start = time.time()
    table = [dict() for _ in grid]
    for match in matches:
        gt = practice.joined([practice.codes(l) for l in halves_of(held_out[match], "labels")],
                             gap=pm.X)
        for k, pred in enumerate(decoded(oof[match], grid)):
            table[k][match] = practice.counts(gt, pred)
        print("  decoder grid on {}: {:.0f} s so far".format(match, time.time() - start))
    chosen = {m: best(table, [o for o in matches if o != m]) for m in matches}
    final = best(table, matches)

    rows = collections.defaultdict(list)
    rates = collections.defaultdict(list)
    held_out_codes = {}
    for match in matches:
        nested, overall = decoded(oof[match], [grid[chosen[match]], grid[final]])
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
          "on all seven ({:.0f} s)".format(len(grid), ", ".join(axes), time.time() - start))
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
    picks = collections.Counter(grid[k] for k in chosen.values())
    print("settings chosen per match: " + "; ".join(
        "{} x{}".format(dict(zip(axes, setting)), n) for setting, n in picks.most_common()))
    setting = dict(zip(axes, grid[final]))
    edges = ["{} {}".format(name, "upper" if value == axes[name][-1] else "lower")
             for name, value in setting.items()
             if value in (axes[name][0], axes[name][-1])]
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
               "per_match": per_match, "nested": {m: grid[chosen[m]] for m in matches}}
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


def oracle_headroom(held_out, oof, matches, axes=None):
    """How far the pipeline is from what its decoder could do with right evidence (M11c).

    For each variant of oracle_evidence, the best pooled spellF1 over the decoder grid of axes
    (DECODER_AXES unless given) on all seven matches, labels at +/-1 s. It is an upper bound
    chosen in sample, not a held-out score: the gap between rows says which part, team or dead
    ball, holds the remaining score.
    """
    axes = DECODER_AXES if axes is None else axes
    grid = tuple(itertools.product(*axes.values()))
    start = time.time()
    print("\nperfect-evidence check (M11c), best setting on all seven, labels at +/-1 s; the "
          "evidence keeps the model's confidence and only its direction is corrected")
    truths = {m: [practice.codes(l) for l in halves_of(held_out[m], "labels")] for m in matches}
    gts = {m: practice.joined(truths[m], gap=pm.X) for m in matches}
    for variant in ORACLES:
        table = [dict() for _ in grid]
        for match in matches:
            evidence = [oracle_evidence(log_p, truth, variant)
                        for log_p, truth in zip(oof[match], truths[match])]
            for k, pred in enumerate(decoded(evidence, grid)):
                table[k][match] = practice.counts(gts[match], pred)
        k = best(table, matches)
        spell, macro = practice.pooled([table[k][m] for m in matches])
        print("  {:<10} spellF1 {:.4f}  mF1 {:.4f}  at {} ({:.0f} s)".format(
            variant, spell, macro, dict(zip(axes, grid[k])), time.time() - start))


def speed_check(held_out, evidence, settings, matches):
    """How the held-out pipeline moves when the players' speeds are scaled (M11f). It chooses
    nothing.

    evidence: {factor: {match: log-probabilities per half}}, the held-out folds' evidence with
    every player speed scaled by factor (1.0: as degraded); settings: {match: the decoder setting
    chosen on the other six}. Returns {factor: (spellF1, macroF1, share of frames called D, that
    share where the ball is seen)}, pooled over matches with labels at +/-1 s, and prints it
    beside the true shares. At 1.0 it reproduces the nested held-out score.
    """
    gts, graded, seen = {}, {}, {}
    for match in matches:
        part = held_out[match]
        gts[match] = practice.joined([practice.codes(l) for l in halves_of(part, "labels")],
                                     gap=pm.X)
        graded[match] = gts[match] != pm.X
        state = practice.joined([diagnose.ball_states(X) for X in halves_of(part, "X")], gap=-1)
        seen[match] = graded[match] & (state == diagnose.STATES.index("seen"))
    true_dead = np.mean(np.concatenate([gts[m][graded[m]] == pm.D for m in matches]))
    true_seen = np.mean(np.concatenate([gts[m][seen[m]] == pm.D for m in matches]))
    print("\nspeed check (M11f), which chooses nothing: the held-out matches with every player "
          "speed scaled, each decoded with its setting chosen on the other six, labels at +/-1 s")
    out = {}
    for factor in sorted(evidence, reverse=True):
        parts, dead, dead_seen = [], [], []
        for match in matches:
            pred = next(decoded(evidence[factor][match], [settings[match]]))
            parts.append(practice.counts(gts[match], pred))
            dead.append(pred[graded[match]] == pm.D)
            dead_seen.append(pred[seen[match]] == pm.D)
        spell, macro = practice.pooled(parts)
        out[factor] = (spell, macro, float(np.mean(np.concatenate(dead))),
                       float(np.mean(np.concatenate(dead_seen))))
        print("  speed x{:.1f}  spellF1 {:.4f}  mF1 {:.4f}  called D {:.1%} (true {:.1%}), where "
              "the ball is seen {:.1%} (true {:.1%})".format(
                  factor, spell, macro, out[factor][2], true_dead, out[factor][3], true_seen))
    return out


def variant_name(version, dead_kind, team_kind, grid="base"):
    """'v1' with logistic models; '+trees' or '+trees-wide' for a tree team model, '+deadtrees'
    for a tree dead-ball model, '+widebias' for the wide decoder grid: 'v1+trees-wide+deadtrees'
    is the one adopted in M11e."""
    name = version
    if team_kind != "logistic":
        name += "+" + team_kind
    if dead_kind != "logistic":
        name += "+dead" + dead_kind
    if grid != "base":
        name += "+" + grid + "bias"
    return name


def fit_design(version, dead_kinds, team_kinds, train, held_out, matches, args, start):
    """One design: each kind of dead-ball and of team model chosen and fitted once, then for every
    pair of them and every decoder grid (args.grids) a decoder, an error report, the speed check
    (args.speed_factors, none to skip it) and a file models/possession_<variant>.json.

    held_out carries this design's feature matrix under 'X', and under 'slow' one per speed
    factor. Returns {variant: fit_decoder's summary, with the speed check's rows under 'speed'
    when it ran}.
    """
    print("\n===== design {}: {} columns =====".format(version, len(design_names(version))))
    rule = rules(held_out, matches)
    chosen = {}
    for model, kinds in (("dead", dead_kinds), ("team", team_kinds)):
        for kind in kinds:
            chosen[(model, kind)] = choose(model, kind, train, held_out, matches, version, rule)
            print("chosen {} {} ({:.0f} s)".format(model, chosen[(model, kind)][0],
                                                   time.time() - start))

    summaries = {}
    for dead_kind in dead_kinds:
        for team_kind in team_kinds:
            dead_label, dead_folds, dead_model = chosen[("dead", dead_kind)]
            team_label, team_folds, team_model = chosen[("team", team_kind)]
            folds = {m: {"dead": dead_folds[m], "team": team_folds[m]} for m in matches}
            # The held-out evidence, as degraded (1.0) and with the players' speeds scaled.
            evidence = {1.0: {m: [log_state_probabilities(folds[m], X)
                                  for X in halves_of(held_out[m], "X")] for m in matches}}
            for factor in args.speed_factors:
                evidence[factor] = {m: [log_state_probabilities(folds[m], X)
                                        for X in split(held_out[m], held_out[m]["slow"][factor])]
                                    for m in matches}
            oof = evidence[1.0]
            print("\nheld-out evidence of dead {}, team {} at speed x{} ({:.0f} s so far)".format(
                dead_label, team_label, ", x".join("{:g}".format(f) for f in evidence),
                time.time() - start))

            for grid in args.grids:
                variant = variant_name(version, dead_kind, team_kind, grid)
                axes = decoder_axes(grid)
                print("\n----- {}: dead {}, team {}, decoder grid {} ({:.0f} s) -----".format(
                    variant, dead_label, team_label, grid, time.time() - start))
                decoder, practice_range, held_out_codes, summary = fit_decoder(
                    held_out, oof, matches, axes)
                report_errors(held_out, held_out_codes, matches)
                if args.oracle:
                    oracle_headroom(held_out, oof, matches, axes)
                if args.speed_factors:
                    summary["speed"] = speed_check(held_out, evidence, summary["nested"], matches)

                models = {"design": version, "dead": dead_model, "team": team_model,
                          "decoder": decoder,
                          "fitted_on": {"matches": matches, "seeds": args.seeds,
                                        "step": args.step, "label_tolerance_s": 1.0,
                                        "pitch": list(PITCH), "scikit_learn": sklearn_version,
                                        "dead_model": dead_label, "team_model": team_label,
                                        "decoder_grid": {k: list(v) for k, v in axes.items()},
                                        "practice_range": practice_range}}
                out = os.path.join(args.out_dir, "possession_{}.json".format(variant))
                os.makedirs(args.out_dir, exist_ok=True)
                save(models, out)
                print("wrote {} ({:.0f} s so far)".format(os.path.relpath(out, ROOT),
                                                          time.time() - start))
                summaries[variant] = summary
    return summaries


def compare(summaries, matches, reference_name="v1"):
    """Each variant against the reference, match by match, with the adoption rule fixed before
    the run. If several variants pass, the one with the highest held-out spellF1 at +/-1 s is
    the one to adopt. Returns its name, or None."""
    reference = summaries[reference_name]
    passing = []
    for version, summary in summaries.items():
        if version == reference_name:
            continue
        gains = {m: summary["per_match"][m] - reference["per_match"][m] for m in matches}
        better = sum(gain > 0 for gain in gains.values())
        spell = {t: summary["pooled"][t][0] - reference["pooled"][t][0] for t in TOLERANCES}
        macro = summary["pooled"][1.0][1] - reference["pooled"][1.0][1]
        adopt = (spell[1.0] >= 0.01 and better >= 5 and spell[0.5] >= 0 and spell[2.0] >= 0
                 and macro >= -0.005)
        if adopt:
            passing.append(version)
        print("\n{} against {}, held out: spellF1 {:+.4f} at +/-1 s ({:+.4f} at 0.5 s, {:+.4f} "
              "at 2 s), macroF1 {:+.4f}; better on {} of {} matches".format(
                  version, reference_name, spell[1.0], spell[0.5], spell[2.0], macro, better,
                  len(matches)))
        print("  per match: " + ", ".join("{} {:+.3f}".format(m, g) for m, g in gains.items()))
        print("  adoption rule (spellF1 +0.01 or more at 1 s, better on 5 or more matches, no "
              "loss at 0.5 s or 2 s, macroF1 down 0.005 at most): {}".format(
                  "ADOPT " + version if adopt else "keep " + reference_name))
    winner = max(passing, key=lambda v: summaries[v]["pooled"][1.0][0]) if passing else None
    print("\nverdict: {}".format("adopt " + winner if winner else "keep " + reference_name))
    return winner


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # v2 lost to v1 on 2026-11-09 (runs/20261109T0833Z-train_models); run --designs v1 v2 to see it.
    parser.add_argument("--designs", nargs="+", default=["v1"], choices=sorted(DESIGNS))
    # M11e adopted trees for both models (v1+trees-wide+deadtrees,
    # runs/20261109T1004Z-train_models) with dead_bias at the top of the base grid; M11f sets the
    # wide grid against it. Run --dead logistic trees --team trees trees-wide --grids base
    # --speed-factors to repeat M11e.
    parser.add_argument("--dead", nargs="+", default=["trees"], choices=DEAD_KINDS)
    parser.add_argument("--team", nargs="+", default=["trees-wide"], choices=TEAM_KINDS)
    parser.add_argument("--grids", nargs="+", default=list(GRIDS), choices=GRIDS,
                        help="decoder grids to search, each a variant of its own")
    parser.add_argument("--reference", default="v1+trees-wide+deadtrees",
                        help="the variant the others are set against")
    parser.add_argument("--speed-factors", nargs="*", type=float, default=list(SPEED_FACTORS),
                        help="factors the speed check scales the players' speeds by; give none "
                             "to skip it")
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
    train, held_out = build(matches, args.seeds, args.step, args.designs, args.speed_factors)
    first = args.designs[0]
    print("built rows in {:.0f} s: dead {}, team {}".format(
        time.time() - start, len(stack(train[first], matches, "dead")[1]),
        len(stack(train[first], matches, "team")[1])))

    summaries = {}
    for version in args.designs:
        view = {m: dict(part, X=part["X"][version], slow=part["slow"][version])
                for m, part in held_out.items()}
        summaries.update(fit_design(version, args.dead, args.team, train[version], view,
                                    matches, args, start))
    if args.reference in summaries and len(summaries) > 1:
        compare(summaries, matches, args.reference)
    else:
        print("\nno comparison: the reference {} was not fitted in this run".format(
            args.reference))
    print("\ndone in {:.0f} s".format(time.time() - start))
    return 0


if __name__ == "__main__":
    sys.exit(main())
