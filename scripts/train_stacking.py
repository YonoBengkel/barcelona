"""M11l: a stage two of trees on the first stage's log-odds (possession.stacking), against the
frozen pipeline.

Two pipelines are fitted and scored on the practice ground, each match held out in turn exactly
as scripts/train_models.py does it, with the same decoder search, nested by match:

- the reference, the frozen pipeline: tree models for both calls (v1+trees-wide+deadtrees), fitted
  here again, so its held-out scores reproduce the frozen model's;
- 'v1+trees-wide+deadtrees+stack': the same first stage, its log-odds corrected by a stage two of
  trees that reads them around each frame (possession.stacking, settings fixed there before the
  run).

Stage two is nested inside that loop, so no label of a held-out match reaches a model that scores
that match. For held-out match k, stage two learns from the six other matches' training copies
degraded with seed 0 (every half once), each match j among them scored by a first stage fitted on
the five matches other than j and k: there, the first stage's log-odds are those of a match it
never saw, as they will be on the match. A first stage fitted without j and k serves both their
folds, so seven matches need 21 such fits of each call, not 42. Stage two is then applied to k's
held-out log-odds from the reference's own fold, fitted without k.

The adoption rule is train_models.compare's, set before the run: held-out spellF1 at +/-1 s at
least 0.01 higher, better on at least 5 of the 7 matches, no loss at +/-0.5 s or +/-2 s, macroF1
down by at most 0.005. It is applied on the held-out halves degraded with EVAL_SEED and again,
with the same fitted folds and the decoder chosen the same nested way, on the held-out halves
degraded with a second seed (--confirm-seed) that no other copy uses. The candidate is adopted
only if it passes both times. The pooled log loss of both calls, before and after stage two, is
printed for insight and chooses nothing.

Last, stage two is fitted on all seven matches, each scored by the reference's fold fitted
without it, and written with the first stage fitted on all seven and the decoder setting chosen on
all seven at EVAL_SEED to models/possession_v1+trees-wide+deadtrees+stack.json, whatever the
verdict. The file is committed only if adopted.

The run of 2026-11-12 (runs/20261112T1009Z-train_stacking) kept the trees: stage two improved both
calls frame by frame but held-out spellF1 at +/-1 s by only +0.0015, and +0.0049 at the second
seed, with a loss at +/-2 s both times. Its model file is not committed.

Public data only; reads no match data. A job for Kaggle on CPU (kaggle/README.md), not for the
team's laptop.

    python scripts/train_stacking.py [--seeds 2] [--step 8] [--confirm-seed 98] [--out-dir models]
"""

import argparse
import os
import sys
import time
from dataclasses import asdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_models as tm  # noqa: E402  (puts the repository on the path)

from possession import external as ex  # noqa: E402
from possession import stacking  # noqa: E402
from possession.models import log_sigmoid, save, score  # noqa: E402
from possession.simulate import DegradeParams  # noqa: E402


def training_evidence(models, halves):
    """[(dead, home, labels)] as stacking.rows takes them: the first stage's log-odds on every
    training copy degraded with seed 0 among halves (build()'s sequences entries of one match).
    One seed gives stage two every half once; the other seeds' copies are the same halves again."""
    return [(score(models["dead"], half["X"]), score(models["team"], half["X"]), half["labels"])
            for half in halves if half["seed"] == 0]


def log_probabilities(dead, home):
    """(frames, 3) log P(H), P(A), P(D) from the two log-odds, exactly as
    possession.models.log_state_probabilities forms them."""
    alive = log_sigmoid(-dead)
    return np.column_stack([alive + log_sigmoid(home), alive + log_sigmoid(-home),
                            log_sigmoid(dead)])


def report_frames(label, parts, held_out, matches, rule):
    """Both calls' pooled held-out log loss, AUC and accuracy, as train_models reports them, from
    (dead, home) log-odds per half of every match; the team call on live frames only."""
    dead = np.concatenate([d for m in matches for d, _ in parts[m]])
    home = np.concatenate([h for m in matches for _, h in parts[m]])
    labels = np.concatenate([held_out[m]["labels"] for m in matches])
    live = labels != "D"
    tm.report_dead(label, np.exp(log_sigmoid(dead)), labels == "D", rule)
    tm.report_team(label, np.exp(log_sigmoid(home[live])), labels[live] == "H", rule)


def run(train, sequences, held_out, confirm, matches, out_dir, start, seeds=2, step=8,
        confirm_seed=98):
    """The whole comparison on built data; see the module docstring.

    train: build()'s rows of design v1; sequences: its whole training copies; held_out and
    confirm: the held-out halves degraded with EVAL_SEED and with confirm_seed, design v1 under
    'X'. seeds and step are recorded in the model file. Returns {'verdict': the candidate's name
    if adopted, else None, 'summaries': {seed: {name: fit_decoder's summary}}}.
    """
    rule = tm.rules(held_out, matches)
    dead_label, dead_folds, dead_final = tm.choose("dead", "trees", train, held_out, matches,
                                                   "v1", rule)
    print("chosen dead {} ({:.0f} s)".format(dead_label, time.time() - start))
    team_label, team_folds, team_final = tm.choose("team", "trees-wide", train, held_out,
                                                   matches, "v1", rule)
    print("chosen team {} ({:.0f} s)".format(team_label, time.time() - start))
    # The inner folds refit the very settings the reference chose, with the harness's fitters.
    fitters = {"dead": dict(tm.candidates("dead", "trees", "v1"))[dead_label],
               "team": dict(tm.candidates("team", "trees-wide", "v1"))[team_label]}
    folds = {m: {"dead": dead_folds[m], "team": team_folds[m]} for m in matches}
    reference = tm.variant_name("v1", "trees", "trees-wide")
    candidate = reference + "+stack"
    grounds = ((tm.EVAL_SEED, held_out), (confirm_seed, confirm))

    stage_one = {seed: {m: [(score(folds[m]["dead"], X), score(folds[m]["team"], X))
                            for X in tm.halves_of(part[m], "X")] for m in matches}
                 for seed, part in grounds}
    print("the first stage's held-out log-odds at seeds {} ({:.0f} s so far)".format(
        " and ".join(str(seed) for seed, _ in grounds), time.time() - start))

    stage_two = {seed: {} for seed, _ in grounds}
    inner, pairs = {k: {} for k in matches}, 0
    for k in matches:
        began = time.time()
        for j in matches:
            if j == k or j in inner[k]:
                continue
            # Fitted without j and k: it scores j for k's stage two and k for j's.
            others = [m for m in matches if m not in (j, k)]
            fitted = {call: fitter(*tm.stack(train, others, call))
                      for call, fitter in fitters.items()}
            inner[k][j] = training_evidence(fitted, sequences[j])
            inner[j][k] = training_evidence(fitted, sequences[k])
            pairs += 1
        own = inner.pop(k)
        halves = [half for j in matches if j != k for half in own[j]]
        stack = stacking.fit(halves)
        for seed, _ in grounds:
            stage_two[seed][k] = [stacking.apply(stack, dead, home)
                                  for dead, home in stage_one[seed][k]]
        print("stage two held out on {}: fitted on {} training halves of the other matches; {} of "
              "{} first stages fitted without a pair of matches ({:.0f} s; {:.0f} s so far)".format(
                  k, len(halves), pairs, len(matches) * (len(matches) - 1) // 2,
                  time.time() - began, time.time() - start))

    for seed, part in grounds:
        print("\nheld-out frame calls, halves degraded with seed {}".format(seed))
        here = tm.rules(part, matches)
        report_frames("stage one", stage_one[seed], part, matches, here)
        report_frames("stage two", stage_two[seed], part, matches, here)

    decoder, summaries = None, {}
    for seed, part in grounds:
        summaries[seed] = {}
        for name, parts in ((reference, stage_one[seed]), (candidate, stage_two[seed])):
            oof = {m: [log_probabilities(dead, home) for dead, home in parts[m]] for m in matches}
            print("\n----- {}, held-out halves degraded with seed {} ({:.0f} s so far) -----"
                  .format(name, seed, time.time() - start))
            setting, practice_range, codes, summary = tm.fit_decoder(part, oof, matches)
            if seed == tm.EVAL_SEED:
                tm.report_errors(part, codes, matches)
                if name == candidate:
                    decoder = (setting, practice_range)
            summaries[seed][name] = summary

    winners = {}
    for seed, _ in grounds:
        print("\n===== the adoption rule on the held-out halves of seed {} =====".format(seed))
        winners[seed] = tm.compare({reference: summaries[seed][reference],
                                    candidate: summaries[seed][candidate]}, matches, reference)
    adopted = all(winners[seed] == candidate for seed, _ in grounds)
    print("\nM11l stacking verdict: {} ({})".format(
        "adopt " + candidate if adopted else "keep " + reference,
        "; ".join("seed {}: spellF1 {:.4f} against {:.4f}, {}".format(
            seed, summaries[seed][candidate]["pooled"][1.0][0],
            summaries[seed][reference]["pooled"][1.0][0],
            "passes" if winners[seed] == candidate else "fails") for seed, _ in grounds)))

    # Stage two on all seven: every match's training copies scored by the fold fitted without it.
    final = stacking.fit([half for m in matches
                          for half in training_evidence(folds[m], sequences[m])])
    setting, practice_range = decoder
    models = {"design": "v1", "dead": dead_final, "team": team_final, "stack": final,
              "decoder": setting,
              "fitted_on": {"matches": matches, "seeds": seeds, "step": step,
                            "label_tolerance_s": 1.0, "pitch": list(tm.PITCH),
                            "scikit_learn": tm.sklearn_version, "dead_model": dead_label,
                            "team_model": team_label,
                            "decoder_grid": {name: list(values)
                                             for name, values in tm.DECODER_AXES.items()},
                            "practice_range": practice_range,
                            "simulator": asdict(DegradeParams()), "stack": final["settings"],
                            "confirm_seed": confirm_seed}}
    out = os.path.join(out_dir, "possession_{}.json".format(candidate))
    os.makedirs(out_dir, exist_ok=True)
    save(models, out)
    print("wrote {} ({:.0f} s so far)".format(os.path.relpath(out, tm.ROOT),
                                              time.time() - start))
    return {"verdict": candidate if adopted else None, "summaries": summaries}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--step", type=int, default=8)
    parser.add_argument("--confirm-seed", type=int, default=98)
    parser.add_argument("--out-dir", default=os.path.join(tm.ROOT, "models"))
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be 1 or more: stage two learns from the copies of seed 0")
    if args.confirm_seed == tm.EVAL_SEED or 0 <= args.confirm_seed < args.seeds:
        parser.error("--confirm-seed must differ from EVAL_SEED and from every training seed")
    matches = sorted(ex.MATCHES)

    start = time.time()
    sequences = {}
    train, held_out = tm.build(matches, args.seeds, args.step, ["v1"], (), None,
                               sequences=sequences)
    _, confirm = tm.build(matches, 0, args.step, ["v1"], (), None, eval_seed=args.confirm_seed)
    print("built rows and whole halves in {:.0f} s: dead {}, team {}, {} training halves".format(
        time.time() - start, len(tm.stack(train["v1"], matches, "dead")[1]),
        len(tm.stack(train["v1"], matches, "team")[1]),
        sum(len(halves) for halves in sequences.values())))
    view = {m: dict(part, X=part["X"]["v1"], slow=part["slow"]["v1"])
            for m, part in held_out.items()}
    confirm_view = {m: dict(part, X=part["X"]["v1"], slow=part["slow"]["v1"])
                    for m, part in confirm.items()}
    run(train["v1"], sequences, view, confirm_view, matches, args.out_dir, start, args.seeds,
        args.step, args.confirm_seed)
    print("\ndone in {:.0f} s".format(time.time() - start))
    return 0


if __name__ == "__main__":
    sys.exit(main())
