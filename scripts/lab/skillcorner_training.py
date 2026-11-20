"""Lab: what training on SkillCorner itself, or on SkillCorner and IDSSE together, wins there.

scripts/lab/cross_provider.py scores the submitted pipeline, fitted on IDSSE alone, on SkillCorner's
20 A-League matches. This job fits the same network (possession.tcn, train_tcn.TRAINING) in five
folds of four held-out SkillCorner matches each, and compares three networks on those matches:

- 'idsse': the submitted network itself (models/possession_v1+tcn+copies8.json), fitted on IDSSE;
- 'idsse+gaps': a diagnostic, not a model: the same evidence with every frame that carries no
  tracking at all (a replay, a close-up) set to a sure D. SkillCorner labels nearly all of those
  frames D, while the submitted network, which never saw a stretch without players, carries the
  possession on through them; this measures what that alone costs;
- 'skillcorner': fitted on the other 16 SkillCorner matches;
- 'both': fitted on those 16 and the seven IDSSE matches, two degraded copies of every half, as
  the practice ground builds them (scripts/train_models.py). The two sources label possession
  their own way (IDSSE: the competition's controlled possession; SkillCorner: its own), so the
  network learns from a blend.

Every SkillCorner half is read as given (extrapolated positions kept) and seen as it is and with
home and away exchanged. Each network's decoder is chosen for each match on the other 19 matches'
held-out evidence (the base grid, scored on every frame), then that pipeline is scored with the
organisers' metric on every frame and on tracked frames only: spellF1 pooled over the matches with
a 95% interval from resampling them, macroF1 beside it, and possessions per ten minutes against
SkillCorner's own.

Public data only. A job for Kaggle with a GPU; kaggle/run.py installs PyTorch for it.

    python scripts/lab/skillcorner_training.py [--folds 5] [--copies 2] [--workers N]
"""

import argparse
import itertools
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cross_provider as cp  # noqa: E402  (puts scripts/ and the repository on the path)
import train_models as tm  # noqa: E402
import train_tcn as tt  # noqa: E402
import train_tcn_ensemble as te  # noqa: E402

from possession import external as ex  # noqa: E402
from possession import practice  # noqa: E402
from possession import skillcorner as sc  # noqa: E402
from possession.features import compute, swapped  # noqa: E402
from possession.models import design, design_names, load  # noqa: E402

NETWORKS = ("idsse", "idsse+gaps", "skillcorner", "both")
VERSION = "v1"
CHOOSE_ON = "every frame"
# Log-probabilities of H, A and D on a frame without tracking, for 'idsse+gaps': a sure D.
GAP_EVIDENCE = (-30.0, -30.0, 0.0)


def gaps_dead(log_p, tracked):
    """log_p with the evidence of every untracked frame set to GAP_EVIDENCE."""
    out = np.array(log_p, dtype=np.float64, copy=True)
    out[~np.asarray(tracked, dtype=bool)] = GAP_EVIDENCE
    return out


def match_halves(match):
    """(match, [per half: 'X', 'mirrored', 'labels', 'tracked']) for one SkillCorner match, as
    given, in the form train_tcn.fit_network takes its training halves."""
    info, periods = sc.load_match(match)
    out = []
    for period in periods:
        half = period["half"]
        out.append({"X": design(compute(half, info["pitch"])[0], VERSION).astype(np.float32),
                    "mirrored": design(compute(swapped(half), info["pitch"])[0],
                                       VERSION).astype(np.float32),
                    "labels": period["labels"], "tracked": period["tracked"]})
    return match, out


def folds_of(matches, count):
    """count folds of held-out matches, every count-th match from each start in turn."""
    return [list(matches[i::count]) for i in range(count)]


def grid_job(halves, log_ps, settings):
    """cp.grid_counts' job for one match: its evidence, its truth for CHOOSE_ON, the grid."""
    return (log_ps, {CHOOSE_ON: cp.truth(halves, CHOOSE_ON)}, settings)


def run(matches, folds, copies, step, device, workers, start):
    """Fit and score every network; returns {(network, scoring): result}."""
    settings = tuple(itertools.product(*tm.decoder_axes("base").values()))
    names = design_names(VERSION)
    pool = te.worker_pool(workers)
    mapped = pool.map if pool else map
    submit = (lambda job: pool.submit(cp.grid_counts, job)) if pool else cp.grid_counts
    try:
        halves = dict(mapped(match_halves, matches))
        print("SkillCorner: {} matches, {} halves built ({:.0f} s so far)".format(
            len(halves), sum(len(h) for h in halves.values()), time.time() - start))
        sequences = {}
        tm.build(sorted(ex.MATCHES), copies, step, [VERSION], (), None, sequences=sequences)
        idsse = [half for m in sorted(sequences) for half in sequences[m]]
        print("IDSSE: {} training halves, {} copies ({:.0f} s so far)".format(
            len(idsse), copies, time.time() - start))

        evidence = {name: {} for name in NETWORKS}
        pending = {name: {} for name in NETWORKS}
        submitted = load(cp.MODELS)
        for m in matches:
            evidence["idsse"][m] = [cp.evidence(submitted, h["X"]) for h in halves[m]]
            evidence["idsse+gaps"][m] = [gaps_dead(log_p, h["tracked"])
                                         for log_p, h in zip(evidence["idsse"][m], halves[m])]
            for name in ("idsse", "idsse+gaps"):
                pending[name][m] = submit(grid_job(halves[m], evidence[name][m], settings))
        for number, fold in enumerate(folds):
            train = [h for m in matches if m not in fold for h in halves[m]]
            for name in ("skillcorner", "both"):
                model = tt.fit_network(train + (idsse if name == "both" else []),
                                       dict(tt.TRAINING), device, names)
                for m in fold:
                    evidence[name][m] = [tt.log_probabilities(*cp.fast_scores(model, h["X"]))
                                         for h in halves[m]]
                    pending[name][m] = submit(grid_job(halves[m], evidence[name][m], settings))
            print("fold {} of {} (held out {}) fitted ({:.0f} s so far)".format(
                number + 1, len(folds), ", ".join(fold), time.time() - start))
        tables = {name: {m: item.result() if pool else item for m, item in by_match.items()}
                  for name, by_match in pending.items()}
    finally:
        if pool is not None:
            pool.shutdown()
    print("decoder grids done ({:.0f} s so far)".format(time.time() - start))

    rng = np.random.default_rng(0)
    results = {}
    print("\n===== networks on SkillCorner's matches, each held out (decoder nested by match) =====")
    print(cp.REFERENCES)
    print("{:<12} {:<15} {:>26} {:>8} {:>10} {:>8}   {}".format(
        "network", "scoring", "spellF1 [95% interval]", "macroF1", "poss/10min", "mean s",
        "decoder settings chosen"))
    for name in NETWORKS:
        table = [{m: tables[name][m][k][CHOOSE_ON] for m in matches}
                 for k in range(len(settings))]
        picks = {m: tm.best(table, [o for o in matches if o != m]) for m in matches}
        codes = {m: next(tm.decoded(evidence[name][m], [settings[picks[m]]])) for m in matches}
        chosen = {}
        for m in matches:
            chosen[settings[picks[m]]] = chosen.get(settings[picks[m]], 0) + 1
        for scoring in cp.SCORINGS:
            counts, rates = [], []
            for m in matches:
                gt = cp.truth(halves[m], scoring)
                counts.append(practice.counts(gt, codes[m]))
                rates.append(tm.spell_rate(codes[m], gt))
            spell, macro = practice.pooled(counts)
            low, high = cp.interval(counts, rng)
            rate, length = np.array(rates).mean(axis=0)
            results[(name, scoring)] = {"spellF1": spell, "interval": (low, high),
                                        "macroF1": macro, "rate": rate, "length": length,
                                        "per_match": [practice.pooled([c])[0] for c in counts]}
            print("{:<12} {:<15} {:>8.4f} [{:.3f}, {:.3f}] {:>8.4f} {:>10.1f} {:>8.1f}   {}".format(
                name, scoring, spell, low, high, macro, rate, length, "; ".join(
                    "{} x{}".format(s, n) for s, n in sorted(chosen.items(),
                                                             key=lambda item: -item[1]))))
    for scoring in cp.SCORINGS:
        rate, length = np.array([tm.spell_rate(cp.truth(halves[m], scoring),
                                               cp.truth(halves[m], scoring))
                                 for m in matches]).mean(axis=0)
        print("SkillCorner's labels, {}: {:.1f} possessions per 10 min, mean {:.1f} s".format(
            scoring, rate, length))
    for name in NETWORKS:
        print("{} (every frame), spellF1 per match: {}".format(name, ", ".join(
            "{} {:.3f}".format(m, v) for m, v in zip(
                matches, results[(name, "every frame")]["per_match"]))))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--copies", type=int, default=2)
    parser.add_argument("--step", type=int, default=8)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) - 1))
    args = parser.parse_args()

    import torch
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    start = time.time()
    matches = list(sc.MATCHES)
    fetched = sc.fetch(matches)
    print("torch {} on {}; SkillCorner at {}: {} files fetched ({:.0f} s)".format(
        torch.__version__, torch.cuda.get_device_name(0) if device == "cuda"
        else "the CPU (slow: is the GPU on?)", sc.COMMIT[:12], fetched, time.time() - start))
    run(matches, folds_of(matches, args.folds), args.copies, args.step, device, args.workers,
        start)
    print("\ndone in {:.0f} s".format(time.time() - start))
    return 0


if __name__ == "__main__":
    sys.exit(main())
