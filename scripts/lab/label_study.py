"""Lab, after the competition: how much the definition of possession moves the score.

The submitted pipeline scored 0.698 held out on the practice ground and 0.526 on the match. One
suspect is the labels: it learned 'controlled' possession (possession.external), while the match's
ground truth may count possession more strictly. This study measures that effect on IDSSE, where
every definition can be computed.

The three definitions of possession.lab_labels, fixed before the run: 'raw' (DFL's possession as
it is), 'controlled' (what the submitted pipeline learned), and 'strict' (controlled, with team
runs under 3 s and fewer than 2 controlled actions merged into the possession they interrupt).

For each definition, the pipeline is fitted on the practice ground with that definition's labels
and its decoder chosen nested by match, exactly as scripts/train_tcn.py does it: the network of
possession.tcn with train_tcn.TRAINING, on two degraded training copies of every half, each match
held out in turn. The held-out codes of each pipeline are then scored against every definition, at
+/-1 s: the diagonal judges a pipeline by the labels it learned, the other cells by labels it never
saw, as a competition's ground truth may. Each cell carries a 95% interval from resampling the
seven matches. Last, the definitions themselves are compared: possessions per ten minutes and
their mean length on the practice matches.

Public data only. A job for Kaggle with a GPU; kaggle/run.py installs PyTorch for it.

    python scripts/lab/label_study.py [--copies 2] [--workers 3]
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_models as tm  # noqa: E402  (puts the repository on the path)
import train_tcn as tt  # noqa: E402
import train_tcn_ensemble as te  # noqa: E402
import train_tcn_variants as tv  # noqa: E402

from possession import external as ex  # noqa: E402
from possession import lab_labels, practice  # noqa: E402
from possession.models import design_names  # noqa: E402

pm = practice.pm
TOLERANCE_S = 1.0
DRAWS = 1000


def truth(match, name):
    """One match's labels under definition name, as codes with an X between halves."""
    _, halves, events = ex.load_match(match)
    labels, _, _ = lab_labels.aligned(name)(halves, events, TOLERANCE_S)
    return practice.joined([practice.codes(labels[s]) for s in ex.SECTIONS], gap=pm.X)


def interval(parts, rng, draws=DRAWS):
    """95% interval of the pooled spellF1 of parts (one counts() per match), matches resampled."""
    values = [practice.pooled([parts[i] for i in rng.integers(len(parts), size=len(parts))])[0]
              for _ in range(draws)]
    return np.percentile(values, [2.5, 97.5])


def fitted_codes(name, matches, copies, step, settings, device, workers, start):
    """{match: held-out codes} of the pipeline fitted and decoded with definition name's labels."""
    with lab_labels.active(name):
        sequences = {}
        _, held_out = tm.build(matches, copies, step, ["v1"], (), None, sequences=sequences)
        view = {m: dict(part, X=part["X"]["v1"], slow=part["slow"]["v1"])
                for m, part in held_out.items()}
        print("\n===== definition {}: built the practice ground ({:.0f} s so far) =====".format(
            name, time.time() - start))
        names = design_names("v1")
        pending = {}
        pool = te.worker_pool(workers)
        try:
            for k in matches:
                halves = [half for m in matches if m != k for half in sequences[m]]
                model = tt.fit_network(halves, dict(settings), device, names)
                pending[k] = tv.submitted(pool, model, tm.halves_of(view[k], "X"))
            parts = {k: tv.results(items) for k, items in pending.items()}
        finally:
            if pool is not None:
                pool.shutdown()
        rule = tm.rules(view, matches)
        tt.report_frames(name, parts, view, matches, rule)
        oof = {m: [tt.log_probabilities(*p) for p in halves] for m, halves in parts.items()}
        _, _, codes, _ = tm.fit_decoder(view, oof, matches, tm.decoder_axes("base"))
    return codes


def run(matches, copies=2, step=8, settings=None, device="cpu", workers=1, start=0.0):
    """The whole study; returns {'cells': {(train, judge): (spellF1, macroF1)}, 'rates': ...}."""
    settings = dict(tt.TRAINING if settings is None else settings)
    defs = lab_labels.DEFINITIONS
    codes = {name: fitted_codes(name, matches, copies, step, settings, device, workers, start)
             for name in defs}
    truths = {name: {m: truth(m, name) for m in matches} for name in defs}

    print("\n===== the definitions on the practice matches, labels at +/-1 s =====")
    rates = {}
    for name in defs:
        rate, length = np.array([tm.spell_rate(truths[name][m], truths[name][m])
                                 for m in matches]).T
        rates[name] = (float(rate.mean()), float(length.mean()))
        print("{:<10} possessions per 10 min {:.1f} (matches {:.1f}-{:.1f}), mean length "
              "{:.1f} s ({:.1f}-{:.1f})".format(name, rate.mean(), rate.min(), rate.max(),
                                                length.mean(), length.min(), length.max()))

    print("\n===== held-out spellF1 / macroF1: fitted on (row), judged by (column) =====")
    print("{:<12}".format("fitted on") + "".join("{:>34}".format(n) for n in defs))
    rng = np.random.default_rng(0)
    cells = {}
    for fitted in defs:
        row = []
        for judge in defs:
            parts = [practice.counts(truths[judge][m], codes[fitted][m]) for m in matches]
            spell, macro = practice.pooled(parts)
            low, high = interval(parts, rng)
            cells[(fitted, judge)] = (spell, macro)
            row.append("{:.4f} [{:.3f}, {:.3f}] / {:.4f}".format(spell, low, high, macro))
        print("{:<12}".format(fitted) + "".join("{:>34}".format(c) for c in row))
    for fitted in defs:
        for judge in defs:
            if fitted != judge:
                per_match = [practice.pooled([practice.counts(truths[judge][m],
                                                              codes[fitted][m])])[0]
                             for m in matches]
                print("{} judged by {}, per match: {}".format(fitted, judge, ", ".join(
                    "{} {:.3f}".format(m, v) for m, v in zip(matches, per_match))))
    return {"cells": cells, "rates": rates}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--copies", type=int, default=2)
    parser.add_argument("--step", type=int, default=8)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) - 1))
    args = parser.parse_args()

    import torch
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("torch {} on {}; definitions {}; strict = runs under {:g} s with fewer than {} "
          "controlled actions merged".format(
              torch.__version__, torch.cuda.get_device_name(0) if device == "cuda"
              else "the CPU (slow: is the GPU on?)", ", ".join(lab_labels.DEFINITIONS),
              lab_labels.STRICT_SECONDS, lab_labels.STRICT_ACTIONS))
    start = time.time()
    run(sorted(ex.MATCHES), args.copies, args.step, device=device, workers=args.workers,
        start=start)
    print("\ndone in {:.0f} s".format(time.time() - start))
    return 0


if __name__ == "__main__":
    sys.exit(main())
