"""Score predictors on the practice ground: degraded IDSSE, aligned labels, official metrics.

The one place where model and parameter choices are made (AGENTS.md). Each predictor runs on
every degraded half; scores are pooled over all matches (each possession counted once) and shown
per match for the spread. The organisers' published full-match scores of their baselines are
printed beside ours: if the practice ground is faithful, the baseline family ranks the same way.
Public data only; reads no match data.

    python scripts/eval_external.py [--tolerance 0.5 1.0 2.0] [--seeds 1] [MATCH_ID ...]
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from possession import external as ex  # noqa: E402
from possession import practice  # noqa: E402
from possession.predictors import baseline  # noqa: E402

# docs/challenge/EVALUATION.md: the baselines on the full match, (macroF1, spellF1).
PUBLISHED = {"baseline_raw": (0.6532, 0.0486), "baseline_smooth25": (0.6752, 0.1996),
             "baseline_smooth50": (0.6724, 0.2433), "baseline_smooth125": (0.6356, 0.2242)}
PREDICTORS = [baseline(0), baseline(25), baseline(50), baseline(125)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tolerance", type=float, nargs="+", default=[1.0])
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("matches", nargs="*", default=sorted(ex.MATCHES))
    args = parser.parse_args()

    for tolerance in args.tolerance:
        cases = []
        for seed in range(args.seeds):
            cases += practice.cases(args.matches, tolerance_s=tolerance, seed=seed)
        print("\nlabels aligned at +/-{:.1f} s; {} halves from {} matches x {} seed(s)".format(
            tolerance, len(cases), len(args.matches), args.seeds))
        print("{:<20} {:>8} {:>8} {:>6} {:>6} {:>14} {:>16}".format(
            "predictor", "macroF1", "spellF1", "poss", "gt", "spellF1 range", "match: mF1/sF1"))
        for predict in PREDICTORS:
            pooled, per_match = practice.score(cases, [predict(c["half"]) for c in cases])
            spells = [r["spell_f1_mean"] for r in per_match.values()]
            published = PUBLISHED.get(predict.__name__)
            print("{:<20} {:>8.4f} {:>8.4f} {:>6} {:>6} {:>6.3f}-{:<6.3f} {:>16}".format(
                predict.__name__, pooled["macro_f1"], pooled["spell_f1_mean"],
                pooled["n_pred_spells"], pooled["n_gt_spells"], min(spells), max(spells),
                "{:.4f}/{:.4f}".format(*published) if published else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
