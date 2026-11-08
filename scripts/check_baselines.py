"""Milestone M3: rebuild the organisers' four baselines from the match data and compare them.

If the rebuilt labels equal the committed ones, the data bundle, the frame alignment and this
environment all agree with the organisers', which is the cheapest end-to-end check of the chain.

    python scripts/check_baselines.py

It reads the match data, so it follows AGENTS.md: it prints aggregates only (team names, ball
coverage, how many frames differ, the public-window scores), and run_safely() withholds exception
messages, which can quote a line of tracking. Labels are compared rather than bytes, because a
Windows checkout may store the committed CSVs with CRLF line endings.
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (ROOT, os.path.join(ROOT, "starter"), os.path.join(ROOT, "evaluation")):
    sys.path.insert(0, path)

from possession.safety import run_safely  # noqa: E402

# Public-window scores of baseline_smooth50, as docs/challenge/README.md prints them. A rebuild
# that matches the committed file scores exactly this.
EXPECTED_PUBLIC = {"macro_f1": 0.6207, "spell_f1_mean": 0.3256}


def committed_labels(name):
    path = os.path.join(ROOT, "evaluation", "results", name + ".csv")
    return pd.read_csv(path, dtype={"label": "object"})["label"].to_numpy()


def main():
    import baseline  # starter/baseline.py, the organisers' reference implementation
    import possession_metrics as pm

    codes = baseline.build()  # prints the team names and the ball coverage, both aggregates
    mismatches = 0
    for name, window in baseline.VARIANTS:
        rebuilt = baseline.smooth(codes, window)
        committed = committed_labels(name)
        if len(rebuilt) != len(committed):
            print("{:<20} DIFFERS: {} rows vs {} committed".format(name, len(rebuilt),
                                                                  len(committed)))
            mismatches += 1
            continue
        differing = int(np.sum(rebuilt != committed))
        mismatches += differing > 0
        print("{:<20} {}".format(name, "identical" if not differing
                                 else "DIFFERS on {} frames".format(differing)))

    gt_path = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
    gt = pm.label_codes(pm.read_label_csv(gt_path))
    result = pm.evaluate(gt, pm.label_codes(baseline.smooth(codes, 50)))
    print("\npublic window, baseline_smooth50: macroF1 {:.4f}  spellF1 {:.4f}  "
          "(expected {:.4f} / {:.4f})".format(result["macro_f1"], result["spell_f1_mean"],
                                              EXPECTED_PUBLIC["macro_f1"],
                                              EXPECTED_PUBLIC["spell_f1_mean"]))

    if mismatches:
        print("\nMISMATCH: the rebuild disagrees with the committed baselines.")
        return 1
    print("\nOK: data, alignment and environment agree with the organisers'.")
    return 0


if __name__ == "__main__":
    sys.exit(run_safely(main))
