"""How much signal each feature carries on the practice ground, one feature at a time.

For every feature: its ROC AUC at telling a dead ball from a live one (all frames) and the home
side from the away side (live frames), with the share of frames where it is known. An AUC far
from 0.5 in either direction is signal; near 0.5 is noise. Also checks that the two ways of
reading each team's attacking direction agree. Public data only; reads no match data.

    python scripts/feature_report.py [--seed 0] [MATCH_ID ...]
"""

import argparse
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from possession import external as ex  # noqa: E402
from possession import practice  # noqa: E402
from possession.features import HOME, AWAY, NAMES, attack_directions, compute, row_mean  # noqa: E402

IDSSE_PITCH = (105.0, 68.0)


def auc(values, target):
    keep = np.isfinite(values)
    if keep.sum() < 100 or target[keep].all() or not target[keep].any():
        return float("nan"), keep.mean()
    return roc_auc_score(target[keep], values[keep]), keep.mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("matches", nargs="*", default=sorted(ex.MATCHES))
    args = parser.parse_args()

    cases = practice.cases(args.matches, tolerance_s=1.0, seed=args.seed)
    blocks, labels, agree = [], [], 0
    for case in cases:
        features, _ = compute(case["half"], IDSSE_PITCH)
        blocks.append(features)
        labels.append(case["labels"])
        means = [np.nanmean(row_mean(case["half"]["x"][:, case["half"]["team"] == t])[0])
                 for t in (HOME, AWAY)]
        by_centroid = 1.0 if means[0] < means[1] else -1.0
        agree += attack_directions(case["half"])[HOME] == by_centroid
    features = np.concatenate(blocks)
    labels = np.concatenate(labels)
    dead = labels == "D"
    alive = ~dead
    home = labels[alive] == "H"

    print("{} halves, {} frames; dead share {:.1%}; attacking direction: keeper rule and "
          "centroid rule agree on {}/{} halves".format(len(cases), len(labels), dead.mean(),
                                                      agree, len(cases)))
    rows = []
    for i, name in enumerate(NAMES):
        dead_auc, dead_cov = auc(features[:, i].astype(np.float64), dead)
        team_auc, team_cov = auc(features[alive, i].astype(np.float64), home)
        rows.append((name, dead_auc, dead_cov, team_auc, team_cov))
    print("\n{:<20} {:>9} {:>8}   {:>9} {:>8}".format("feature", "dead AUC", "known", "home AUC",
                                                     "known"))
    for name, dead_auc, dead_cov, team_auc, team_cov in rows:
        print("{:<20} {:>9.3f} {:>8.0%}   {:>9.3f} {:>8.0%}".format(name, dead_auc, dead_cov,
                                                                  team_auc, team_cov))
    return 0


if __name__ == "__main__":
    sys.exit(main())
