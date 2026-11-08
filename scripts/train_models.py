"""Fit the dead-ball and team models on the practice ground, choosing C leave-one-match-out.

Each IDSSE match is held out in turn: the models are fitted on the other six (every half degraded
with --seeds seeds, every --step-th frame kept) and judged on the held-out match, degraded with a
seed no training copy used. The C with the lowest pooled log loss wins, the final models are
refitted on all seven matches, and they are written to models/possession_v1.json. The baseline's
own rules are scored beside them on the same frames: "no ball means dead" and "the nearest
player's team has it". Public data only; reads no match data.

    python scripts/train_models.py [--seeds 2] [--step 8] [--out models/possession_v1.json]
"""

import argparse
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
from possession.features import HOME, compute, swapped  # noqa: E402
from possession.models import design, fit, probability, save  # noqa: E402
from possession.predictors import NOBODY, nearest_team  # noqa: E402

PITCH = (105.0, 68.0)
GRID = (0.01, 0.1, 1.0)
EVAL_SEED = 99


def build(match_ids, seeds, step):
    """Training rows per match, and full held-out frames per match for evaluation."""
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
                                                   "ball_known": []})
        part["X"].append(X)
        part["labels"].append(case["labels"])
        part["nearest"].append(nearest_team(case["half"]))
        part["ball_known"].append(X[:, 0] + X[:, 1] > 0)  # ball_seen or ball_bridged
    for part in held_out.values():
        for key in part:
            part[key] = np.concatenate(part[key])
    return train, held_out


def stack(train, matches, model):
    X = np.concatenate([x for m in matches for x in train[m][model][0]])
    y = np.concatenate([y for m in matches for y in train[m][model][1]])
    return X, y


def evaluate(train, held_out, matches, C):
    """Pooled held-out predictions over all folds, for one C."""
    pooled = {"dead_p": [], "dead_y": [], "team_p": [], "team_y": [], "nearest": [],
              "known": [], "no_ball": []}
    for match in matches:
        others = [m for m in matches if m != match]
        dead_model = fit(*stack(train, others, "dead"), C=C)
        team_model = fit(*stack(train, others, "team"), C=C)
        part = held_out[match]
        live = part["labels"] != "D"
        pooled["dead_p"].append(probability(dead_model, part["X"]))
        pooled["dead_y"].append(part["labels"] == "D")
        pooled["no_ball"].append(~part["ball_known"])
        pooled["team_p"].append(probability(team_model, part["X"][live]))
        pooled["team_y"].append(part["labels"][live] == "H")
        pooled["nearest"].append(part["nearest"][live])
        pooled["known"].append(part["ball_known"][live])
    return {key: np.concatenate(value) for key, value in pooled.items()}


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

    losses = {}
    for C in GRID:
        losses[C] = report(C, evaluate(train, held_out, matches, C))
    best_dead = min(GRID, key=lambda c: losses[c][0])
    best_team = min(GRID, key=lambda c: losses[c][1])
    print("chosen C: dead {}, team {} ({:.0f} s)".format(best_dead, best_team,
                                                          time.time() - start))

    models = {"dead": fit(*stack(train, matches, "dead"), C=best_dead),
              "team": fit(*stack(train, matches, "team"), C=best_team),
              "fitted_on": {"matches": matches, "seeds": args.seeds, "step": args.step,
                            "label_tolerance_s": 1.0, "pitch": list(PITCH),
                            "scikit_learn": sklearn_version}}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save(models, args.out)
    print("wrote {}".format(os.path.relpath(args.out, ROOT)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
