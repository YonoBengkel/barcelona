"""Label every frame of the match and write the submission CSV (milestone M9).

    python scripts/make_submission.py [--out evaluation/results/possession_v1.csv]

The one command that turns the match's tracking into the file the organisers score. It reads the
tactical tracking and cuts out the two graded halves (games.half_video_blocks). It computes the
position features (possession.features), weighs them with the dead-ball and team models, and
decodes them into possessions with the decoder chosen on the practice ground
(models/possession_v1.json). It writes one frame,label row per video frame with LF line endings:
H, A or D inside the graded halves, and X outside them, where nothing is graded. The organisers'
own validation checks the file before anything else is reported.

It reads the match data, so it follows AGENTS.md: it runs on the team's laptop, never on Kaggle;
it prints aggregates only; and run_safely() withholds exception messages, which can quote a line of
tracking. What it prints:

- how many frames carry each label;
- how many possessions the file reports per ten minutes and how long they last, beside the range
  on the practice matches;
- whether the attacking directions inferred from the tracking agree with the metadata, yes or no;
- how often each label is called while the ball is seen, bridged, or missing for up to 5 s,
  5-15 s or longer, to compare with the same table on the practice ground (possession.diagnose);
- the scores on the public five-minute window. They check that the pipeline is wired the right
  way round and never choose anything (AGENTS.md, rule 3);
- the file's SHA-256, to compare two runs.
"""

import argparse
import hashlib
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (ROOT, os.path.join(ROOT, "evaluation")):
    sys.path.insert(0, path)

from possession.safety import run_safely  # noqa: E402

DEFAULT_OUT = os.path.join(ROOT, "evaluation", "results", "possession_v1.csv")
DEFAULT_MODELS = os.path.join(ROOT, "models", "possession_v1.json")
PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
# baseline_smooth50 on the public window, as docs/challenge/README.md prints it.
PUBLIC_BASELINE = (0.6207, 0.3256)


def shown(path):
    """The path relative to the repository when it can be, else as given (another drive)."""
    try:
        return os.path.relpath(path, ROOT)
    except ValueError:
        return path


def main():
    import possession_metrics as pm
    from games import GAME, GAMES, half_video_blocks, require
    from possession import diagnose, practice
    from possession.atd import halves, metadata_attack_directions, read_atd
    from possession.decode import decode
    from possession.features import FPS, attack_directions, compute
    from possession.models import design, load, log_state_probabilities
    from possession.submission import assemble, write

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--models", default=DEFAULT_MODELS)
    args = parser.parse_args()
    start = time.time()

    models = load(args.models)
    if "decoder" not in models:
        print("{} has no decoder yet: run scripts/train_models.py on Kaggle first "
              "(kaggle/README.md)".format(shown(args.models)))
        return 1
    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME)
    n_frames, pitch = GAMES[GAME]["n_video_frames"], GAMES[GAME]["pitch"]
    atd = read_atd(meta_path, raw_path, n_frames, pitch)
    declared = metadata_attack_directions(meta_path)

    parts, agreement, states, calls, speeds = [], [], [], [], []
    for number, half in enumerate(halves(atd, half_video_blocks(GAME))):
        features = compute(half, pitch)[0]
        # float32 as in training (scripts/train_models.py), so trees see the same values.
        X = design(features, models.get("design", "v1")).astype(np.float32)
        labels = decode(log_state_probabilities(models, X), models["decoder"])
        parts.append((half["first_frame"], labels))
        states.append(diagnose.ball_states(features))
        calls.append(practice.codes(labels))
        speeds.append(diagnose.player_speed(features))
        if 0 in declared:
            agreement.append("not readable from the metadata")
        else:
            ends = 1 if number == 0 else -1  # the teams change ends at halftime
            expected = [ends * direction for direction in declared]
            agreement.append("yes" if list(attack_directions(half)) == expected else "NO")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write(args.out, assemble(n_frames, parts))
    table = pm.read_label_csv(args.out)
    reason = pm.validate_submission(table, n_frames)
    if reason:
        print("INVALID  {}".format(reason))
        return 1
    codes = pm.label_codes(table)

    print("models {} (design {}), decoder {}".format(
        shown(args.models), models.get("design", "v1"), models["decoder"]))
    graded = codes != pm.X
    print("labels on the {} graded frames: {}; X on the other {}".format(
        int(graded.sum()), "  ".join("{} {:.1%}".format(name, np.mean(codes[graded] == index))
                                     for index, name in enumerate(pm.SCORED)),
        int((~graded).sum())))
    runs = pm.valid_runs(codes)  # the two graded halves: every row there is H, A or D
    stats = pm.possession_stats(codes, runs)
    rate = stats["n_spells"] / (sum(b - a for a, b in runs) / FPS / 60.0) * 10.0
    print("possessions: {} ({:.1f} per 10 min of graded play), mean {:.1f} s".format(
        stats["n_spells"], rate, stats["mean_duration_s"]))
    # 'ours' is what the same pipeline produced on the held-out practice matches: the like-for-like
    # check. 'labels' is the practice matches' own possessions, which run shorter than the
    # challenge's, so falling outside that range alone is no alarm.
    for name in ("ours", "labels"):
        practice_range = models["fitted_on"].get("practice_range", {}).get(name)
        if not practice_range:
            continue
        low, high = practice_range["spells_per_10min"]
        short, long_ = practice_range["mean_spell_s"]
        inside = low <= rate <= high and short <= stats["mean_duration_s"] <= long_
        print("  practice matches, {}: {:.1f}-{:.1f} per 10 min, mean {:.1f}-{:.1f} s; inside: {}"
              .format("our decoder held out" if name == "ours" else "aligned labels", low, high,
                      short, long_, "yes" if inside else "no"))
    print("attacking directions inferred from the tracking agree with the metadata: first half "
          "{}, second half {}".format(*agreement))
    print("calls by ball state on the graded frames (train_models' log has the practice ground's):")
    for line in diagnose.format_state_table(diagnose.state_table(
            np.concatenate(states), np.concatenate(calls), speed=np.concatenate(speeds))):
        print("  " + line)

    public = pm.evaluate(pm.label_codes(pm.read_label_csv(PUBLIC_GT)), codes)
    print("public five minutes, a wiring check that never chooses anything: macroF1 {:.4f} "
          "spellF1 {:.4f} (baseline_smooth50 {:.4f} / {:.4f})".format(
              public["macro_f1"], public["spell_f1_mean"], *PUBLIC_BASELINE))

    with open(args.out, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    print("wrote {}: {} rows, SHA-256 {} ({:.0f} s)".format(
        shown(args.out), len(codes), digest, time.time() - start))
    return 0


if __name__ == "__main__":
    sys.exit(run_safely(main))
