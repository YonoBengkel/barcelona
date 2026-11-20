"""Lab: does adding SkillCorner to the training data cost anything on IDSSE?

scripts/lab/skillcorner_training.py scores networks on SkillCorner's matches. This job looks from
the other side, on the practice ground itself: each IDSSE match held out in turn, the network
(possession.tcn, train_tcn.TRAINING) fitted on

- 'idsse': the other six IDSSE matches, two degraded copies of every half, as the practice ground
  builds them, which is the label study's 'controlled' pipeline (scripts/lab/label_study.py);
- 'both': the same halves together with all 20 SkillCorner matches as given, each seen as it is
  and with home and away exchanged. The two sources label possession their own way, so the
  network learns from a blend.

Each pipeline's decoder is chosen nested by match on the base grid, and its held-out codes are
scored against the controlled labels at +/-0.5, 1 and 2 s by train_models.fit_decoder, the same
table the practice ground has always printed.

Public data only. A job for Kaggle with a GPU; kaggle/run.py installs PyTorch for it.

    python scripts/lab/multisource_idsse.py [--copies 2] [--workers N]
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cross_provider as cp  # noqa: E402  (puts scripts/ and the repository on the path)
import skillcorner_training as st  # noqa: E402
import train_models as tm  # noqa: E402
import train_tcn as tt  # noqa: E402
import train_tcn_ensemble as te  # noqa: E402

from possession import external as ex  # noqa: E402
from possession import skillcorner as sc  # noqa: E402
from possession.models import design_names  # noqa: E402

SOURCES = ("idsse", "both")


def run(matches, copies, step, device, workers, start):
    """Fit and score both pipelines; returns {source: fit_decoder's summary}."""
    names = design_names(st.VERSION)
    sequences = {}
    _, held_out = tm.build(matches, copies, step, [st.VERSION], (), None, sequences=sequences)
    view = {m: dict(part, X=part["X"][st.VERSION], slow=part["slow"][st.VERSION])
            for m, part in held_out.items()}
    print("IDSSE: practice ground built, {} copies ({:.0f} s so far)".format(
        copies, time.time() - start))
    pool = te.worker_pool(workers)
    try:
        corner = dict((pool.map if pool else map)(st.match_halves, sc.MATCHES))
    finally:
        if pool is not None:
            pool.shutdown()
    extra = [half for m in sc.MATCHES for half in corner[m]]
    print("SkillCorner: {} halves from {} matches ({:.0f} s so far)".format(
        len(extra), len(corner), time.time() - start))

    summaries = {}
    for source in SOURCES:
        parts = {}
        for k in matches:
            train = [half for m in matches if m != k for half in sequences[m]]
            if source == "both":
                train = train + extra
            model = tt.fit_network(train, dict(tt.TRAINING), device, names)
            parts[k] = [cp.fast_scores(model, X) for X in tm.halves_of(view[k], "X")]
        print("\n===== {}: fitted with {} held out in turn ({:.0f} s so far) =====".format(
            source, len(matches), time.time() - start))
        tt.report_frames(source, parts, view, matches, tm.rules(view, matches))
        oof = {m: [tt.log_probabilities(*p) for p in halves] for m, halves in parts.items()}
        _, _, _, summaries[source] = tm.fit_decoder(view, oof, matches, tm.decoder_axes("base"))

    print("\n===== held-out spellF1 on IDSSE, controlled labels =====")
    print(cp.REFERENCES)
    for source in SOURCES:
        pooled = summaries[source]["pooled"]
        print("{:<6} ".format(source) + "  ".join(
            "+/-{:g} s spellF1 {:.4f} mF1 {:.4f}".format(t, *pooled[t]) for t in tm.TOLERANCES))
    better = sum(summaries["both"]["per_match"][m] > summaries["idsse"]["per_match"][m]
                 for m in matches)
    print("'both' better than 'idsse' on {} of {} matches".format(better, len(matches)))
    return summaries


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
    start = time.time()
    fetched = sc.fetch()
    print("torch {} on {}; SkillCorner at {}: {} files fetched ({:.0f} s)".format(
        torch.__version__, torch.cuda.get_device_name(0) if device == "cuda"
        else "the CPU (slow: is the GPU on?)", sc.COMMIT[:12], fetched, time.time() - start))
    run(sorted(ex.MATCHES), args.copies, args.step, device, args.workers, start)
    print("\ndone in {:.0f} s".format(time.time() - start))
    return 0


if __name__ == "__main__":
    sys.exit(main())
