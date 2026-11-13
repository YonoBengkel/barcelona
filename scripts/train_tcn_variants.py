"""M11n: three more candidates against the one network adopted in M11l.

The network of M11l learns its seven matches far better than it predicts a held-out one (training
loss near 0.125, held-out near 0.36), and two earlier candidates nearly passed: the trees averaged
with the network (M11l, higher at the second seed) and five networks averaged (M11m, +0.0096 at
EVAL_SEED but better on only 4 of 7 matches). So four pipelines are fitted and scored on the
practice ground, each match held out in turn exactly as scripts/train_tcn.py does it, with the
decoder chosen the same nested way on the base grid:

- the reference, 'v1+tcn': one network, seed 0, train_tcn.TRAINING, fitted on the two training
  copies of every half (seeds 0 and 1). It is the adopted pipeline, fitted again here;
- 'v1+tcn+copies8': the same network fitted on eight degraded copies of every half (seeds 0 to 7),
  more varied training data against the overfitting;
- 'v1+tcn+dil256': the same network with one more residual block, dilations 1 to 256, so a frame
  reads 511 frames (about 20 s) either side instead of 255: on the match, most frames without the
  ball fall in gaps of 10 s or more;
- 'v1+trees-wide+deadtrees+tcn5': the frozen pipeline's trees, fitted and chosen as
  scripts/train_tcn.py does it on the two copies, averaged with five networks (seeds 0 to 4, the
  reference's among them) as possession.models.scores averages a file that holds both.

Everything else is the adopted pipeline's, and nothing is tuned on a held-out match.

The adoption rule is train_models.compare's, set before the run: held-out spellF1 at +/-1 s at
least 0.01 higher than the reference, better on at least 5 of the 7 matches, no loss at +/-0.5 s
or +/-2 s, macroF1 down by at most 0.005. A candidate is adopted only if it passes on the held-out
halves degraded with EVAL_SEED and again, with the same fitted folds and the decoder chosen the
same nested way, on a fresh degradation (--confirm-seed 96, which no earlier run used). If more
than one passes twice, the one higher at EVAL_SEED is adopted.

The held-out log-odds come from possession.tcn.scores, the numpy code the submission would run,
in --workers processes while the GPU fits the next network. Last, only an adopted candidate is
fitted on all seven matches and written to models/possession_<name>.json.

Public data only; reads no match data. A job for Kaggle with a GPU (kaggle/README.md); the runner
installs PyTorch for it from requirements-torch.txt. Nothing the submission runs needs PyTorch.

    python scripts/train_tcn_variants.py [--step 8] [--confirm-seed 96] [--workers 3]
                                         [--out-dir models]
"""

import argparse
import concurrent.futures
import os
import sys
import time
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_models as tm  # noqa: E402  (puts the repository on the path)
import train_tcn as tt  # noqa: E402
import train_tcn_ensemble as te  # noqa: E402

from possession import external as ex  # noqa: E402
from possession import tcn  # noqa: E402
from possession.models import combine, design_names, save, score  # noqa: E402
from possession.simulate import DegradeParams  # noqa: E402

COPIES = 2  # the adopted network's training copies of every half
MORE_COPIES = 8
WIDE_DILATIONS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
NETWORKS = 5
CONFIRM_SEED = 96
REFERENCE = "v1+tcn"


def copies(sequences, count):
    """{match: its training halves of the copies (degradation seeds) below count}, in build()'s
    order, so the first two copies of an eight-copy build are the halves a two-copy build gives."""
    return {m: [half for half in halves if half["seed"] < count]
            for m, halves in sequences.items()}


def specs(settings, networks, wide_dilations, copies_count, more_copies):
    """[(key, settings, copies)]: the networks fitted per fold, the reference first. The reference
    is also the first of the averaged networks, whose others follow it seed by seed."""
    plan = [("reference", dict(settings), copies_count),
            ("more", dict(settings), more_copies),
            ("wider", dict(settings, dilations=list(wide_dilations)), copies_count)]
    return plan + [("seed{}".format(s), dict(settings, seed=s), copies_count)
                   for s in range(1, networks)]


def submitted(pool, model, Xs):
    """possession.tcn.scores of model on each half: futures in the pool's worker processes, so the
    GPU fits the next network meanwhile, or without a pool the results themselves."""
    if pool is None:
        return [tcn.scores(model, X) for X in Xs]
    return [pool.submit(tcn.scores, model, X) for X in Xs]


def results(items):
    """The (dead, home) of each of submitted()'s items, waiting for those still being scored."""
    return [item.result() if isinstance(item, concurrent.futures.Future) else item
            for item in items]


def with_trees(trees, networks):
    """Per half, the trees' (dead, home) averaged with the networks' average, as
    possession.models.scores averages a file that holds both: combine([trees, combine(networks)]).
    networks: one list of per-half (dead, home) per network, in the file's order."""
    return [combine([part, combine([net[i] for net in networks])])
            for i, part in enumerate(trees)]


def run(train, sequences, held_out, confirm, matches, out_dir, start, step=8,
        confirm_seed=CONFIRM_SEED, settings=None, device="cpu", workers=1, copies_count=COPIES,
        more_copies=MORE_COPIES, wide_dilations=WIDE_DILATIONS, networks=NETWORKS):
    """The whole comparison on built data; see the module docstring. train holds the training rows
    of copies_count copies (the trees'), sequences the whole training halves of more_copies
    copies. Returns {'verdict': the adopted name or None, 'summaries': {seed: {name:
    fit_decoder's summary}}}."""
    settings = dict(tt.TRAINING if settings is None else settings)
    more = "v1+tcn+copies{}".format(more_copies)
    wider = "v1+tcn+dil{}".format(wide_dilations[-1])
    both = tm.variant_name("v1", "trees", "trees-wide") + "+tcn{}".format(networks)
    candidates = (more, wider, both)
    single = {REFERENCE: "reference", more: "more", wider: "wider"}
    averaged = ["reference"] + ["seed{}".format(s) for s in range(1, networks)]
    grounds = ((tm.EVAL_SEED, held_out), (confirm_seed, confirm))
    names = design_names("v1")
    halves_of_copies = {count: copies(sequences, count) for count in (copies_count, more_copies)}
    rule = tm.rules(held_out, matches)

    dead_label, dead_folds, dead_final = tm.choose("dead", "trees", train, held_out, matches,
                                                   "v1", rule)
    team_label, team_folds, team_final = tm.choose("team", "trees-wide", train, held_out,
                                                   matches, "v1", rule)
    print("chosen dead {}, team {} ({:.0f} s so far)".format(dead_label, team_label,
                                                            time.time() - start))
    trees = {seed: {m: [(score(dead_folds[m], X), score(team_folds[m], X))
                        for X in tm.halves_of(part[m], "X")] for m in matches}
             for seed, part in grounds}

    plan = specs(settings, networks, wide_dilations, copies_count, more_copies)
    pending = {seed: {m: {} for m in matches} for seed, _ in grounds}
    pool = te.worker_pool(workers)
    try:
        for k in matches:
            for key, spec, count in plan:
                halves = [half for m in matches if m != k for half in halves_of_copies[count][m]]
                model = tt.fit_network(halves, spec, device, names)
                for seed, part in grounds:
                    pending[seed][k][key] = submitted(pool, model, tm.halves_of(part[k], "X"))
            print("{} networks held out on {}: fitted and sent to be scored ({:.0f} s so far)"
                  .format(len(plan), k, time.time() - start))
        net = {seed: {m: {key: results(items) for key, items in by_key.items()}
                      for m, by_key in by_match.items()} for seed, by_match in pending.items()}
    finally:
        if pool is not None:
            pool.shutdown()
    print("every held-out half scored ({:.0f} s so far)".format(time.time() - start))

    def parts_of(seed, name):
        if name == both:
            return {m: with_trees(trees[seed][m], [net[seed][m][key] for key in averaged])
                    for m in matches}
        return {m: net[seed][m][single[name]] for m in matches}

    print()
    for name in (REFERENCE,) + candidates:
        tt.report_frames(name, parts_of(tm.EVAL_SEED, name), held_out, matches, rule)

    axes = tm.decoder_axes("base")
    decoders, summaries = {}, {}
    for seed, part in grounds:
        summaries[seed] = {}
        for name in (REFERENCE,) + candidates:
            oof = {m: [tt.log_probabilities(*p) for p in halves]
                   for m, halves in parts_of(seed, name).items()}
            print("\n----- {}, held-out halves degraded with seed {} ({:.0f} s so far) -----"
                  .format(name, seed, time.time() - start))
            setting, practice_range, codes, summary = tm.fit_decoder(part, oof, matches, axes)
            if seed == tm.EVAL_SEED:
                tm.report_errors(part, codes, matches)
                decoders[name] = (setting, practice_range)
            summaries[seed][name] = summary

    passing = {}
    for seed, _ in grounds:
        print("\n===== the adoption rule on the held-out halves of seed {} =====".format(seed))
        passing[seed] = [name for name in candidates if tm.compare(
            {REFERENCE: summaries[seed][REFERENCE], name: summaries[seed][name]}, matches,
            REFERENCE)]
    adopted = [name for name in candidates if all(name in passing[s] for s, _ in grounds)]
    verdict = max(adopted, key=lambda name: summaries[tm.EVAL_SEED][name]["pooled"][1.0][0]) \
        if adopted else None
    print("\nM11n verdict: {} (passing at seed {}: {}; at seed {}: {})".format(
        "adopt " + verdict if verdict else "keep " + REFERENCE,
        tm.EVAL_SEED, ", ".join(passing[tm.EVAL_SEED]) or "none",
        confirm_seed, ", ".join(passing[confirm_seed]) or "none"))
    if verdict is None:
        return {"verdict": verdict, "summaries": summaries}

    def everything(count):
        return [half for m in matches for half in halves_of_copies[count][m]]

    setting, practice_range = decoders[verdict]
    fitted_on = {"matches": matches, "step": step, "label_tolerance_s": 1.0,
                 "pitch": list(tm.PITCH), "decoder_grid": {k: list(v) for k, v in axes.items()},
                 "practice_range": practice_range, "simulator": asdict(DegradeParams()),
                 "confirm_seed": confirm_seed}
    if verdict == both:
        final = [tt.fit_network(everything(count), spec, device, names)
                 for key, spec, count in plan if key in averaged]
        models = {"design": "v1", "dead": dead_final, "team": team_final, "tcn": final}
        fitted_on.update(seeds=copies_count, scikit_learn=tm.sklearn_version,
                         dead_model=dead_label, team_model=team_label,
                         network=dict(settings, channels=tcn.CHANNELS, kernel=tcn.KERNEL,
                                      dilations=list(tcn.DILATIONS),
                                      seeds=list(range(networks))))
    else:
        _, spec, count = next(entry for entry in plan if entry[0] == single[verdict])
        models = {"design": "v1", "tcn": tt.fit_network(everything(count), spec, device, names)}
        fitted_on.update(seeds=count, network=dict(
            spec, channels=tcn.CHANNELS, kernel=tcn.KERNEL,
            dilations=list(spec.get("dilations", tcn.DILATIONS))))
    models.update(decoder=setting, fitted_on=fitted_on)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "possession_{}.json".format(verdict))
    save(models, out)
    print("wrote {} ({:.0f} s so far)".format(os.path.relpath(out, tm.ROOT), time.time() - start))
    return {"verdict": verdict, "summaries": summaries}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--step", type=int, default=8)
    parser.add_argument("--confirm-seed", type=int, default=CONFIRM_SEED)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) - 1))
    parser.add_argument("--out-dir", default=os.path.join(tm.ROOT, "models"))
    args = parser.parse_args()
    if args.confirm_seed == tm.EVAL_SEED or 0 <= args.confirm_seed < MORE_COPIES:
        parser.error("--confirm-seed must differ from EVAL_SEED and from every training seed")

    import torch
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("torch {} on {}; {} worker processes score the held-out halves".format(
        torch.__version__, torch.cuda.get_device_name(0) if device == "cuda"
        else "the CPU (slow: is the GPU on?)", args.workers))
    matches = sorted(ex.MATCHES)
    start = time.time()
    # The trees' rows and the held-out halves as M11l built them, then the whole training halves
    # of all eight copies, whose first two are the halves M11l's network was fitted on.
    train, held_out = tm.build(matches, COPIES, args.step, ["v1"], (), None)
    sequences = {}
    tm.build(matches, MORE_COPIES, args.step, ["v1"], (), None, sequences=sequences)
    _, confirm = tm.build(matches, 0, args.step, ["v1"], (), None, eval_seed=args.confirm_seed)
    print("built rows and whole halves in {:.0f} s: dead {}, team {}, {} training halves".format(
        time.time() - start, len(tm.stack(train["v1"], matches, "dead")[1]),
        len(tm.stack(train["v1"], matches, "team")[1]),
        sum(len(halves) for halves in sequences.values())))
    view = {m: dict(part, X=part["X"]["v1"], slow=part["slow"]["v1"])
            for m, part in held_out.items()}
    confirm_view = {m: dict(part, X=part["X"]["v1"], slow=part["slow"]["v1"])
                    for m, part in confirm.items()}
    run(train["v1"], sequences, view, confirm_view, matches, args.out_dir, start, args.step,
        args.confirm_seed, device=device, workers=args.workers)
    print("\ndone in {:.0f} s".format(time.time() - start))
    return 0


if __name__ == "__main__":
    sys.exit(main())
