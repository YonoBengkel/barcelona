"""M11m: an ensemble of networks against the one network adopted in M11l.

The network of M11l (possession.tcn, fitted by scripts/train_tcn.py) learns its seven matches far
better than it predicts a held-out one: its training loss ends near 0.125, its held-out loss of
both calls together near 0.36, and its team log-odds are overconfident. Averaging networks that
differ only in the seed of their fitting is the standard way to cut that variance without tuning
anything. And its decoder chose dead_bias 1.0, the top of the grid, in all seven folds.

So three pipelines are fitted and scored on the practice ground, each match held out in turn
exactly as scripts/train_tcn.py does it, with the decoder chosen the same nested way:

- the reference, 'v1+tcn': one network, fitted with seed 0 and train_tcn.TRAINING, the base
  decoder grid. It is the adopted pipeline, fitted again here;
- 'v1+tcn5': the log-odds of --networks networks averaged (possession.models.combine), fitted with
  seeds 0 to 4 and the same settings, so the reference's network is one of them; base grid;
- 'v1+tcn5+widebias': the same ensemble with train_models' wide grid, whose dead_bias reaches 3.0.

The adoption rule is train_models.compare's, set before the run: held-out spellF1 at +/-1 s at
least 0.01 higher than the reference, better on at least 5 of the 7 matches, no loss at +/-0.5 s
or +/-2 s, macroF1 down by at most 0.005. A candidate is adopted only if it passes on the held-out
halves degraded with EVAL_SEED and again, with the same fitted networks and the decoder chosen the
same nested way, on a fresh degradation (--confirm-seed 97, which no earlier run used). If both
pass twice, the one higher at EVAL_SEED is adopted.

The held-out log-odds come from possession.tcn.scores, the numpy code the submission would run, in
--workers processes at once (each network and half is independent). Last, the networks are
fitted on all seven matches and both candidates are written to models/possession_<name>.json,
whatever the verdict. A file is committed only if adopted.

Public data only; reads no match data. A job for Kaggle with a GPU (kaggle/README.md); the runner
installs PyTorch for it from requirements-torch.txt. Nothing the submission runs needs PyTorch.

    python scripts/train_tcn_ensemble.py [--networks 5] [--seeds 2] [--step 8]
                                         [--confirm-seed 97] [--workers 4] [--out-dir models]
"""

import argparse
import concurrent.futures
import multiprocessing
import os
import sys
import time
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_models as tm  # noqa: E402  (puts the repository on the path)
import train_tcn as tt  # noqa: E402

from possession import external as ex  # noqa: E402
from possession import tcn  # noqa: E402
from possession.models import combine, design_names, save  # noqa: E402
from possession.simulate import DegradeParams  # noqa: E402

NETWORKS = 5
CONFIRM_SEED = 97


def scored(models, Xs, pool=None):
    """[(dead, home)]: possession.tcn.scores of each model on each half, pairwise and in order,
    in the pool's worker processes when one is given. Each result is computed by the same numpy
    code either way, so a pool changes the time, never the numbers."""
    if pool is None:
        return [tcn.scores(model, X) for model, X in zip(models, Xs)]
    return list(pool.map(tcn.scores, models, Xs))


def worker_pool(workers):
    """A pool of worker processes for scored(), or None for one process. The workers start from a
    fork server, a clean process, so the GPU state of the process that fits never reaches them
    (where there is no fork server, as on Windows, they are spawned afresh)."""
    if workers <= 1:
        return None
    method = "forkserver" if "forkserver" in multiprocessing.get_all_start_methods() else "spawn"
    return concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context(method))


def run(train, sequences, held_out, confirm, matches, out_dir, start, seeds=2, step=8,
        confirm_seed=CONFIRM_SEED, networks=NETWORKS, settings=None, device="cpu", workers=1):
    """The whole comparison on built data; see the module docstring. train is unused but kept
    beside sequences for the signature train_tcn.run has. Returns {'verdict': the adopted name or
    None, 'summaries': {seed: {name: fit_decoder's summary}}}."""
    settings = dict(tt.TRAINING if settings is None else settings)
    reference = "v1+tcn"
    alone = "v1+tcn{}".format(networks)
    wide = alone + tm.GRID_SUFFIX["wide"]
    grounds = ((tm.EVAL_SEED, held_out), (confirm_seed, confirm))
    names = design_names("v1")
    # parts[seed][match][n]: the n-th network's (dead, home) on each held-out half of the match.
    parts = {seed: {m: [] for m in matches} for seed, _ in grounds}
    pool = worker_pool(workers)
    try:
        for k in matches:
            halves = [half for m in matches if m != k for half in sequences[m]]
            fitted = [tt.fit_network(halves, dict(settings, seed=s), device, names)
                      for s in range(networks)]
            jobs = [(model, seed, X) for model in fitted for seed, part in grounds
                    for X in tm.halves_of(part[k], "X")]
            results = iter(scored([j[0] for j in jobs], [j[2] for j in jobs], pool))
            for _ in fitted:
                for seed, part in grounds:
                    parts[seed][k].append([next(results) for _ in tm.halves_of(part[k], "X")])
            print("{} networks held out on {}: fitted on {} halves of the other matches and "
                  "applied ({:.0f} s so far)".format(networks, k, len(halves),
                                                     time.time() - start))
    finally:
        if pool is not None:
            pool.shutdown()

    def one(seed):
        return {m: parts[seed][m][0] for m in matches}

    def averaged(seed):
        return {m: [combine([parts[seed][m][n][i] for n in range(networks)])
                    for i in range(len(parts[seed][m][0]))] for m in matches}

    rule = tm.rules(held_out, matches)
    print()
    tt.report_frames("one network", one(tm.EVAL_SEED), held_out, matches, rule)
    tt.report_frames("{} networks averaged".format(networks), averaged(tm.EVAL_SEED), held_out,
                     matches, rule)

    decoders, summaries = {}, {}
    for seed, part in grounds:
        evidence = {name: {m: [tt.log_probabilities(*p) for p in halves]
                           for m, halves in source.items()}
                    for name, source in ((reference, one(seed)), (alone, averaged(seed)))}
        evidence[wide] = evidence[alone]
        summaries[seed] = {}
        for name in (reference, alone, wide):
            axes = tm.decoder_axes("wide" if name == wide else "base")
            print("\n----- {}, held-out halves degraded with seed {} ({:.0f} s so far) -----"
                  .format(name, seed, time.time() - start))
            setting, practice_range, codes, summary = tm.fit_decoder(part, evidence[name],
                                                                     matches, axes)
            if seed == tm.EVAL_SEED:
                tm.report_errors(part, codes, matches)
                decoders[name] = (setting, practice_range, axes)
            summaries[seed][name] = summary

    passing = {}
    for seed, _ in grounds:
        print("\n===== the adoption rule on the held-out halves of seed {} =====".format(seed))
        passing[seed] = [name for name in (alone, wide) if tm.compare(
            {reference: summaries[seed][reference], name: summaries[seed][name]}, matches,
            reference)]
    adopted = [name for name in (alone, wide) if all(name in passing[s] for s, _ in grounds)]
    verdict = max(adopted, key=lambda name: summaries[tm.EVAL_SEED][name]["pooled"][1.0][0]) \
        if adopted else None
    print("\nM11m ensemble verdict: {} (passing at seed {}: {}; at seed {}: {})".format(
        "adopt " + verdict if verdict else "keep " + reference,
        tm.EVAL_SEED, ", ".join(passing[tm.EVAL_SEED]) or "none",
        confirm_seed, ", ".join(passing[confirm_seed]) or "none"))

    final = [tt.fit_network([half for m in matches for half in sequences[m]],
                            dict(settings, seed=s), device, names) for s in range(networks)]
    os.makedirs(out_dir, exist_ok=True)
    for name in (alone, wide):
        setting, practice_range, axes = decoders[name]
        models = {"design": "v1", "tcn": final, "decoder": setting,
                  "fitted_on": {"matches": matches, "seeds": seeds, "step": step,
                                "label_tolerance_s": 1.0, "pitch": list(tm.PITCH),
                                "decoder_grid": {k: list(v) for k, v in axes.items()},
                                "practice_range": practice_range,
                                "simulator": asdict(DegradeParams()),
                                "confirm_seed": confirm_seed,
                                "network": dict(settings, channels=tcn.CHANNELS,
                                                kernel=tcn.KERNEL,
                                                dilations=list(tcn.DILATIONS),
                                                seeds=list(range(networks)))}}
        out = os.path.join(out_dir, "possession_{}.json".format(name))
        save(models, out)
        print("wrote {} ({:.0f} s so far)".format(os.path.relpath(out, tm.ROOT),
                                                  time.time() - start))
    return {"verdict": verdict, "summaries": summaries}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--networks", type=int, default=NETWORKS)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--step", type=int, default=8)
    parser.add_argument("--confirm-seed", type=int, default=CONFIRM_SEED)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--out-dir", default=os.path.join(tm.ROOT, "models"))
    args = parser.parse_args()
    if args.networks < 2:
        parser.error("--networks must be 2 or more: one network is the reference")
    if args.confirm_seed == tm.EVAL_SEED or 0 <= args.confirm_seed < args.seeds:
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
    sequences = {}
    train, held_out = tm.build(matches, args.seeds, args.step, ["v1"], (), None,
                               sequences=sequences)
    _, confirm = tm.build(matches, 0, args.step, ["v1"], (), None, eval_seed=args.confirm_seed)
    print("built whole halves in {:.0f} s: {} training halves".format(
        time.time() - start, sum(len(halves) for halves in sequences.values())))
    view = {m: dict(part, X=part["X"]["v1"], slow=part["slow"]["v1"])
            for m, part in held_out.items()}
    confirm_view = {m: dict(part, X=part["X"]["v1"], slow=part["slow"]["v1"])
                    for m, part in confirm.items()}
    run(train["v1"], sequences, view, confirm_view, matches, args.out_dir, start, args.seeds,
        args.step, args.confirm_seed, args.networks, device=device, workers=args.workers)
    print("\ndone in {:.0f} s".format(time.time() - start))
    return 0


if __name__ == "__main__":
    sys.exit(main())
