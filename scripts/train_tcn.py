"""M11l: a temporal convolutional network for the per-frame calls, alone and beside the trees.

Three pipelines are fitted and scored on the practice ground, each match held out in turn exactly
as scripts/train_models.py does it, with the same decoder search, nested by match:

- the reference, the frozen pipeline: tree models for both calls (v1+trees-wide+deadtrees), fitted
  here again, so its held-out scores reproduce the frozen model's;
- 'v1+tcn': the network of possession.tcn alone, fitted with PyTorch on the whole training halves
  (both training copies of every half, each as degraded and with home and away exchanged);
- 'v1+trees-wide+deadtrees+tcn': the trees' and the network's log-odds averaged
  (possession.models.combine).

The network's settings (TRAINING, and the architecture in possession.tcn) were fixed before the
run and are never tuned on a held-out match. Its held-out log-odds come from possession.tcn.scores,
the numpy code the submission would run.

The adoption rule is train_models.compare's, set before the run: held-out spellF1 at +/-1 s at
least 0.01 higher, better on at least 5 of the 7 matches, no loss at +/-0.5 s or +/-2 s, macroF1
down by at most 0.005. Two candidates raise the chance that one passes by luck, so a candidate is
adopted only if it passes on the held-out halves degraded with EVAL_SEED and again, with the same
fitted folds and the decoder chosen the same nested way, on the held-out halves degraded with a
second seed (--confirm-seed) that no other copy uses. If both candidates pass twice, the one with
the higher held-out spellF1 at +/-1 s is adopted.

Last, the network is fitted on all seven matches and both candidates are written to
models/possession_<name>.json, whatever the verdict. A file is committed only if adopted.

The run of 2026-11-12 (runs/20261112T1010Z-train_tcn) adopted 'v1+tcn': held-out spellF1 at +/-1 s
0.6800 against 0.6498 for the trees, better on 6 of 7 matches, and 0.6742 against 0.6494 at the
second seed. Both candidates passed twice; the network alone scored higher at EVAL_SEED.
models/possession_v1+tcn.json is that run's file, which scripts/make_submission.py applies.

Public data only; reads no match data. A job for Kaggle with a GPU (kaggle/README.md); the runner
installs PyTorch for it from requirements-torch.txt. Nothing the submission runs needs PyTorch.

    python scripts/train_tcn.py [--seeds 2] [--step 8] [--confirm-seed 98] [--out-dir models]
"""

import argparse
import os
import sys
import time
from dataclasses import asdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_models as tm  # noqa: E402  (puts the repository on the path)

from possession import external as ex  # noqa: E402
from possession import tcn  # noqa: E402
from possession.models import combine, design_names, log_sigmoid, save, score  # noqa: E402
from possession.simulate import DegradeParams  # noqa: E402

# How the network is fitted, fixed before the run (M11l). One step is one whole training half, as
# degraded or with home and away exchanged, drawn at random: the network sees exactly the zeros
# beyond the half that it sees when applied, and no crop edges.
TRAINING = {"steps": 3000, "learning_rate": 1e-3, "weight_decay": 1e-4, "dropout": 0.1,
            "clip": 1.0, "seed": 0}


def mirrored_labels(labels):
    """The labels of a half with home and away exchanged: H and A swap, D stays."""
    labels = np.asarray(labels)
    return np.where(labels == "H", "A", np.where(labels == "A", "H", labels))


def normaliser(halves):
    """(mean, scale) of every design column over every frame of the training halves, as degraded
    and mirrored, a missing value ignored; a column that never varies gets a scale of 1."""
    total = square = count = 0.0
    for half in halves:
        for X in (half["X"], half["mirrored"]):
            X = np.asarray(X, dtype=np.float64)
            valid = np.isfinite(X)
            values = np.where(valid, X, 0.0)
            total = total + values.sum(axis=0)
            square = square + (values * values).sum(axis=0)
            count = count + valid.sum(axis=0)
    mean = np.where(count > 0, total / np.maximum(count, 1), 0.0)
    variance = np.where(count > 1, square / np.maximum(count, 1) - mean * mean, 0.0)
    scale = np.sqrt(np.maximum(variance, 0.0))
    return mean, np.where(scale > 1e-9, scale, 1.0)


def network(in_channels, settings):
    """The PyTorch twin of possession.tcn.scores, for fitting."""
    import torch
    from torch import nn

    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            channels, kernel = tcn.CHANNELS, tcn.KERNEL
            self.input = nn.Conv1d(in_channels, channels, 1)
            self.dilated = nn.ModuleList(
                nn.Conv1d(channels, channels, kernel, dilation=d, padding=(kernel - 1) // 2 * d)
                for d in tcn.DILATIONS)
            self.pointwise = nn.ModuleList(nn.Conv1d(channels, channels, 1)
                                           for _ in tcn.DILATIONS)
            self.drop = nn.Dropout(settings["dropout"])
            self.output = nn.Conv1d(channels, 2, 1)

        def forward(self, x):
            h = self.input(x)
            for dilated, pointwise in zip(self.dilated, self.pointwise):
                h = h + pointwise(self.drop(torch.relu(dilated(torch.relu(h)))))
            return self.output(torch.relu(h))

    return Network()


def export(net, mean, scale, names, settings):
    """The JSON-ready network possession.tcn.scores applies: float32 weights as exact floats."""
    def layer(conv):
        return {"weight": conv.weight.detach().double().cpu().numpy().tolist(),
                "bias": conv.bias.detach().double().cpu().numpy().tolist()}

    return {"kind": "tcn", "names": list(names), "mean": [float(v) for v in mean],
            "scale": [float(v) for v in scale], "channels": tcn.CHANNELS, "kernel": tcn.KERNEL,
            "input": layer(net.input),
            "blocks": [{"dilation": int(dilated.dilation[0]), "dilated": layer(dilated),
                        "pointwise": layer(pointwise)}
                       for dilated, pointwise in zip(net.dilated, net.pointwise)],
            "output": layer(net.output), "settings": dict(settings)}


def fit_network(halves, settings, device, names):
    """One network fitted on training halves (entries of build()'s sequences), JSON-ready.

    Each step draws one half, as degraded or mirrored, and takes the dead-ball loss on every frame
    and the team loss on its live frames, both binary cross-entropy on the log-odds.
    """
    import torch
    import torch.nn.functional as F

    torch.manual_seed(settings["seed"])
    rng = np.random.default_rng(settings["seed"])
    mean, scale = normaliser(halves)
    standard = {"mean": mean, "scale": scale}
    data = []
    for half in halves:
        for X, labels in ((half["X"], np.asarray(half["labels"])),
                          (half["mirrored"], mirrored_labels(half["labels"]))):
            x = torch.from_numpy(tcn.inputs(standard, X).astype(np.float32))[None].to(device)
            dead = torch.from_numpy((labels == "D").astype(np.float32)).to(device)
            home = torch.from_numpy((labels == "H").astype(np.float32)).to(device)
            data.append((x, dead, home, dead < 0.5))
    net = network(data[0][0].shape[1], settings).to(device)
    optimiser = torch.optim.AdamW(net.parameters(), lr=settings["learning_rate"],
                                  weight_decay=settings["weight_decay"])
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=settings["steps"])
    losses = []
    net.train()
    for _ in range(settings["steps"]):
        x, dead, home, alive = data[int(rng.integers(len(data)))]
        out = net(x)[0]
        loss = F.binary_cross_entropy_with_logits(out[0], dead)
        if bool(alive.any()):
            loss = loss + F.binary_cross_entropy_with_logits(out[1][alive], home[alive])
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), settings["clip"])
        optimiser.step()
        schedule.step()
        losses.append(float(loss.detach()))
    net.eval()
    window = max(1, min(100, len(losses)))
    print("  network: {} steps on {} halves, training loss {:.4f} over the first {} steps, "
          "{:.4f} over the last {}".format(len(losses), len(data), np.mean(losses[:window]),
                                          window, np.mean(losses[-window:]), window))
    model = export(net.cpu(), mean, scale, names, settings)
    del data, net
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return model


def log_probabilities(dead, home):
    """(frames, 3) log P(H), P(A), P(D) from the two log-odds, as possession.models does it."""
    alive = log_sigmoid(-dead)
    return np.column_stack([alive + log_sigmoid(home), alive + log_sigmoid(-home),
                            log_sigmoid(dead)])


def report_frames(label, parts, held_out, matches, rule):
    """The pooled held-out log loss and accuracy of both calls, as train_models reports them."""
    dead = np.concatenate([d for m in matches for d, _ in parts[m]])
    home = np.concatenate([h for m in matches for _, h in parts[m]])
    labels = np.concatenate([held_out[m]["labels"] for m in matches])
    live = labels != "D"
    tm.report_dead(label, np.exp(log_sigmoid(dead)), labels == "D", rule)
    tm.report_team(label, np.exp(log_sigmoid(home[live])), labels[live] == "H", rule)


def run(train, sequences, held_out, confirm, matches, out_dir, start, seeds=2, step=8,
        confirm_seed=98, settings=None, device="cpu"):
    """The whole comparison on built data; see the module docstring. Returns {'verdict': the
    adopted name or None, 'summaries': {seed: {name: fit_decoder's summary}}}."""
    settings = dict(TRAINING if settings is None else settings)
    rule = tm.rules(held_out, matches)
    dead_label, dead_folds, dead_final = tm.choose("dead", "trees", train, held_out, matches,
                                                   "v1", rule)
    print("chosen dead {} ({:.0f} s)".format(dead_label, time.time() - start))
    team_label, team_folds, team_final = tm.choose("team", "trees-wide", train, held_out,
                                                   matches, "v1", rule)
    print("chosen team {} ({:.0f} s)".format(team_label, time.time() - start))
    reference = tm.variant_name("v1", "trees", "trees-wide")
    alone, both = "v1+tcn", reference + "+tcn"
    grounds = ((tm.EVAL_SEED, held_out), (confirm_seed, confirm))

    trees = {seed: {m: [(score(dead_folds[m], X), score(team_folds[m], X))
                        for X in tm.halves_of(part[m], "X")] for m in matches}
             for seed, part in grounds}
    print("the trees' held-out log-odds at seeds {} ({:.0f} s so far)".format(
        " and ".join(str(seed) for seed, _ in grounds), time.time() - start))
    names = design_names("v1")
    net = {seed: {} for seed, _ in grounds}
    for k in matches:
        halves = [half for m in matches if m != k for half in sequences[m]]
        model = fit_network(halves, settings, device, names)
        for seed, part in grounds:
            net[seed][k] = [tcn.scores(model, X) for X in tm.halves_of(part[k], "X")]
        print("network held out on {}: fitted on {} halves of the other matches and applied "
              "({:.0f} s so far)".format(k, len(halves), time.time() - start))
    print()
    report_frames("trees " + team_label, trees[tm.EVAL_SEED], held_out, matches, rule)
    report_frames("network", net[tm.EVAL_SEED], held_out, matches, rule)
    report_frames("trees and network averaged",
                  {m: [combine([a, b]) for a, b in zip(trees[tm.EVAL_SEED][m],
                                                       net[tm.EVAL_SEED][m])] for m in matches},
                  held_out, matches, rule)

    decoders, summaries = {}, {}
    for seed, part in grounds:
        evidence = {
            reference: {m: [log_probabilities(*p) for p in trees[seed][m]] for m in matches},
            alone: {m: [log_probabilities(*p) for p in net[seed][m]] for m in matches},
            both: {m: [log_probabilities(*combine([a, b]))
                       for a, b in zip(trees[seed][m], net[seed][m])] for m in matches}}
        summaries[seed] = {}
        for name, oof in evidence.items():
            print("\n----- {}, held-out halves degraded with seed {} ({:.0f} s so far) -----"
                  .format(name, seed, time.time() - start))
            setting, practice_range, codes, summary = tm.fit_decoder(part, oof, matches)
            if seed == tm.EVAL_SEED:
                tm.report_errors(part, codes, matches)
                decoders[name] = (setting, practice_range)
            summaries[seed][name] = summary

    passing = {}
    for seed, _ in grounds:
        print("\n===== the adoption rule on the held-out halves of seed {} =====".format(seed))
        passing[seed] = [name for name in (alone, both) if tm.compare(
            {reference: summaries[seed][reference], name: summaries[seed][name]}, matches,
            reference)]
    adopted = [name for name in (alone, both) if all(name in passing[s] for s, _ in grounds)]
    verdict = max(adopted, key=lambda name: summaries[tm.EVAL_SEED][name]["pooled"][1.0][0]) \
        if adopted else None
    print("\nM11l network verdict: {} (passing at seed {}: {}; at seed {}: {})".format(
        "adopt " + verdict if verdict else "keep " + reference,
        tm.EVAL_SEED, ", ".join(passing[tm.EVAL_SEED]) or "none",
        confirm_seed, ", ".join(passing[confirm_seed]) or "none"))

    final = fit_network([half for m in matches for half in sequences[m]], settings, device, names)
    fitted_on = {"matches": matches, "seeds": seeds, "step": step, "label_tolerance_s": 1.0,
                 "pitch": list(tm.PITCH), "scikit_learn": tm.sklearn_version,
                 "decoder_grid": {k: list(v) for k, v in tm.DECODER_AXES.items()},
                 "simulator": asdict(DegradeParams()), "confirm_seed": confirm_seed,
                 "network": dict(settings, channels=tcn.CHANNELS, kernel=tcn.KERNEL,
                                 dilations=list(tcn.DILATIONS))}
    os.makedirs(out_dir, exist_ok=True)
    for name, models in ((alone, {"design": "v1", "tcn": final}),
                         (both, {"design": "v1", "dead": dead_final, "team": team_final,
                                 "tcn": final})):
        setting, practice_range = decoders[name]
        details = dict(fitted_on, practice_range=practice_range)
        if "dead" in models:
            details.update(dead_model=dead_label, team_model=team_label)
        models.update(decoder=setting, fitted_on=details)
        out = os.path.join(out_dir, "possession_{}.json".format(name))
        save(models, out)
        print("wrote {} ({:.0f} s so far)".format(os.path.relpath(out, tm.ROOT),
                                                  time.time() - start))
    return {"verdict": verdict, "summaries": summaries}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--step", type=int, default=8)
    parser.add_argument("--confirm-seed", type=int, default=98)
    parser.add_argument("--out-dir", default=os.path.join(tm.ROOT, "models"))
    args = parser.parse_args()
    if args.confirm_seed == tm.EVAL_SEED or 0 <= args.confirm_seed < args.seeds:
        parser.error("--confirm-seed must differ from EVAL_SEED and from every training seed")

    import torch
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("torch {} on {}".format(torch.__version__, torch.cuda.get_device_name(0)
                                  if device == "cuda" else "the CPU (slow: is the GPU on?)"))
    matches = sorted(ex.MATCHES)
    start = time.time()
    sequences = {}
    train, held_out = tm.build(matches, args.seeds, args.step, ["v1"], (), None,
                               sequences=sequences)
    _, confirm = tm.build(matches, 0, args.step, ["v1"], (), None, eval_seed=args.confirm_seed)
    print("built rows and whole halves in {:.0f} s: dead {}, team {}, {} training halves".format(
        time.time() - start, len(tm.stack(train["v1"], matches, "dead")[1]),
        len(tm.stack(train["v1"], matches, "team")[1]),
        sum(len(halves) for halves in sequences.values())))
    view = {m: dict(part, X=part["X"]["v1"], slow=part["slow"]["v1"])
            for m, part in held_out.items()}
    confirm_view = {m: dict(part, X=part["X"]["v1"], slow=part["slow"]["v1"])
                    for m, part in confirm.items()}
    run(train["v1"], sequences, view, confirm_view, matches, args.out_dir, start, args.seeds,
        args.step, args.confirm_seed, device=device)
    print("\ndone in {:.0f} s".format(time.time() - start))
    return 0


if __name__ == "__main__":
    sys.exit(main())
