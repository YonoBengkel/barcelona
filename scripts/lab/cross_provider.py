"""Lab, after the competition: the submitted pipeline on another provider's tracking.

The submitted pipeline scored 0.698 held out on the practice ground (IDSSE: DFL's optical tracking
of seven Bundesliga matches, degraded by possession.simulate) and 0.526 on the competition's match.
This job asks how it travels to tracking it has never seen: SkillCorner's broadcast tracking of 20
A-League matches, judged by SkillCorner's own frame-by-frame team possession
(possession.skillcorner, MIT). Another provider, league, camera and definition of possession, as
the competition's match was.

Pipelines, all scored with the organisers' metric:

- as submitted: models/possession_v1+tcn+copies8.json, network and decoder as they were sent;
- decoder re-chosen: the same network's evidence, the decoder's base grid chosen for each match on
  the other 19 (the practice ground's nesting), to see how much of a drop recalibrating wins back;
- baseline_smooth50: the organisers' baseline on the same tracking, a scale for the others (it
  scored 0.2433 on the competition's match).

Each pipeline reads the tracking two ways, as SkillCorner gives it (extrapolated positions included)
and detected positions only; and is scored two ways, on every frame of each half and on tracked
frames only (the frames with no tracking at all, mostly replays and close-ups, left ungraded).
spellF1 is pooled over the matches with a 95% interval from resampling them, macroF1 beside it.
Possessions per ten minutes and their mean length come last, SkillCorner's beside each pipeline's.

The network runs as matrix products here (fast_scores), which equal possession.tcn's
order-fixed sums up to rounding; the byte-for-byte guarantee mattered for the submission only.

Public data only. CPU only, a few minutes per match on one core: a job for Kaggle.

    python scripts/lab/cross_provider.py [--matches 20] [--workers N]
"""

import argparse
import itertools
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_models as tm  # noqa: E402  (puts the repository on the path)
import train_tcn_ensemble as te  # noqa: E402

from possession import practice  # noqa: E402
from possession import skillcorner as sc  # noqa: E402
from possession import tcn  # noqa: E402
from possession.decode import decode  # noqa: E402
from possession.features import compute  # noqa: E402
from possession.models import combine, design, load, log_sigmoid  # noqa: E402
from possession.predictors import nearest_team, smooth  # noqa: E402

pm = practice.pm
MODELS = os.path.join(tm.ROOT, "models", "possession_v1+tcn+copies8.json")
READINGS = {"as given": False, "detected only": True}
SCORINGS = ("every frame", "tracked frames")
PIPELINES = ("as submitted", "decoder re-chosen", "baseline_smooth50")
BASELINE_WINDOW = 50
DRAWS = 1000
REFERENCES = ("references: practice ground held out 0.698 (M11n); the competition's match 0.526, "
              "baseline_smooth50 0.2433 there")


def fast_convolve(x, layer, dilation=1):
    """possession.tcn.convolve as one matrix product per tap: equal up to rounding, much faster."""
    weight = np.asarray(layer["weight"], dtype=np.float64)
    taps = weight.shape[2]
    out = np.repeat(np.asarray(layer["bias"], dtype=np.float64)[:, None], x.shape[1], axis=1)
    for k in range(taps):
        out += weight[:, :, k] @ tcn.shifted(x, (k - (taps - 1) // 2) * dilation)
    return out


def fast_scores(model, X):
    """possession.tcn.scores by fast_convolve: (dead, home) log-odds per frame."""
    h = fast_convolve(tcn.inputs(model, X), model["input"])
    for block in model["blocks"]:
        r = fast_convolve(tcn.relu(h), block["dilated"], block["dilation"])
        r = fast_convolve(tcn.relu(r), block["pointwise"])
        h = h + r
    out = fast_convolve(tcn.relu(h), model["output"])
    return out[0].copy(), out[1].copy()


def evidence(models, X):
    """(frames, 3) log-probabilities of H, A, D from the file's network or networks, as
    possession.models.log_state_probabilities gives them; for a file that holds networks only."""
    if "dead" in models or "stack" in models:
        raise ValueError("fast evidence covers files holding networks alone")
    networks = models["tcn"] if isinstance(models["tcn"], list) else [models["tcn"]]
    dead, home = combine([fast_scores(network, X) for network in networks])
    alive = log_sigmoid(-dead)
    return np.column_stack([alive + log_sigmoid(home), alive + log_sigmoid(-home),
                            log_sigmoid(dead)])


def match_parts(match, models_path=MODELS):
    """{reading: [per half: 'log_p', 'labels', 'tracked', 'submitted', 'baseline']} for one
    match; both readings come from one parse of its tracking (sc.load_match caches it)."""
    models = load(models_path)
    out = {}
    for reading, detected_only in READINGS.items():
        info, periods = sc.load_match(match, detected_only=detected_only)
        halves = []
        for period in periods:
            features = compute(period["half"], info["pitch"])[0]
            X = design(features, models.get("design", "v1")).astype(np.float32)
            log_p = evidence(models, X)
            halves.append({"log_p": log_p, "labels": period["labels"],
                           "tracked": period["tracked"],
                           "submitted": decode(log_p, models["decoder"]),
                           "baseline": smooth(nearest_team(period["half"]), BASELINE_WINDOW)})
        out[reading] = halves
    return match, out


def truth(halves, scoring):
    """The labels as codes with an X between halves; X too on untracked frames for that scoring."""
    parts = []
    for half in halves:
        codes = practice.codes(half["labels"])
        if scoring == "tracked frames":
            codes = np.where(half["tracked"], codes, pm.X).astype(codes.dtype)
        parts.append(codes)
    return practice.joined(parts, gap=pm.X)


def predicted(halves, key):
    return practice.joined([practice.codes(half[key]) for half in halves], gap=pm.X)


def grid_counts(job):
    """[setting][scoring] counts of every decoder setting on one match, for both scorings."""
    log_ps, truths, settings = job
    return [{scoring: practice.counts(gt, pred) for scoring, gt in truths.items()}
            for pred in tm.decoded(log_ps, settings)]


def interval(parts, rng, draws=DRAWS):
    """95% interval of the pooled spellF1 of parts (one counts() per match), matches resampled."""
    values = [practice.pooled([parts[i] for i in rng.integers(len(parts), size=len(parts))])[0]
              for _ in range(draws)]
    return np.percentile(values, [2.5, 97.5])


def run(matches, workers=1, start=0.0):
    """Every pipeline, reading and scoring; returns {(reading, scoring, pipeline): result}."""
    pool = te.worker_pool(workers)
    mapped = pool.map if pool else map
    try:
        parts = dict(mapped(match_parts, matches))
        print("parsed and weighed {} matches ({:.0f} s so far)".format(len(parts),
                                                                      time.time() - start))
        settings = tuple(itertools.product(*tm.decoder_axes("base").values()))
        tables = {}
        for reading in READINGS:
            jobs = [([half["log_p"] for half in parts[m][reading]],
                     {scoring: truth(parts[m][reading], scoring) for scoring in SCORINGS},
                     settings) for m in matches]
            tables[reading] = dict(zip(matches, mapped(grid_counts, jobs)))
            print("decoder grid of {} settings on the '{}' reading ({:.0f} s so far)".format(
                len(settings), reading, time.time() - start))
    finally:
        if pool is not None:
            pool.shutdown()

    rng = np.random.default_rng(0)
    results, chosen = {}, {}
    for reading in READINGS:
        for scoring in SCORINGS:
            table = [{m: tables[reading][m][k][scoring] for m in matches}
                     for k in range(len(settings))]
            picks = {m: tm.best(table, [o for o in matches if o != m]) for m in matches}
            chosen[(reading, scoring)] = picks
            for pipeline in PIPELINES:
                counts, rates = [], []
                for m in matches:
                    halves = parts[m][reading]
                    gt = truth(halves, scoring)
                    if pipeline == "decoder re-chosen":
                        codes = next(tm.decoded([h["log_p"] for h in halves],
                                                [settings[picks[m]]]))
                    else:
                        codes = predicted(halves, "submitted" if pipeline == "as submitted"
                                          else "baseline")
                    counts.append(practice.counts(gt, codes))
                    rates.append(tm.spell_rate(codes, gt))
                spell, macro = practice.pooled(counts)
                low, high = interval(counts, rng)
                per_match = [practice.pooled([c])[0] for c in counts]
                rate, length = np.array(rates).mean(axis=0)
                results[(reading, scoring, pipeline)] = {
                    "spellF1": spell, "interval": (low, high), "macroF1": macro,
                    "per_match": per_match, "rate": rate, "length": length}
    labels = {}
    for scoring in SCORINGS:
        rates = [tm.spell_rate(truth(parts[m]["as given"], scoring),
                               truth(parts[m]["as given"], scoring)) for m in matches]
        labels[scoring] = np.array(rates).mean(axis=0)
    report(results, labels, chosen, settings, matches, parts)
    return results


def report(results, labels, chosen, settings, matches, parts):
    print("\n===== the submitted pipeline on SkillCorner's 20 A-League matches =====")
    print(REFERENCES)
    tracked = np.mean([np.mean(h["tracked"]) for m in matches for h in parts[m]["as given"]])
    print("frames carrying tracking: {:.1%}".format(tracked))
    print("{:<14} {:<15} {:<18} {:>26} {:>8} {:>10} {:>8}".format(
        "reading", "scoring", "pipeline", "spellF1 [95% interval]", "macroF1", "poss/10min",
        "mean s"))
    for (reading, scoring, pipeline), r in results.items():
        print("{:<14} {:<15} {:<18} {:>8.4f} [{:.3f}, {:.3f}] {:>8.4f} {:>10.1f} {:>8.1f}".format(
            reading, scoring, pipeline, r["spellF1"], r["interval"][0], r["interval"][1],
            r["macroF1"], r["rate"], r["length"]))
    for scoring, (rate, length) in labels.items():
        print("SkillCorner's labels, {}: {:.1f} possessions per 10 min, mean {:.1f} s".format(
            scoring, rate, length))
    for (reading, scoring), picks in chosen.items():
        counts = {}
        for m in matches:
            counts[settings[picks[m]]] = counts.get(settings[picks[m]], 0) + 1
        print("decoder settings chosen per match, {}, {}: {}".format(reading, scoring, "; ".join(
            "{} x{}".format(dict(zip(tm.DECODER_AXES, s)), n)
            for s, n in sorted(counts.items(), key=lambda item: -item[1]))))
    for pipeline in ("as submitted", "decoder re-chosen"):
        r = results[("as given", "every frame", pipeline)]
        print("{} (as given, every frame), spellF1 per match: {}".format(pipeline, ", ".join(
            "{} {:.3f}".format(m, v) for m, v in zip(matches, r["per_match"]))))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--matches", type=int, default=len(sc.MATCHES),
                        help="the first N of the 20 matches, for a quick look")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()
    start = time.time()
    matches = list(sc.MATCHES[:args.matches])
    fetched = sc.fetch(matches)
    print("SkillCorner open data at {}: {} files fetched, {} matches ({:.0f} s)".format(
        sc.COMMIT[:12], fetched, len(matches), time.time() - start))
    run(matches, args.workers, start)
    print("\ndone in {:.0f} s".format(time.time() - start))
    return 0


if __name__ == "__main__":
    sys.exit(main())
