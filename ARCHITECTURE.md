Status: plan. Built so far: in `possession/`, `safety`, `external` (IDSSE reader, cache and label
alignment), `atd` (the match's tracking reader), `profile` and `simulate` (IDSSE degraded to
automatic-tracking quality), `practice` (the practice ground and its official scoring),
`predictors` (the organisers' baseline on our arrays), `features` (33 per-frame features from
positions alone) and `models` (the dead-ball and team logistic regressions, applied with numpy);
the scripts under "How to run"; and the Kaggle runner in `kaggle/`. The decoder and the
submission writer are still planned; rewrite this file from the code as they land.

# Architecture

## Components and data flow

```
data/tactical  (match bundle, gitignored)
   │  possession.atd.read_atd, halves
   ▼
per-frame arrays: ball xy or NaN · object xy · team
   │  possession.features.compute: ball gaps and where the ball was last seen, player speeds,
   │  nearest player and crowding per team, ball direction and team shape along each team's
   │  attacking direction (inferred from where its keeper stands); positions only, so the same
   │  on IDSSE and on the match
   ▼
per-frame features ──► possession.models.design: log scale for skewed features, plus rolling
   │  means over 2.5-5 s for context
   │  possession.models.state_probabilities: P(dead) and P(home | alive) from two logistic
   │  regressions in models/possession_v1.json, summed column by column in a fixed order so the
   │  result does not depend on the CPU's BLAS
   ▼
per-frame evidence: log-likelihood of H, A, D
   │  possession.decode  (HMM / semi-Markov, Viterbi)
   ▼
labels ──► possession.submission.write (LF line endings) ──► evaluation/validate_submission.py

external/idsse  (public IDSSE tracking and events of seven Bundesliga matches, gitignored)
   │  possession.external.read_positions, read_events
   ▼
same per-frame arrays + raw labels from DFL's BallStatus and BallPossession
   │  possession.external.controlled_labels: keeps a possession change only where the new team
   │  restarts play or acts with control, the challenge's definition (raw DFL changes on any touch)
   │  possession.simulate.degrade  ──►  same arrays at automatic-tracking quality: the ball hidden
   │  in risk-weighted gaps, players dropped, drifting noise, no ball height. Its parameters are
   │  calibrated against possession.profile run on the match's tracking (aggregates only).
   ▼
possession.practice.cases ──► scripts/eval_external.py: runs every predictor on every degraded
half, scores it with evaluation/possession_metrics.py (pooled over matches and per match), and
prints the organisers' published full-match baseline scores beside ours as a fidelity check.
Models and parameters are chosen here; the match itself never chooses anything.
```

Every script that reads the match data runs through `possession.safety.run_safely` and prints
aggregates only (see [AGENTS.md](AGENTS.md)).

## Folder map

| Path | What | Committed |
|---|---|---|
| `possession/` | the pipeline package | yes |
| `scripts/` | entry points | yes |
| `tests/` | tests on synthetic or public data | yes |
| `models/` | fitted model weights as JSON, the only artefact the submission loads | yes |
| `kaggle/` | the runner for jobs on public data: one notebook cell, and the script it calls | yes |
| `runs/` | log and manifest of each Kaggle run: commit, command, versions, output digests | yes |
| `games.py`, `evaluation/`, `starter/` | the organisers' starter kit, unchanged | yes |
| `docs/challenge/` | the organisers' documents | yes |
| `data/` | match bundle and per-frame artefacts derived from it | no |
| `external/` | downloaded public datasets | no |
| `evaluation/results/` | our submission files, scored locally against the public ground truth | no, except the organisers' baselines |

## How to run

Jobs on public data run on Kaggle, through the runner in `kaggle/`, which builds the pinned
environment, downloads IDSSE, runs the tests and then the job, and returns its outputs, log and
manifest as one zip ([kaggle/README.md](kaggle/README.md)). Set the job as `COMMAND` in the
notebook cell:

```bash
scripts/train_models.py        # fit the models leave-one-match-out, write models/possession_v1.json
scripts/eval_external.py       # score predictors on the practice ground
scripts/calibrate_simulator.py # fit DegradeParams to the match's profile (aggregates in the code)
scripts/feature_report.py      # signal of each feature (ROC AUC)
scripts/simulate_report.py     # IDSSE profile before and after degradation
scripts/idsse_label_report.py  # raw against aligned possession statistics
```

Jobs that read the match data run on the team's laptop only, and print aggregates only:

```bash
pip install -r requirements.txt
python scripts/check_baselines.py    # rebuilds the organisers' baselines and compares them
python scripts/atd_profile.py        # profile of the match's tracking
python scripts/data_properties.py    # structure of the data bundle
python -m pytest                     # synthetic data only; also runs in every Kaggle run
```

`scripts/fetch_idsse.py` downloads the IDSSE files from the Hugging Face mirror
`pysport/idsse-data` into `external/idsse/`, names unchanged, at a pinned revision, and checks
every file's size and digest. The first read of an IDSSE match parses its XML (about 20 s on the
laptop) and caches the arrays under `external/idsse/cache/`; later reads take a fraction of a
second.

Planned: `scripts/make_submission.py` (match data to CSV, on the laptop) and the decoder's
parameter search (on Kaggle).

## Application layers

| Layer | This project |
|---|---|
| Front end | none |
| API | none |
| Data | files only: the match bundle and public datasets, both gitignored; committed model parameters |
| Auth | none |
| Hosting | none |
| Compute | Kaggle CPU notebooks for jobs on public data, started by hand through `kaggle/`; the team's laptop, on CPU, for every job that reads the match data, which never leaves it. Single process, deterministic |
| CI/CD | none; `.githooks/pre-commit` refuses commits that contain match data, and every Kaggle run runs the tests before its job |
| Row-level security | not applicable |
| Rate limiting | not applicable |
| Cache and CDN | per-frame artefacts cached under `data/` and `external/` |
| Scaling | not applicable: one match |
| Error tracking and logs | stdout with aggregates only; exception messages are withheld when the match data is read. Kaggle runs keep their full log in `runs/` |
| Recovery | git history plus a deterministic pipeline: re-running reproduces every output |
