Status: plan. Built so far: in `possession/`, `safety`, `external` (IDSSE reader, cache and label
alignment), `atd` (the match's tracking reader), `profile` and `simulate` (IDSSE degraded to
automatic-tracking quality), `practice` (the practice ground and its official scoring),
`predictors` (the organisers' baseline on our arrays), `features` (33 per-frame features from
positions alone), `models` (logistic regressions and gradient-boosted trees for the dead-ball and
team calls, both applied with numpy alone),
`decode` (the three-state hidden Markov model and its Viterbi decoder), `submission` (the CSV
writer) and `diagnose` (errors by ball state and by possession); M11l's two candidates, `tcn` (a
temporal convolutional network applied with numpy alone, adopted on 2026-11-12) and `stacking` (a
stage two of trees on the first stage's log-odds, not adopted); the scripts under "How to run";
and the Kaggle runner in `kaggle/`. Rewrite this file from the code as later milestones land.

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
   │  possession.models.log_state_probabilities: P(dead) and P(home | alive) from
   │  possession.models.scores. Since M11l (2026-11-12) both come from a temporal convolutional
   │  network (possession.tcn) in models/possession_v1+tcn.json, which reads the design columns
   │  over about 10 s either side of the frame. It is applied with numpy alone, summed tap by tap
   │  and input by input in a fixed order, so the result does not depend on the CPU's BLAS. The
   │  previous model, gradient-boosted trees for both calls
   │  (models/possession_v1+trees-wide+deadtrees.json), is walked tree by tree in a fixed order,
   │  and a logistic regression is summed column by column
   ▼
per-frame evidence: log-probability of H, A, D
   │  possession.decode: three-state hidden Markov model, Viterbi. A path pays team_switch for
   │  H <-> A and dead_switch for each stoppage; dead_bias shifts the D evidence. The three
   │  numbers sit in the model file beside the models. Two optional ones, team_min and
   │  dead_min, hold every spell to a minimum length in frames (M11j's 'mindur' grid; the
   │  adopted model has neither)
   ▼
labels per graded half ──► possession.submission: X outside the graded halves, LF line endings
   │  scripts/make_submission.py validates the file with possession_metrics.validate_submission
   ▼
evaluation/results/possession_v1.csv (gitignored; regenerated, never committed)

external/idsse  (public IDSSE tracking and events of seven Bundesliga matches, gitignored)
   │  possession.external.read_positions, read_events
   ▼
same per-frame arrays + raw labels from DFL's BallStatus and BallPossession
   │  possession.external.controlled_labels: keeps a possession change only where the new team
   │  restarts play or acts with control, the challenge's definition (raw DFL changes on any touch)
   │  possession.simulate.degrade  ──►  same arrays at automatic-tracking quality: the ball hidden
   │  in risk-weighted gaps, players dropped, drifting noise on the ball, no ball height. Its
   │  parameters are calibrated against possession.profile run on the match's tracking
   │  (aggregates only), the players' speeds included since M11g.
   ▼
possession.practice.cases ──► scripts/eval_external.py: runs every predictor on every degraded
half, scores it with evaluation/possession_metrics.py (pooled over matches and per match), and
prints the organisers' published full-match baseline scores beside ours as a fidelity check.
   │
   └─► scripts/train_models.py: holds out each match in turn. It fits the two models on the
       other six and keeps their probabilities on the held-out match. It then scores every setting
       of the decoder's grid on every match with possession.practice.counts, which pools exactly
       as possession_metrics.evaluate does. Each match's setting is chosen on the other six (a
       held-out score for the whole pipeline); the setting best on all seven is written out.
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
scripts/train_models.py        # refit the trees (the previous model, M12) and choose their
                               # decoder, match by match, into
                               # models/possession_v1+trees-wide+deadtrees.json; its options set
                               # other designs, model kinds, decoder grids and practice grounds
                               # against it by a rule fixed in the code
scripts/eval_external.py       # score predictors on the practice ground
scripts/calibrate_simulator.py # fit DegradeParams to the match's profile (aggregates in the code)
scripts/feature_report.py      # signal of each feature (ROC AUC)
scripts/simulate_report.py     # IDSSE profile before and after degradation
scripts/idsse_label_report.py  # raw against aligned possession statistics
scripts/train_stacking.py      # M11l: a stage two of trees against the trees (not adopted)
scripts/train_tcn.py           # M11l: a temporal convolutional network, alone and beside the
                               # trees, against them; writes models/possession_v1+tcn.json, the
                               # adopted model. Needs the GPU accelerator, and the runner
                               # installs PyTorch for it from requirements-torch.txt
scripts/train_tcn_ensemble.py  # M11m: an ensemble of networks against the adopted one, the same
                               # way (not adopted); GPU and PyTorch as above
```

M11l's candidates were set against the trees frozen in M12 by the same rule, at the held-out seed
and again at a second one; the network alone was adopted. `possession.models.scores` applies
whichever models a file holds: the trees alone (the frozen file, unchanged), a network
(`possession.tcn`), both averaged, or a stage two on top (`possession.stacking`). All of them run
with numpy alone at submission time.

Jobs that read the match data run on the team's laptop only, and print aggregates only:

```bash
pip install -r requirements.txt
python scripts/make_submission.py    # the submission: evaluation/results/possession_v1.csv
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

`scripts/make_submission.py` applies `models/possession_v1+tcn.json` (adopted on 2026-11-12 by
the rule in `scripts/train_tcn.py`; `--models` picks another file, such as the previous trees).
The file must carry a `decoder` section, written by a Kaggle run of `scripts/train_tcn.py` or
`scripts/train_models.py`; without one the script stops before it reads the match.

## Application layers

| Layer | This project |
|---|---|
| Front end | none |
| API | none |
| Data | files only: the match bundle and public datasets, both gitignored; committed model parameters |
| Auth | none |
| Hosting | none |
| Compute | Kaggle CPU notebooks for jobs on public data, started by hand through `kaggle/`, and a GPU one to fit M11l's network; the team's laptop, on CPU, for every job that reads the match data, which never leaves it. Single process; the submission is deterministic |
| CI/CD | none; `.githooks/pre-commit` refuses commits that contain match data, and every Kaggle run runs the tests before its job |
| Row-level security | not applicable |
| Rate limiting | not applicable |
| Cache and CDN | per-frame artefacts cached under `data/` and `external/` |
| Scaling | not applicable: one match |
| Error tracking and logs | stdout with aggregates only; exception messages are withheld when the match data is read. Kaggle runs keep their full log in `runs/` |
| Recovery | git history plus a deterministic pipeline: re-running reproduces every output |
