Status: plan. Built so far: in `possession/`, `safety`, `external` (IDSSE reader and label
alignment), `atd` (the match's tracking reader), `profile` and `simulate` (IDSSE degraded to
automatic-tracking quality); and the scripts under "How to run". Features, the decoder and the
submission writer are still planned; rewrite this file from the code as they land.

# Architecture

## Components and data flow

```
data/tactical  (match bundle, gitignored)
   │  possession.atd.read_atd, halves
   ▼
per-frame arrays: ball xy or NaN · object xy · team · speed
   │  possession.features
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
scripts/eval_external.py (planned): scores every variant with evaluation/possession_metrics.py and
chooses models and parameters. The choices are committed; the match itself never chooses anything.
```

Every script that reads the match data runs through `possession.safety.run_safely` and prints
aggregates only (see [AGENTS.md](AGENTS.md)).

## Folder map

| Path | What | Committed |
|---|---|---|
| `possession/` | the pipeline package | yes |
| `scripts/` | entry points | yes |
| `tests/` | tests on synthetic or public data | yes |
| `games.py`, `evaluation/`, `starter/` | the organisers' starter kit, unchanged | yes |
| `docs/challenge/` | the organisers' documents | yes |
| `data/` | match bundle and per-frame artefacts derived from it | no |
| `external/` | downloaded public datasets | no |
| `evaluation/results/` | our submission files, scored locally against the public ground truth | no, except the organisers' baselines |

## How to run

```bash
pip install -r requirements.txt
python -m pytest                     # synthetic and public data only
python scripts/check_baselines.py    # reads the match data; prints aggregates only
python scripts/idsse_label_report.py # public data: raw against aligned possession statistics
python scripts/atd_profile.py        # reads the match data; prints aggregates only
python scripts/simulate_report.py    # public data: IDSSE profile before and after degradation
```

The IDSSE files come from the Hugging Face mirror `pysport/idsse-data` into `external/idsse/`,
names unchanged (`possession.external.paths` lists them).

Planned: `scripts/make_submission.py` (match data to CSV) and `scripts/eval_external.py`
(scores variants on the practice matches).

## Application layers

| Layer | This project |
|---|---|
| Front end | none |
| API | none |
| Data | files only: the match bundle and public datasets, both gitignored; committed model parameters |
| Auth | none |
| Hosting | none; runs on the team's machine |
| Compute | local CPU, single process, deterministic |
| CI/CD | none; `.githooks/pre-commit` refuses commits that contain match data |
| Row-level security | not applicable |
| Rate limiting | not applicable |
| Cache and CDN | per-frame artefacts cached under `data/` and `external/` |
| Scaling | not applicable: one match |
| Error tracking and logs | stdout with aggregates only; exception messages are withheld when the match data is read |
| Recovery | git history plus a deterministic pipeline: re-running reproduces every output |
