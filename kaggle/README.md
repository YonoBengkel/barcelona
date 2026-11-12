# Running jobs on Kaggle

Jobs on public data, such as fitting the models, choosing parameters and scoring the practice
ground, run in a Kaggle notebook, not on the team's laptop. The match data never goes to Kaggle:
the scripts that read it run on the laptop only (see the table below).

## Once

1. **A GitHub token that can only read this repository.** On GitHub: Settings > Developer
   settings > Personal access tokens > Fine-grained tokens > Generate new token.
   - Repository access: only `YonoBengkel/barcelona`.
   - Permissions: Contents, read-only. Metadata, read-only, comes with it.
   - Expiration: a date after the submission, for example 2026-11-30.
   - Copy the token: GitHub shows it only once.
2. **A Kaggle notebook.** Create a new notebook, then in the right-hand panel:
   - Session options: Accelerator None, Internet on. Internet needs a phone-verified account.
   - Add-ons > Secrets > Add secret: label `GITHUB_TOKEN`, value the token. Tick it for this
     notebook.
3. Paste the whole of [cell.py](cell.py) into the notebook's only cell, and save the notebook.

## Each run

Work in progress is pushed to the `dev` branch, which the notebook runs (`REF = "dev"`). The
tests run in every run, before the job, so `main` is moved up to `dev` only after a run passes.

1. Set `REF` and `COMMAND` at the top of the cell.
2. Save Version > Save & Run All (Commit). The run carries on with the browser closed.
3. When it has finished, open that version > Output, and download `kaggle-results.zip`.
4. Put the zip at the repository root on the laptop and unzip it there. The job's outputs land at
   their paths in the repository, and the log and manifest land under `runs/<time>-<script>/`.
   The zip itself is gitignored.

## What a run does

[run.py](run.py) carries out the run in five steps:

1. Builds the pinned environment, Python 3.12.4 with `requirements.txt`, using uv. A job named
   in `EXTRA_REQUIREMENTS` gets its own extra file on top, held to `requirements.txt`'s pins:
   `scripts/train_tcn.py` and `scripts/train_tcn_ensemble.py` get PyTorch from
   `requirements-torch.txt`.
2. Downloads the IDSSE files at a pinned revision of the mirror and checks every file's size and
   digest ([scripts/fetch_idsse.py](../scripts/fetch_idsse.py)).
3. Runs the tests. A broken environment stops the run here.
4. Runs the job.
5. Zips what the job created or changed, with the full log and a manifest. The manifest records
   the commit, the command, the time of each step, the package versions, and the SHA-256 of each
   output.

The zip is written even when a step fails, so the log always comes back.

## Which jobs run where

| Job | Where | Reads |
|---|---|---|
| `scripts/train_models.py`: the models and the decoder | Kaggle | IDSSE |
| `scripts/eval_external.py` | Kaggle | IDSSE |
| `scripts/calibrate_simulator.py` | Kaggle | IDSSE |
| `scripts/feature_report.py`, `simulate_report.py`, `idsse_label_report.py` | Kaggle | IDSSE |
| `scripts/train_stacking.py` (M11l) | Kaggle | IDSSE |
| `scripts/train_tcn.py` (M11l): set Accelerator to GPU T4 x2 before saving the version | Kaggle, GPU | IDSSE |
| `scripts/train_tcn_ensemble.py` (M11m): GPU T4 x2 as well | Kaggle, GPU | IDSSE |
| `python -m pytest` | anywhere; step 3 of every Kaggle run | synthetic data |
| `scripts/make_submission.py`, `check_baselines.py`, `atd_profile.py`, `data_properties.py` | the laptop only | the match data |

## The token

The token can read this one repository and nothing else. It is kept in Kaggle's secret store and
passed once, as a one-off header, to a single `git clone`. It is never written to disk, never
printed, and never passed to the job, which runs without any Kaggle credentials in its
environment. Never paste the token into a file, a cell or a chat, and revoke it after the
submission.
