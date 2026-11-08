# Barcelona: ball possession for More than a Hack 2027

Team entry for **More than a Hack 2027, Phase 1: Ball Possession** (Metrica Sports, FC Barcelona
and Mobile World Capital Barcelona). For every frame of the tactical video of FC Barcelona vs
Athletic Club, say whether the home side has the ball (`H`), the away side has it (`A`), the ball
is dead (`D`), or the system declines to answer (`X`).

> **Status: work in progress.** The pipeline is being built. Scope and milestones are in
> [PRD.md](PRD.md), the design in [ARCHITECTURE.md](ARCHITECTURE.md).

## Layout

| Path | What |
|---|---|
| `possession/` | the team's pipeline |
| `scripts/` | entry points, one per task |
| `tests/` | tests; they run on synthetic or public data only |
| `games.py`, `evaluation/`, `starter/` | the organisers' starter kit, unchanged, from [metrica-sports/more-than-a-hack-2027](https://github.com/metrica-sports/more-than-a-hack-2027) at commit `f8f2629` (MIT, see [LICENSE](LICENSE)) |
| `docs/challenge/` | the organisers' documents: the challenge, the data, the scoring, the rules |
| `data/` | the match bundle. Gitignored and never committed |

## Setup

Python 3.12, CPU only.

```bash
python -m venv .venv
.venv\Scripts\activate                 # Windows; source .venv/bin/activate elsewhere
pip install -r requirements.txt
git config core.hooksPath .githooks    # refuses any commit that contains match data
```

Unpack the match bundle into `data/` exactly as [docs/challenge/DATA.md](docs/challenge/DATA.md)
describes.

## Reproducing the submission

The goal is one command that regenerates the submitted CSV byte for byte on CPU. It does not exist
yet. Until then, this rebuilds the organisers' four baselines from the match data and confirms
they match the committed ones, which checks the data bundle, the frame alignment and the
environment in one go:

```bash
python scripts/check_baselines.py
```

## Data and rules

- The match data is licensed for this competition only. It is never committed: `data/`, `*.mp4`
  and the bundle files are gitignored, and `.githooks/pre-commit` refuses them.
- No hand labels on this match, for any purpose. Every model and parameter choice is made on
  other matches from public datasets, never on the public five-minute ground truth.
- AI agents working in this repository follow [AGENTS.md](AGENTS.md), which keeps the match data
  away from hosted models, as the terms of the competition require.
