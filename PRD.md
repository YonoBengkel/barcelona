# PRD: frame-level ball possession

## Problem

Label each of the 199,319 frames of the tactical video of FC Barcelona vs Athletic Club (LaLiga
2024-25, matchday 2) as `H`, `A`, `D` or `X`. The file is ranked on `spellF1`, which matches
predicted possessions one-to-one against the true ones by temporal overlap, with `macroF1` as the
first tie-break. The input is automatic, uncorrected tracking: no player identities, fragmented
tracks, and no ball on 41% of the graded frames. The full-match ground truth is private; the
public one covers five minutes.

## Users

- **The organisers' scorer** (`evaluation/run_evaluation.py`), which ranks the CSV.
- **The organisers**, who may re-run this repository to confirm it regenerates the CSV byte for
  byte, and who read the technical document.
- **The team**, who run the pipeline on the match data on their own machine.

## Constraints

- **Competition rules** ([docs/challenge/RULES.md](docs/challenge/RULES.md) and the organisers'
  answers in the repository's issues): labels come from an algorithm; no hand labels on this match
  for any purpose, including validating or choosing parameters; the public five minutes check the
  pipeline and never tune it; no human-produced data about this match; the submission must be
  reproducible from this repository.
- **Terms of data use**: no match data or extract of it may reach a hosted LLM. See
  [AGENTS.md](AGENTS.md).
- **Compute**: a CPU-only machine. Inference must be deterministic.
- **Time**: submission deadline 17 November 2026.

## MVP scope (v1)

1. A reader that turns the tactical tracking into per-frame arrays: ball position or none, the
   tracked objects, their team and speed.
2. Per-frame evidence for a dead ball and for each team having the ball, from the geometry and
   kinematics of the players and the ball.
3. A temporal decoder over `H`, `A`, `D` (a hidden Markov or semi-Markov model, decoded with
   Viterbi) that turns the evidence into whole possessions instead of flickering frames.
4. A practice ground on other matches: the public IDSSE tracking, whose per-frame ball status and
   possession are aligned to the challenge's controlled-possession definition and degraded to
   look like automatic tracking. Every model and parameter choice is made there.
5. One command that regenerates the submission, and a short technical document.

## Non-goals

- Re-detecting players or the ball from video, re-tracking, or camera calibration.
- The broadcast recording: it is not frame-aligned to the tactical one.
- Deep learning on the match video.
- Live (causal) inference. The task is post-match, so future frames may be used.

## Success measures

| Measure | Target |
|---|---|
| Valid file | `evaluation/validate_submission.py` passes on the exact file submitted |
| Better than the floor | v1 beats the baseline algorithm (nearest player, ±50-frame majority) on the degraded practice matches by more than the `spellF1` noise band of about 0.04, scored leave-one-match-out |
| Reproducible | a clean clone and one command reproduce the CSV byte for byte on CPU |
| Plausible possessions | possession count and mean duration on the match fall inside the range seen on the practice matches |
| On time | submitted by 16 November 2026, a day before the deadline |

## Milestones

| Phase | Milestone | Target |
|---|---|---|
| A. Foundations | M1 team registered · M2 repository and environment · M3 official baselines rebuilt and identical | 8-10 Nov |
| B. Practice ground | M4 public tracking verified (go/no-go) · M5 degradation to automatic-tracking quality · M6 evaluation harness | 9-11 Nov |
| C. Pipeline v1 | M7 per-frame features · M8 dead-ball and possession evidence · M9 decoder and first valid submission | 10-12 Nov |
| D. Improvement | M10 error analysis · M11 optional signals (Smart Tagging, audio) · M12 model frozen | 13-15 Nov |
| E. Release | M13 clean reproduction · M14 technical document · M15 submission | 15-17 Nov |
