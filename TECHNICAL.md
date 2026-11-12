# Ball possession from automatic tracking: technical document

Team BasahKuyups, More than a Hack 2027, Phase 1. For every frame of the tactical video of FC
Barcelona vs Athletic Club, the system says whether the home side has the ball (`H`), the away
side has it (`A`) or the ball is dead (`D`), from the provided automatic tracking alone. This
document covers what was built and why, the assets and data used, the possession rules, the
limitations and how to reproduce the file. The code is in the repository
[github.com/YonoBengkel/barcelona](https://github.com/YonoBengkel/barcelona), where
[ARCHITECTURE.md](ARCHITECTURE.md) maps it.

## 1. Approach

```
tactical tracking ─► 46 columns per frame ─► P(dead), P(home | alive) ─► 3-state HMM ─► CSV
  (positions only)    (possession.features)   (temporal conv. network)     (Viterbi)
```

1. **Features from positions alone** (`possession/features.py`). The provided tracking has no ball
   on 41% of the graded frames, no ball height, no player identities and a speed channel that is
   missing on most ball frames. So every feature is computed from x and y, team by team:
   - how long the ball has been missing and will stay missing, and where it was last seen;
   - how fast the players move (they slow down when play stops);
   - the nearest player of each team, and how many of each team crowd the ball;
   - where the ball is and where it is heading along each team's attacking direction, which is
     inferred from where its deepest player stands;
   - how each team is shaped and moving: width, length and advance.

   Ball gaps up to 1 s are bridged by linear interpolation; longer ones are left unknown and
   flagged. Rolling means over 2.5 to 5 s add context, for 46 columns in all.
2. **Two probabilities per frame from a temporal convolutional network** (`possession/tcn.py`):
   the log-odds that the ball is dead, and, if it is alive, that the home side has it.
   - It reads the 46 columns of every frame of a half, standardised, with a missing value set to 0
     and flagged in a column of its own.
   - A 1x1 convolution to 64 channels is followed by eight residual blocks of convolutions over
     three frames, spaced 1, 2, 4, ... 128 frames apart. So each frame's output can draw on about
     10 s either side of it.
   - PyTorch fits it on whole training halves, each also with home and away swapped, so that it
     treats both sides alike: 3,000 steps of one half each, AdamW, settings fixed before the run.
   - The submission applies it with numpy alone. Each output is summed tap by tap and input by
     input in a fixed order, element by element, never through a matrix product. ReLU and
     element-wise arithmetic are exact, so the result does not depend on the CPU or on how a
     linear algebra library splits the work.
   - It replaced gradient-boosted trees for both calls (Section 5). The trees remain in the
     repository as the previous model and the reference every change is set against.
3. **A decoder** (`possession/decode.py`). `spellF1` matches whole possessions, and per-frame calls
   flicker, so the frame evidence is decoded with a three-state hidden Markov model.
   - A path pays `team_switch` = 16 nats of evidence to pass the ball from one team to the other,
     and `dead_switch` = 16 nats for a stoppage (half when play stops, half when it restarts).
   - `dead_bias` = 1.0 is added to the dead-ball evidence on every frame.
   - Viterbi gives each possession one start and one end. A short run of contrary frames cannot pay
     for the two switches it would take.

**Why this design.** Rules forbid any human label on this match, so a model has to learn on other
matches and transfer. The practice ground is seven public matches, made to look like this match's
tracking (Section 3). We judged that position features, two probabilities and an explicit decoder
would transfer more safely than an end-to-end model trained on footage of another league. The
network reads the same features the trees read and changes only how they are weighed over time;
the decoder stays explicit. All of it is cheap enough to be chosen by held-out scores on every
match.

**How every choice was made.**
- Each of the seven practice matches is held out in turn. The models are fitted on the other six,
  and the decoder's setting for the held-out match is chosen on the other six as well (nested), so
  the reported score is a held-out score for the whole pipeline.
- A change replaces the current pipeline only by a rule fixed in the code before each run: held-out
  `spellF1` at ±1 s at least 0.01 higher, better on at least 5 of the 7 matches, no loss at ±0.5 s
  or ±2 s, and `macroF1` down by at most 0.005.
- The last change, the network, was one of three candidates tried together. So it had to pass that
  rule twice: on the held-out matches as degraded for scoring, and again on a second, independent
  degradation of them.
- The public five minutes never chose anything.

## 2. Assets: used, and tried and dropped

| Asset | Use |
|---|---|
| Tactical tracking (`_atd.txt`) | The only input to the submission: ball and player positions in pitch metres |
| Tracking metadata (`_atd.xml`) | Each slot's team and the pitch size; its attacking directions serve only as a yes/no check of ours |
| Homography | Not used: the tracking is already in pitch coordinates |
| Smart Tagging | Not used: IDSSE has nothing like it, so its weight could not be chosen on the practice ground. Kept in mind as a cue for restarts |
| Video and audio (tactical, broadcast) | Not used. Detecting the referee's whistle from audio was considered and not built; IDSSE has no audio to choose it on either |
| Tracking speed channel | Not used: its unit is unclear and it is missing on 89.5% of the ball's frames |
| Public five-minute ground truth | A wiring check only: frames aligned, labels meaning what we think |
| Organisers' baseline | Rebuilt identically, then scored beside us on the practice ground as the floor |
| IDSSE (Section 3) | Training data and every model and parameter choice |

**Tried and dropped.** Where pipelines are compared, the numbers are held-out `spellF1` on the
practice ground at ±1 s labels.

- **Raw DFL possession labels.** They change on any touch: 11-17% of DFL possessions last under a
  second, against 1 in 17 in the public window. They were replaced by the controlled-possession
  labels of Section 4.
- **Logistic regressions for both calls** (0.609). A tree team model raised it to 0.630, then a
  tree dead-ball model with larger team trees to 0.651. The larger team trees alone did not pass
  the rule (+0.006).
- **One-second context features** beside the 2.5-5 s ones, aimed at short possessions: -0.002,
  better on 3 of 7 matches. Dropped.
- **Last-touch features**: which side last had a player within 1.5 m of the ball, how long ago,
  and how many touches in a row it made, a completed pass being the challenge's sign of control.
  They improved the team model frame by frame (log loss 0.2526 to 0.2466) but not the
  possessions: +0.0008, better on 3 of 7 matches. Kept in the code as design v3, not used.
- **A wider decoder grid** (`dead_bias` up to 3.0): the search chose the same setting.
- **A minimum length for every spell in the decoder**, aimed at the possessions merged with a
  neighbour: up to 1 s for a team's spell and 2 s for a stoppage, beside a lower cost of a change
  of team. The search kept that cost where it was and held a team's spell to at least 1 s:
  +0.0003, better on 2 of 7 matches. Kept in the code as the decoder grid `mindur`, not used.
- **Twice the training copies**: four degraded copies of every training half instead of two. The
  frame models improved a little (team log loss 0.2526 to 0.2465), but the decoder then chose a
  higher `team_switch` and merged more possessions: -0.0016, better on 3 of 7 matches.
- **A stage two of trees** that reads the trees' log-odds 0.2 to 2 s around each frame, fitted
  nested inside the leave-one-match-out loop so that no held-out label reached it. It improved both
  calls frame by frame (team log loss 0.2504 to 0.2288 on one degradation of the held-out
  matches), but the possessions by only +0.0015, with a loss at ±2 s. Not adopted.
- **The trees and the network averaged** passed the rule twice as well: 0.678, and 0.682 on the
  second degradation. The rule fixed before the run takes the candidate higher on the first, the
  network alone at 0.680.
- **Five networks averaged**, differing only in the seed of their fitting, against the one network:
  +0.0096, but better on only 4 of 7 matches, most of the gain coming from two of them. Allowing
  `dead_bias` up to 3.0 as well gave +0.0104, again on 4 of 7. On a fresh degradation of the
  held-out matches, used by no earlier run, both gained +0.0107, on 5 of 7. Frame by frame the
  average was clearly better (team log loss 0.2868 to 0.2654). But the rule asks for a gain on
  most matches, so the single network stayed.
- **0.15 m of player position noise in the simulator.** Once set against the match's player speeds
  it came out at none, which changed the practice ground itself (Section 3).

## 3. Training data and the practice ground

**The data.** IDSSE, the integrated dataset of Bassek et al. (2025):
- seven complete matches of the 2022/23 Bundesliga and 2. Bundesliga, about a million frames;
- positions at 25 fps from TRACAB optical tracking, collected by Sportec Solutions for the DFL,
  with the DFL's event data;
- owned by the DFL and released under CC BY 4.0;
- downloaded from the Hugging Face mirror `pysport/idsse-data` at revision
  `a715a38dfbaf5f58e431727c2b78d174101a703c`. `scripts/fetch_idsse.py` checks every file's size
  and digest.

**The labels.** They come with the data. Every position frame carries the DFL's ball status (in
play or not) and ball possession (which team), the same two facts the challenge's ground truth is
derived from, and the event feed records each pass, shot, claim and duel. We added no labels of our
own and corrected none by hand. A fixed rule turns the DFL's labels into the challenge's controlled
possession (Section 4), giving 2,201 possessions at ±1 s. **Nothing comes from the provided game.**

**The practice ground.** IDSSE is corrected tracking; the match's tracking is not. So
`possession/simulate.py` degrades every IDSSE half before anything is fitted or scored on it:
- **The ball is hidden on 40.7% of frames**, in gaps drawn from a mixture of short flickers and
  long losses. Gaps start where the ball is near a line, in a box, slow or crowded.
- **Players drop out** to 20.5 tracked on average.
- **The ball's position drifts** with 0.2 m of noise, and its height is dropped.

The parameters were fitted to label-free statistics of the match's own tracking, printed by
`scripts/atd_profile.py` as aggregates only:
- gap and seen-run lengths, where gaps start, crowding, and ball speeds (search loss 16.1 → 3.1);
- in a later refit, the players' speed percentiles. Degraded IDSSE matches them best with no
  added player noise (loss 0.302, against 5.069 at the earlier 0.15 m).

Every training half is degraded with two seeds.
- The network learns from those 28 halves whole, each also mirrored.
- The trees learned from every eighth frame of them: 250,672 dead-ball rows and 276,572 team rows.
- The held-out matches are degraded with a third seed. The network's adoption was checked again on
  a fourth.

## 4. Possession rules

The models learn the practice labels, so the rules live there. The labels follow
[CHALLENGE.md](docs/challenge/CHALLENGE.md), in `controlled_labels` of `possession/external.py`:

- **Dead ball.** Frames the DFL marks as out of play are `D`. A stoppage is never a possession.
- **Restarts.** A possession that opens a half or follows a dead ball belongs to the team that
  restarts play.
- **A change of possession needs control.** It counts only if the new team performs a controlled
  action within it, widened by 1 s at each end:
  - a pass, a shot, or a ball claimed, intercepted or held;
  - a ball action that is not a defensive clearance;
  - a duel its team wins with the ball (`ballClaimed`, `ballControlRetained`, `dribbledAround`, or
    one the DFL marks as changing possession).
- **Loose balls and challenges.** Without such an action, the run was a touch, a deflection, a
  clearance or a duel won without the ball, and it takes the label of the possession it
  interrupted: possession stays with the last team in control. A foul is a stoppage.
- **Timing.** The DFL's event clock runs ahead of or behind the tracking. Each match's median offset
  between possession changes and the new team's controlled actions is removed, and the ±1 s window
  absorbs the rest. Labels at ±0.5 s and ±2 s are scored as well, to show how much the line
  matters.

On the match, the decoder applies the same spirit. A team change has to be backed by about
16 nats of evidence, so a touch or a deflection does not move the ball. Every graded frame gets
`H`, `A` or `D`; `X` is written only outside the two graded halves, where nothing is scored.

## 5. Results on the practice ground

Held-out over the seven matches, nested as in Section 1, labels at ±1 s:

| Pipeline | `spellF1` | `macroF1` |
|---|---|---|
| Organisers' baseline (nearest player, ±50-frame majority) | 0.261 | 0.606 |
| v1: logistic dead-ball and team models | 0.609 | 0.889 |
| + tree team model | 0.630 | 0.891 |
| + tree dead-ball model and larger team trees | 0.651 | 0.910 |
| The same, on the ground recalibrated to the match's player speeds: the previous submission | 0.650 | 0.910 |
| The previous line's pipeline, fitted on the earlier ground, scored on the recalibrated one | 0.539 | 0.840 |
| The previous submission with a stage two of trees on its log-odds | 0.651 | 0.912 |
| The trees and the network averaged | 0.678 | 0.921 |
| Five networks averaged (not adopted: better on only 4 of 7 matches) | 0.690 | 0.924 |
| The network in place of the trees: **submitted** | **0.680** | **0.918** |

- The submitted pipeline scores 0.686 at ±0.5 s and 0.654 at ±2 s, and between 0.647 and 0.707 per
  match. It is better than the trees on 6 of the 7 matches; on the seventh it is 0.001 lower.
- On a second, independent degradation of the held-out matches it scores 0.674 against the trees'
  0.649, better on 5 of 7.
- Its decoder setting is `team_switch` 16, `dead_switch` 16, `dead_bias` 1.0.
- It matches 1,414 of the 2,201 practice possessions at IoU 0.5, against 1,353 for the trees, and
  20% of those under 2 s, against 16%.
- On the earlier ground, the baseline scored 0.261 against 0.243 published for it on the full
  match: for that baseline at least, the practice ground is about as hard as the match.
- The recalibration rows: on a ground whose players move as fast as the match's, the trees fitted
  with noisy player speeds lost 0.11. An earlier check pointed the same way: scaling the held-out
  players' speeds by 0.8 cost those trees 0.094 and raised their share of frames called dead from
  46% to 55%.

**On the match itself**, without labels, the file passes the organisers' validation, and every
check we can make falls where the practice ground predicts:
- 47.2% of the graded frames are called dead, against 44.8% dead in the practice labels.
- 319 possessions, 30.1 per 10 minutes with a mean of 10.5 s, inside the range the network gives on
  the practice matches (26.0-30.2 and 9.4-13.1 s), near its top.
- The attacking directions inferred from the tracking agree with the metadata in both halves.
- On the public five minutes, a check that chose nothing (the network was adopted on the practice
  ground before it ran): `macroF1` 0.836 and `spellF1` 0.682, against the baseline's 0.621 and
  0.326. With 17 possessions in the window, these numbers move a lot with one or two possessions.

## 6. Limitations

- **Short possessions and the team call.** Of the 2,201 practice possessions, 1,414 are matched at
  IoU 0.5. Of the missed ones:
  - 48% go to the wrong team;
  - 28% are merged with a neighbour;
  - 10% have their edges too far off;
  - 9% are called dead;
  - 5% are split.

  Only 20% of possessions under 2 s are matched, and 55% of those of 2-5 s.
- **The network learned from seven matches.**
  - Its training loss falls well below its held-out loss.
  - Its team log-odds are overconfident: held-out log loss 0.287 against the trees' 0.253, although
    it calls more live frames right (90.4% against 89.9%). The decoder, chosen on held-out scores,
    absorbs this.
  - The decoder's `dead_bias` sits at the top of its grid (1.0).
  - On the match its possessions per 10 minutes sit near the top of the practice range.
- **The practice ground is not the match.**
  - Bundesliga, not LaLiga.
  - The simulator still differs from the match's tracking in places. The ball before a gap moves
    at a median 4.8 m/s against 1.5 m/s, and IDSSE players are 5-9% faster in the middle of the
    speed distribution.
  - The simulator places ball gaps without regard to stoppages, while the match's tracker loses the
    ball mostly when play has stopped. We tried placing them at stoppages, and letting those gaps
    last until the restart. That made the long gaps 82-98% dead, with players there almost as slow
    as the match's. But it moved other statistics away from the match: the players ran faster
    while the ball was seen, and more gaps started near a line. No setting fitted the match's
    tracking better overall, so the simulator stayed as it was.
  - Anything the simulator does not imitate, the held-out scores cannot see.
- **The line on control is ours.** The DFL's labels, filtered by our rule, approximate the
  challenge's ground truth: practice possessions still run shorter (mean 7.7-13.2 s per match)
  than the public window's (14.2 s).
- **With more time we would:**
  - model where the match's tracker loses the ball, for instance in flight before it goes out, so
    that gaps can sit at stoppages without starting near a line too often;
  - add the referee's whistle from audio and the Smart Tagging as evidence, once there is a fair
    way to weigh them without labels on this match;
  - give the decoder a fuller model of how long possessions last: holding every spell to a
    minimum, the simplest such model, gained almost nothing (Section 2);
  - fit the network on more matches, and decide there on an ensemble of networks (ahead on
    average here, but not on most matches) and on its average with the trees (ahead on the second
    degradation).

## 7. Reproduction

Python 3.12, CPU only, the versions pinned in [requirements.txt](requirements.txt). Unpack the data
bundle into `data/` as [DATA.md](docs/challenge/DATA.md) describes, then:

```bash
pip install -r requirements.txt
python scripts/make_submission.py    # writes evaluation/results/possession_v1.csv, about a minute
```

- It reads two files of the bundle, whose SHA-256 should be:

  | File | SHA-256 |
  |---|---|
  | tactical `_atd.xml` | `327e35e32e46a647389b88f883d4f9a2ab931e3e6148aef2fcf8f22b1e93c43a` |
  | tactical `_atd.txt` | `f130e350d2bfdf734b7e32f072931fccc6d015c1e837d082064875fe0a700431` |

- It applies the committed `models/possession_v1+tcn.json`, validates the file with the
  organisers' own checks, and prints aggregates and the file's SHA-256. The network was adopted
  on 12 November 2026; it gives
  `17b2110c78b534d682e9c22f75e74b75f6e2b64e4f13e4d2b212499f7ed27432`. The file we submit,
  `BasahKuyups.csv`, is this output renamed.
- On the same day, a fresh clone of the repository at the adopting commit, whose code `main`
  carries unchanged, reproduced that digest on a Windows 11 laptop. It used Python 3.12.4 and a
  new virtual environment from `requirements.txt`, and took 39 seconds. The bundle was read in
  place through `MTAH_DATA_DIR`, the starter kit's own setting, instead of being copied into
  `data/`.
- The previous model, the trees frozen on 10 November (`--models
  models/possession_v1+trees-wide+deadtrees.json`), gives
  `7657abaebff87e85c2599a90031c5912f031e62bc3e8d79f9afb2936dcf2906e`.

**Refitting the model** uses public data only and is not needed to reproduce the file:

```bash
python scripts/fetch_idsse.py      # IDSSE at the pinned revision, size and digest checked
python scripts/train_tcn.py        # the network and its comparison, about 70 minutes on a T4 GPU
```

- The refit needs PyTorch (`requirements-torch.txt`), which nothing at submission time uses. A
  later refit on the same kind of GPU, a T4, gave exactly the same held-out scores. A refit on
  other hardware may differ in the last bits of the weights; the submission is reproduced from the
  committed weights.
- `python scripts/train_models.py` refits the trees, in about 25 minutes on 4 CPUs.
- We ran every refit on Kaggle through [kaggle/](kaggle/README.md). Each run's log and manifest
  (commit, versions, output digests) are in `runs/`.
- The tests (`python -m pytest`) use synthetic or public data only.

## 8. Rules we kept

- No human label on this match, for any purpose, and no hand correction of its tracking.
- No human-produced data about this match. The public five minutes checked the wiring and chose
  nothing.
- The only use of this match beyond producing the file is the label-free aggregates of Section 3.
- The match data never left the team's machine and is not in this repository; a pre-commit hook
  refuses it.
- Development was assisted by an AI coding assistant. It never received the match data or any
  extract of it, only aggregate statistics such as those quoted here, as the terms of data use
  require.

## Reference

Bassek, M., Rein, R., Weber, H. and Memmert, D. An integrated dataset of spatiotemporal and event
data in elite soccer. *Scientific Data* 12, 195 (2025). https://doi.org/10.1038/s41597-025-04505-y
