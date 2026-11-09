"""Profile the match's automatic tracking: aggregates only, to calibrate the simulator.

The practice ground (IDSSE) has complete, corrected tracking. possession.simulate degrades it to
look like this match's automatic tracking, and these aggregates are its target: how often and for
how long the ball disappears, in what situations, how many players are tracked, and how fast the
players seem to move, which shows how noisy the positions are (M11g). This is unsupervised use of
the match's own tracking: nothing here reads a ground truth or a label, and every number is a
statistic over both graded halves (AGENTS.md).

    python scripts/atd_profile.py

Takes about a minute: it parses all 199,319 lines of tracking.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from games import GAME, GAMES, half_video_blocks, require  # noqa: E402
from possession.atd import halves, read_atd  # noqa: E402
from possession.profile import format_profile, profile  # noqa: E402
from possession.safety import run_safely  # noqa: E402


def main():
    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    pitch = GAMES[GAME]["pitch"]
    atd = read_atd(meta_path, raw_path, GAMES[GAME]["n_video_frames"], pitch)
    result = profile(halves(atd, half_video_blocks(GAME)), pitch)
    print("ATD profile of {}, both graded halves (aggregates only)".format(GAME))
    for line in format_profile(result):
        print("  " + line)
    return 0


if __name__ == "__main__":
    sys.exit(run_safely(main))
