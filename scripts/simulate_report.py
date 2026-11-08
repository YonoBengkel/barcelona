"""Profile IDSSE halves before and after degradation, for comparison with the match's profile.

Public data only. Put the output beside scripts/atd_profile.py's to see how far the simulator is
from the match's automatic tracking, and calibrate possession.simulate.DegradeParams from it.

    python scripts/simulate_report.py [MATCH_ID ...]
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from possession import external as ex  # noqa: E402
from possession.profile import format_profile, profile  # noqa: E402
from possession.simulate import DegradeParams, degrade  # noqa: E402

IDSSE_PITCH = (105.0, 68.0)


def main(match_ids):
    params = DegradeParams()
    print("DegradeParams:", params)
    clean, degraded = [], []
    for i, match_id in enumerate(match_ids):
        files = ex.paths(match_id)
        info = ex.read_info(files["info"])
        halves = ex.read_positions(files["positions"], info["home"], info["away"])
        for j, section in enumerate(ex.SECTIONS):
            clean.append(halves[section])
            degraded.append(degrade(halves[section], params, seed=10 * i + j))
        print("read {}".format(match_id))
    for name, data in (("clean IDSSE", clean), ("degraded IDSSE", degraded)):
        print("\n{} ({} halves)".format(name, len(data)))
        for line in format_profile(profile(data, IDSSE_PITCH)):
            print("  " + line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or sorted(ex.MATCHES)))
