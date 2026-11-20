"""Lab: what the three definitions of possession look like on one IDSSE match.

Light and local: one public match (IDSSE, CC BY 4.0), parsed once and cached by
possession.external. It prints, per definition of possession.lab_labels, the whole match's number
of possessions and their mean length, and draws two figures:

- the same five minutes labelled three ways, as strips of H, A and D;
- the pitch at the first moment of that window where the raw and strict labels disagree, with the
  ball's last two seconds.

    python scripts/lab/plot_definitions.py [--match J03WMX] [--start-min 10] [--out-dir runs/lab-plots]
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from possession import external as ex  # noqa: E402
from possession import lab_labels  # noqa: E402

COLOURS = {"H": "#2a78d6", "A": "#eb6834", "D": "#c3c2b7"}
WINDOW_MIN = 5.0
TRAIL_S = 2.0


def possessions(labels):
    """(count, mean length in s) of the runs of H or A, each run between changes one possession."""
    lengths = [stop - start for start, stop, label in ex.runs(np.asarray(labels)) if label != "D"]
    return len(lengths), (float(np.mean(lengths)) / ex.FPS if lengths else 0.0)


def strips(ax, labelled, start, stop):
    """Draw one strip per definition over frames start..stop, minutes on the x axis."""
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    order = ["H", "A", "D"]
    image = np.array([[order.index(v) for v in labels[start:stop]]
                      for labels in labelled.values()])
    ax.imshow(image, aspect="auto", interpolation="nearest",
              cmap=ListedColormap([COLOURS[k] for k in order]), vmin=0, vmax=2,
              extent=(start / ex.FPS / 60, stop / ex.FPS / 60, len(labelled) - 0.5, -0.5))
    ax.set_yticks(range(len(labelled)))
    ax.set_yticklabels(list(labelled))
    ax.set_xlabel("minutes into the first half")
    names = {"H": "H: home has the ball", "A": "A: away has the ball", "D": "D: ball dead"}
    ax.legend(handles=[Patch(color=COLOURS[k], label=names[k]) for k in order],
              loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8)


def pitch(ax, half, frame, title):
    """Players, ball and the ball's recent trail at one frame, on a 105 x 68 m pitch."""
    from matplotlib.patches import Circle, Rectangle
    ax.add_patch(Rectangle((-52.5, -34), 105, 68, fill=False, color="#898781", lw=1))
    ax.plot([0, 0], [-34, 34], color="#898781", lw=1)
    ax.add_patch(Circle((0, 0), 9.15, fill=False, color="#898781", lw=1))
    for team, key in ((ex.HOME, "H"), (ex.AWAY, "A")):
        on = half["team"] == team
        ax.scatter(half["x"][frame, on], half["y"][frame, on], s=60, color=COLOURS[key],
                   edgecolors="white", linewidths=1, zorder=3, label="home" if key == "H"
                   else "away")
    trail = slice(max(0, frame - int(TRAIL_S * ex.FPS)), frame + 1)
    ax.plot(half["ball_x"][trail], half["ball_y"][trail], color="#0b0b0b", lw=1, zorder=4)
    ax.scatter([half["ball_x"][frame]], [half["ball_y"][frame]], s=30, color="#0b0b0b",
               zorder=5, label="ball")
    ax.set_xlim(-55, 55)
    ax.set_ylim(-36, 36)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)
    ax.legend(loc="lower left", fontsize=8, frameon=False, ncol=3)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--match", default="J03WMX")
    parser.add_argument("--start-min", type=float, default=10.0)
    parser.add_argument("--out-dir", default=os.path.join(ROOT, "runs", "lab-plots"))
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, halves, events = ex.load_match(args.match)
    per_definition = {name: lab_labels.aligned(name)(halves, events, 1.0)[0]
                      for name in lab_labels.DEFINITIONS}
    print("match {}: possessions over both halves, and their mean length".format(args.match))
    for name, labels in per_definition.items():
        counts = [possessions(labels[s]) for s in ex.SECTIONS]
        total = sum(c for c, _ in counts)
        mean = sum(c * m for c, m in counts) / max(total, 1)
        print("  {:<10} {:>4} possessions, mean {:.1f} s".format(name, total, mean))

    section = ex.SECTIONS[0]
    half = halves[section]
    labelled = {name: labels[section] for name, labels in per_definition.items()}
    start = int(args.start_min * 60 * ex.FPS)
    stop = min(len(half["alive"]), start + int(WINDOW_MIN * 60 * ex.FPS))
    os.makedirs(args.out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 2.4))
    strips(ax, labelled, start, stop)
    window = {name: possessions(labels[start:stop])[0] for name, labels in labelled.items()}
    ax.set_title("{} first half, minutes {:g}-{:g}: possessions in the window: {}".format(
        args.match, args.start_min, args.start_min + WINDOW_MIN,
        ", ".join("{} {}".format(k, v) for k, v in window.items())), fontsize=10)
    fig.tight_layout()
    strip_path = os.path.join(args.out_dir, "{}-definitions.png".format(args.match))
    fig.savefig(strip_path, dpi=130)
    plt.close(fig)
    print("wrote " + strip_path)

    raw, strict = labelled["raw"], labelled["strict"]
    seen = np.isfinite(half["ball_x"])
    differ = np.flatnonzero((raw[start:stop] != strict[start:stop]) & seen[start:stop])
    if len(differ):
        frame = start + int(differ[0])
        fig, ax = plt.subplots(figsize=(7, 5))
        pitch(ax, half, frame, "{} {:.1f} min: raw {}, controlled {}, strict {}".format(
            args.match, frame / ex.FPS / 60, raw[frame], labelled["controlled"][frame],
            strict[frame]))
        fig.tight_layout()
        pitch_path = os.path.join(args.out_dir, "{}-disagreement.png".format(args.match))
        fig.savefig(pitch_path, dpi=130)
        plt.close(fig)
        print("wrote " + pitch_path)
    else:
        print("raw and strict agree on every frame of the window")
    return 0


if __name__ == "__main__":
    sys.exit(main())
