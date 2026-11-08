"""Read the challenge's automatic tracking (ATD) into the arrays the pipeline shares.

The raw file is Metrica's FIFA/EPTS text export, one line per video frame:
"<frame_id>:<slot>;<slot>;...:<ball>", every slot group "x,y,speed", NaN where nothing was tracked,
coordinates normalised 0-1 from a corner. starter/baseline.py parses the same lines but keeps only
the slot nearest the ball; this keeps all of them.

The arrays follow possession.external, so one pipeline serves both: metres from the centre spot,
NaN where missing, frames indexed from 0. For the tactical recording frame i is submission row i,
because the ATD counts frame_id from 1.
"""

import xml.etree.ElementTree as ET

import numpy as np

HOME, AWAY = 0, 1


def team_of_slots(meta_path):
    """(slots,) HOME or AWAY, in slot order.

    The first team in the metadata is the home side: the FIFA/EPTS convention that
    starter/baseline.py relies on too, and for this match it is Barcelona.
    """
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("expected 2 teams in the ATD metadata, found {}".format(len(team_ids)))
    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError("no Player entries in the ATD metadata")
    return np.array([HOME if team_id == team_ids[0] else AWAY for team_id in slots], dtype=np.int8)


def read_atd(meta_path, raw_path, n_frames, pitch):
    """{'ball_x', 'ball_y': (frames,), 'x', 'y': (frames, slots), 'team': (slots,)} in metres.

    Lines are placed by their own frame id, so a missing line leaves its frame NaN rather than
    shifting every later one. Errors name a count, never a value: a message that quoted the line
    would be an extract of the match data (AGENTS.md).
    """
    team = team_of_slots(meta_path)
    n_slots = len(team)
    length, width = pitch
    x = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    y = np.full_like(x, np.nan)
    ball_x = np.full(n_frames, np.nan, dtype=np.float32)
    ball_y = np.full_like(ball_x, np.nan)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue  # a blank trailing line; starter/baseline.py skips these too
            frame = int(parts[0]) - 1
            if not 0 <= frame < n_frames:
                raise ValueError("a frame id lies outside the {} video frames".format(n_frames))
            values = np.array(parts[1].replace(";", ",").split(","), dtype=np.float32)
            if values.size != 3 * n_slots:
                raise ValueError("a line has {} slot values, expected {}".format(
                    values.size, 3 * n_slots))
            values = values.reshape(n_slots, 3)
            x[frame] = values[:, 0]
            y[frame] = values[:, 1]
            ball = parts[2].split(",")
            ball_x[frame] = float(ball[0])
            ball_y[frame] = float(ball[1])

    return {"ball_x": (ball_x - 0.5) * length, "ball_y": (ball_y - 0.5) * width,
            "x": (x - 0.5) * length, "y": (y - 0.5) * width, "team": team}


def halves(atd, blocks):
    """[half] sliced to the graded blocks, e.g. games.half_video_blocks('BAR-ATH').

    Each half keeps 'first_frame', the video frame of its row 0, so labels can be written back
    into the full-length submission.
    """
    out = []
    for _, first, last in blocks:
        part = {key: atd[key][first:last + 1] for key in ("ball_x", "ball_y", "x", "y")}
        part["team"] = atd["team"]
        part["first_frame"] = first
        out.append(part)
    return out
