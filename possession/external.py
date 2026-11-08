"""Read the IDSSE open data into per-half arrays, and align its labels with the challenge.

IDSSE (Bassek, Rein, Weber and Memmert, Scientific Data 12, 195, 2025; CC BY 4.0) is the practice
ground: seven Bundesliga matches whose tracking states, on every frame, whether the ball is in
play (BallStatus) and which team owns it (BallPossession), the two facts the challenge's ground
truth is derived from. Download the files from the Hugging Face mirror pysport/idsse-data into
external/idsse/, keeping their names. (Figshare, where kloppy fetches them, answers scripted
downloads with a bot challenge.)

DFL's BallPossession changes on contacts the challenge would not count: 11-17% of DFL possessions
last under a second, against 1 in 17 in the public window, so raw DFL labels would teach a model
to switch far too often. controlled_labels() keeps a change only where the challenge's own
definition would: the new team restarts play, or performs a controlled action -- a pass, a shot,
a claimed ball, a duel won with the ball. The DFL event feed records those actions.
"""

import os
from datetime import datetime

import numpy as np
from lxml import etree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IDSSE_DIR = os.path.join(ROOT, "external", "idsse")
FPS = 25

# Match id -> competition id, which every file name carries.
MATCHES = {"J03WMX": "000001", "J03WN1": "000001", "J03WOH": "000002", "J03WOY": "000002",
           "J03WPY": "000002", "J03WQQ": "000002", "J03WR9": "000002"}
SECTIONS = ("firstHalf", "secondHalf")
HOME, AWAY = 0, 1

# Duel outcomes after which the winner has the ball under control. 'ballcontactSucceeded' and
# 'layoff' are contacts, 'fouled' is a stoppage: none of them is control on its own.
CONTROLLING_DUEL_RESULTS = ("ballClaimed", "ballControlRetained", "dribbledAround")
CONTROLLING_CLAIMS = ("InterceptedBall", "BallClaimed", "BallHeld")


def paths(match_id, root=IDSSE_DIR):
    stem = "DFL-COM-{}_DFL-MAT-{}.xml".format(MATCHES[match_id], match_id)
    return {"info": os.path.join(root, "DFL_02_01_matchinformation_" + stem),
            "events": os.path.join(root, "DFL_03_02_events_raw_" + stem),
            "positions": os.path.join(root, "DFL_04_03_positions_raw_observed_" + stem)}


def epoch(timestamp):
    """ISO 8601 with an offset -> seconds since 1970 UTC. DFL mixes +00:00 and +02:00."""
    return datetime.fromisoformat(timestamp).timestamp()


def read_info(path):
    """{'home': team id, 'away': team id} from the match information file."""
    general = etree.parse(path).getroot().find(".//General")
    return {"home": general.get("HomeTeamId"), "away": general.get("GuestTeamId")}


def read_positions(path, home, away):
    """{section: half} where half is a dict of arrays indexed by frame from the half's first.

    t0: epoch of frame 0. ball_x, ball_y, ball_z: metres. alive: bool. owner: HOME or AWAY, as
    DFL states it (also on dead frames). x, y: (frames, players) metres, NaN while a player is
    off the pitch. team: (players,) HOME or AWAY.

    Streams the file, which is 330-400 MB: each FrameSet is one person in one half.
    """
    framesets = {}
    team = rows = first_t = None
    for event, elem in etree.iterparse(path, events=("start", "end"),
                                       tag=("FrameSet", "Frame")):
        if elem.tag == "FrameSet":
            if event == "start":
                team, rows, first_t = elem.get("TeamId"), [], None
            else:
                framesets.setdefault(elem.get("GameSection"), []).append((team, rows, first_t))
                elem.clear()
                while elem.getprevious() is not None:
                    del elem.getparent()[0]
            continue
        if event != "end":
            continue
        if first_t is None:
            first_t = elem.get("T")
        if team == "BALL":
            rows.append((int(elem.get("N")), float(elem.get("X")), float(elem.get("Y")),
                         float(elem.get("Z", "nan")), elem.get("BallStatus") == "1",
                         elem.get("BallPossession") == "2"))
        else:
            rows.append((int(elem.get("N")), float(elem.get("X")), float(elem.get("Y"))))
        elem.clear()

    halves = {}
    for section, sets in framesets.items():
        ball = [(rows, first_t) for set_team, rows, first_t in sets if set_team == "BALL"]
        if len(ball) != 1:
            raise ValueError("{}: expected one ball frameset, found {}".format(section, len(ball)))
        rows, first_t = ball[0]
        frames = np.array([r[0] for r in rows])
        if not np.array_equal(frames, np.arange(frames[0], frames[0] + len(frames))):
            raise ValueError("{}: ball frames are not contiguous".format(section))
        players = [(set_team, set_rows) for set_team, set_rows, _ in sets
                   if set_team in (home, away) and set_rows]
        x = np.full((len(frames), len(players)), np.nan, dtype=np.float32)
        y = np.full_like(x, np.nan)
        for k, (_, set_rows) in enumerate(players):
            index = np.array([r[0] for r in set_rows]) - frames[0]
            keep = (index >= 0) & (index < len(frames))
            x[index[keep], k] = np.array([r[1] for r in set_rows])[keep]
            y[index[keep], k] = np.array([r[2] for r in set_rows])[keep]
        halves[section] = {
            "t0": epoch(first_t),
            "ball_x": np.array([r[1] for r in rows], dtype=np.float32),
            "ball_y": np.array([r[2] for r in rows], dtype=np.float32),
            "ball_z": np.array([r[3] for r in rows], dtype=np.float32),
            "alive": np.array([r[4] for r in rows]),
            "owner": np.array([r[5] for r in rows], dtype=np.int8),
            "x": x, "y": y,
            "team": np.array([HOME if t == home else AWAY for t, _ in players], dtype=np.int8),
        }
    return halves


def _actions(event, teams):
    """[(team, controlled)] for every ball action inside one DFL event element."""
    out = []
    for elem in event.iter():
        tag = elem.tag
        if tag in ("Play", "ShotAtGoal"):
            out.append((elem.get("Team"), True))
        elif tag == "BallClaiming":
            out.append((elem.get("Team"), elem.get("Type") in CONTROLLING_CLAIMS))
        elif tag == "OtherBallAction":
            out.append((elem.get("Team"), elem.get("DefensiveClearance") != "true"))
        elif tag == "TacklingGame":
            out.append((elem.get("WinnerTeam"),
                        elem.get("WinnerResult") in CONTROLLING_DUEL_RESULTS
                        or elem.get("PossessionChange") == "true"))
    return [(teams[team], controlled) for team, controlled in out if team in teams]


def read_events(path, home, away):
    """(times, teams, controlled) arrays of ball actions: epoch seconds, HOME/AWAY, bool.

    Delete markers stand alone in the published feed (no other event shares their id), so they
    are simply skipped.
    """
    teams = {home: HOME, away: AWAY}
    times, owners, controlled = [], [], []
    for event in etree.parse(path).getroot().iter("Event"):
        if not len(event) or event[0].tag == "Delete":
            continue
        t = epoch(event.get("EventTime"))
        for team, is_controlled in _actions(event[0], teams):
            times.append(t)
            owners.append(team)
            controlled.append(is_controlled)
    return np.array(times), np.array(owners, dtype=np.int8), np.array(controlled)


# Bump when read_positions or read_events change what they return, so stale caches are ignored.
CACHE_VERSION = 1


def load_match(match_id, root=IDSSE_DIR):
    """(info, halves, events) for one match: parsed once, then read from external/idsse/cache.

    Parsing a 330-400 MB position file takes about 20 s; the cached arrays load in about one.
    events is the (times, teams, controlled) triple of read_events.
    """
    files = paths(match_id, root)
    info = read_info(files["info"])
    cache = os.path.join(root, "cache", "{}.v{}.npz".format(match_id, CACHE_VERSION))
    if os.path.isfile(cache):
        with np.load(cache) as data:
            halves = {section: {key.split("__", 1)[1]: data[key] for key in data.files
                                if key.startswith(section + "__")} for section in SECTIONS}
            events = (data["events__times"], data["events__teams"], data["events__controlled"])
        for half in halves.values():
            half["t0"] = float(half["t0"])
        return info, halves, events

    halves = read_positions(files["positions"], info["home"], info["away"])
    events = read_events(files["events"], info["home"], info["away"])
    flat = {"{}__{}".format(section, key): np.asarray(value)
            for section in SECTIONS for key, value in halves[section].items()}
    flat.update({"events__times": events[0], "events__teams": events[1],
                 "events__controlled": events[2]})
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, **flat)
    return info, halves, events


def raw_labels(half):
    """'H', 'A' or 'D' per frame, straight from BallStatus and BallPossession."""
    return np.where(~half["alive"], "D", np.where(half["owner"] == HOME, "H", "A"))


def runs(labels):
    """[(start, stop, label)] maximal constant runs, stop exclusive."""
    breaks = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [len(labels)]))
    return [(int(bounds[i]), int(bounds[i + 1]), labels[bounds[i]])
            for i in range(len(bounds) - 1)]


def flip_offsets(labels, frames, teams, window=5 * FPS):
    """Frames from each raw possession change to the nearest controlled action of the new team.

    Raw DFL event times are wall-clock entries that run ahead of or behind the tracking; the
    median of these offsets is the correction, and their spread sets the tolerance.
    """
    out, previous = [], None
    for start, _, label in runs(labels):
        if label != "D" and previous not in (None, "D"):
            mine = frames[teams == (HOME if label == "H" else AWAY)]
            if len(mine):
                delta = mine[np.argmin(np.abs(mine - start))] - start
                if abs(delta) <= window:
                    out.append(int(delta))
        previous = label
    return np.array(out)


def aligned_labels(halves, events, tolerance_s):
    """({section: labels}, shift, offsets): controlled-possession labels for one match.

    shift is the median frame offset from DFL's possession changes to the new team's controlled
    actions, measured on this match and removed before matching; offsets are the measurements.
    """
    times, teams, controlled = events
    times, teams = times[controlled], teams[controlled]
    per_half = {}
    for section in SECTIONS:
        half = halves[section]
        frames = np.round((times - half["t0"]) * FPS).astype(int)
        inside = (frames >= 0) & (frames < len(half["alive"]))
        per_half[section] = (raw_labels(half), frames[inside], teams[inside])
    offsets = np.concatenate([flip_offsets(*per_half[s]) for s in SECTIONS])
    shift = int(np.median(offsets)) if len(offsets) else 0
    tolerance = int(round(tolerance_s * FPS))
    labels = {s: controlled_labels(raw, frames - shift, half_teams, tolerance)
              for s, (raw, frames, half_teams) in per_half.items()}
    return labels, shift, offsets


def controlled_labels(labels, frames, teams, tolerance):
    """Labels whose possession changes are only the ones the challenge would count.

    labels: raw 'H'/'A'/'D' for one half. frames, teams: controlled actions on that half's frame
    index, already offset-corrected. A possession run is kept when it follows a dead ball or opens
    the half (every restart has a clear owner), or when its team has a controlled action inside it
    widened by `tolerance` frames on each side. Otherwise it was a touch without control, and it
    takes the label of the possession it interrupted.
    """
    out = labels.copy()
    owner = None
    by_team = {"H": np.sort(frames[teams == HOME]), "A": np.sort(frames[teams == AWAY])}
    for start, stop, label in runs(labels):
        if label == "D":
            owner = None
            continue
        actions = by_team[label]
        low, high = np.searchsorted(actions, [start - tolerance, stop + tolerance])
        if owner is None or high > low:
            owner = label
        else:
            out[start:stop] = owner
    return out
