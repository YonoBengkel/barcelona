"""Read SkillCorner's open broadcast tracking into the pipeline's per-half arrays, with its labels.

SkillCorner open data (github.com/SkillCorner/opendata, MIT licence; SkillCorner asks to be
credited): 20 matches of the 2024/25 Australian A-League, tracked from the broadcast video at
10 fps. Every frame names the team in possession ('possession': {'player_id', 'group'}, the group
'home team', 'away team' or none). The group holds through passes while the player id changes, so
it is team possession; it is empty while play is stopped and while the broadcast shows no play (a
replay, a close-up), when the frame carries no tracking at all. Another provider, league and way of
tracking than the practice ground's (IDSSE, DFL's optical tracking at 25 fps): the lab uses it to
see how the submitted pipeline travels.

Labels per frame: 'H' for the home team's group, 'A' for the away team's, 'D' for none.

The arrays follow possession.external and possession.atd, so possession.features computes the same
columns: metres from the centre spot with x along the long side, NaN where missing, at 25 fps. The
10 fps samples are interpolated linearly in time (a frame next to a missing sample stays missing)
and the labels come from the nearest sample. Each player keeps one column for the whole match, by
player id, NaN while off the pitch. SkillCorner extrapolates a player or the ball it cannot see
('is_detected' false); detected_only=True drops those positions, closer to a tracker that reports
only what it sees.

The files are pinned to one commit of the repository and checked before use: the tracking files
are Git LFS objects, listed by SHA-256; the match files by their Git blob id. Standard library and
numpy only.
"""

import hashlib
import json
import os
import shutil
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKILLCORNER_DIR = os.path.join(ROOT, "external", "skillcorner")
COMMIT = "4340d274572876239c154c90bc507a9b3250a656"  # master on 2026-11-21
RAW_URL = "https://raw.githubusercontent.com/SkillCorner/opendata/{}/data/matches/{}/{}"
LFS_URL = "https://media.githubusercontent.com/media/SkillCorner/opendata/{}/data/matches/{}/{}"
SOURCE_FPS = 10
FPS = 25
HOME, AWAY = 0, 1
GROUPS = {"home team": "H", "away team": "A"}
# Bump when read_match changes what it returns, so stale caches are ignored.
CACHE_VERSION = 1

# match id -> ((match file bytes, Git blob id), (tracking file bytes, SHA-256)), at COMMIT.
FILES = {
    "1874553": ((33730, "51ef142ceba0f85700b376faf1fdbc4e4a728629"),
                (90729279, "ea97f58f8eaad925feaeacc6395ec24860dd80027af8a89450276adebd29d265")),
    "1886347": ((27541, "10779035f95d26a30adf8330c351e4cc4d16a513"),
                (89280839, "3577e2803da95390f8b2f85d47829eb55fbbfee6211c94adc65db0dc78e46b41")),
    "1899585": ((27810, "42ca159d85dc59476bf5278961ccbda6633a3303"),
                (85550410, "90b6289ec2181192d1793ec6240fbbe8225ff0219c7956710c6278c31a661100")),
    "1925299": ((28396, "62b69aba5be11f795ccdbcf65c6d07f662832b66"),
                (97091519, "48528be40ad82fa3e2b232e1af8d311705f7d6d119925ef199d2b59ea36fd3c9")),
    "1927964": ((33400, "e42fb7edbd49c81e087d0d73cc885cc3095d6a20"),
                (91049828, "858c7a756ca3d33017f864775af5999d2c0f364e58ab60ae040c8b9ebab17827")),
    "1953632": ((27767, "b63200aace34ada146ab615e776d9a022f1a72ad"),
                (90027965, "1c2014f97c3eff1ce2475203512c9d126930d7599c036105c9ac442d1de1667e")),
    "1959846": ((32946, "3513a7cd787f59e3f8027ba17365c17d78878969"),
                (87710167, "5ba2d87f6fd1ba3138301b369bd9266be69416a2d799a3fba5b5ec8a52e1284b")),
    "1986691": ((33792, "dee942a7a5464f0730aff27b93b5269f5697d88b"),
                (88531597, "630e946950fe459f5e48872b01a36e6dfc319cf9027e7b98156150bb6e2c30f6")),
    "1996435": ((27830, "4b6aea20c71dd1128228aff2d8a12d848a0109af"),
                (88634016, "f42764d8bde290e8f738fa21141cadbed75408314b0b337bbf1d5a3444540a02")),
    "1996436": ((32859, "61736f019eea4d8c97e891dfee3df7e169fc27ff"),
                (88945845, "fab9d331e06f3042c83ac17798901b49bb259c99a8f3b319cd0146263cde65e7")),
    "2006229": ((28213, "71bff731f85f5266dafd4fa073b52ef66eb91ff8"),
                (90078007, "3ee193ba01f1fe2cbce087eca8a881c21b8b0ea3485ae6c3d7e32e8b2f51e738")),
    "2006363": ((33294, "edaf492c5fc4c98cb871a8ade25065e9a38d6ef7"),
                (90591887, "61dfdbd83f8e654127b249b8503f062338c14ec0c9e65404bcf70fb7c1346bd4")),
    "2007448": ((32670, "7798b2874d7593e33c9700ba19b12b380a64c65b"),
                (90025383, "12d00d6e4257218812df6e4c4263c49387ba0313a83ff8d826c4416828f9c798")),
    "2007721": ((33745, "0bbbabf46dd7330bf8d16faf979fbf576f37218d"),
                (92872706, "fea913b8d4cbde8d85c6bdf5f1a4b3fd203b78227e976a171c8ab0946705d378")),
    "2010085": ((33744, "82b15d3e8cac13d0403259785302023ad89ab9ca"),
                (92516156, "04562d9754b2977c80430535e9b5733426bc4ce910db56fe7a3760876211ac5a")),
    "2011166": ((28420, "6468e14c07f0108c56cd3457544c106643161099"),
                (87441570, "8ec2937050930d87e4f52b2d65515f70ef8edd0ae2814bf40155c4448133390e")),
    "2013725": ((27899, "4309772e339cd1f9ebc2aeb4a429076287af3a00"),
                (96059074, "13eb83329814ed5f5d73da8f8ab5c5979d6c8a7858e89e28a7fa8072f91fb250")),
    "2015213": ((27937, "1d58d487d5ca7f8c9cfe28de8a9757e3fae2aff4"),
                (102030769, "413990d0024d84856ce1f2a16866b8887472c0cea22c61caef0d1f46505a609e")),
    "2016236": ((33791, "8bb23c902bb3d5ea82092f6b9c8c20e2379fa67b"),
                (94737872, "d78555f9dfd7d1ce9de38066d3c98e6d54417c6d45d73b1f1ce7e26f07263d8f")),
    "2017461": ((28545, "281fd462ac1ca85a6d9ed2d5b5d5a7654627abe4"),
                (89543442, "59cfa5588efee041da439f44df5b5b92f2b7305571dc831164e3d05f7415787a")),
}
MATCHES = tuple(sorted(FILES))


def names(match_id):
    """(match file, tracking file) names of one match."""
    return match_id + "_match.json", match_id + "_tracking_extrapolated.jsonl"


def paths(match_id, root=SKILLCORNER_DIR):
    return tuple(os.path.join(root, match_id, name) for name in names(match_id))


def digest(path, size, expected):
    """The file's digest in the form `expected` is listed in: SHA-256, or a Git blob id."""
    h = hashlib.sha256() if len(expected) == 64 else hashlib.sha1(b"blob %d\0" % size)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def intact(path, size, expected):
    return (os.path.isfile(path) and os.path.getsize(path) == size
            and digest(path, size, expected) == expected)


def download(url, path):
    part = path + ".part"
    request = urllib.request.Request(url, headers={"User-Agent": "possession-lab"})
    with urllib.request.urlopen(request, timeout=120) as response, open(part, "wb") as out:
        shutil.copyfileobj(response, out, 1 << 20)
    os.replace(part, path)


def fetch(matches=MATCHES, root=SKILLCORNER_DIR, log=print):
    """Download the pinned files of matches that are missing or damaged; return how many were
    fetched. Raises OSError when a file cannot be fetched intact in three tries."""
    fetched = 0
    for match in matches:
        (match_size, match_digest), (tracking_size, tracking_digest) = FILES[match]
        match_name, tracking_name = names(match)
        for name, size, expected, url in (
                (match_name, match_size, match_digest, RAW_URL.format(COMMIT, match, match_name)),
                (tracking_name, tracking_size, tracking_digest,
                 LFS_URL.format(COMMIT, match, tracking_name))):
            path = os.path.join(root, match, name)
            if intact(path, size, expected):
                continue
            os.makedirs(os.path.dirname(path), exist_ok=True)
            for _ in range(3):
                try:
                    download(url, path)
                except OSError as error:
                    log("retrying {}: {}".format(name, type(error).__name__))
                    continue
                if intact(path, size, expected):
                    break
                log("retrying {}: size or digest does not match".format(name))
            else:
                raise OSError("could not fetch {} intact".format(name))
            fetched += 1
    return fetched


def read_info(path):
    """{'team': {player id: HOME or AWAY}, 'pitch': (length, width), 'periods': [(period,
    first frame, last frame)]} from a match file; frames count at 10 fps from the video's start."""
    with open(path, encoding="utf-8") as f:
        match = json.load(f)
    sides = {match["home_team"]["id"]: HOME, match["away_team"]["id"]: AWAY}
    team = {player["id"]: sides[player["team_id"]] for player in match["players"]
            if player["team_id"] in sides}
    periods = [(int(p["period"]), int(p["start_frame"]), int(p["end_frame"]))
               for p in match["match_periods"]]
    return {"team": team, "pitch": (float(match["pitch_length"]), float(match["pitch_width"])),
            "periods": periods}


def read_tracking(path, info):
    """(players, {period: arrays at 10 fps}) from a tracking file.

    players: the player ids, one column each, in increasing order. Per period, indexed from its
    first frame: 'x', 'y' (frames, players) and 'ball_x', 'ball_y' (frames,) in metres, NaN where
    the frame has none; 'seen' and 'ball_seen', whether SkillCorner detected it rather than
    extrapolated it; 'tracked', whether the frame carries any tracking; 'labels', 'H', 'A' or 'D'.
    Frames outside the periods are skipped; a player id missing from the match file raises.
    """
    players = sorted(info["team"])
    column = {player: j for j, player in enumerate(players)}
    spans = {period: (first, last) for period, first, last in info["periods"]}
    out = {}
    for period, (first, last) in spans.items():
        n = last - first + 1
        out[period] = {"x": np.full((n, len(players)), np.nan, dtype=np.float32),
                       "y": np.full((n, len(players)), np.nan, dtype=np.float32),
                       "seen": np.zeros((n, len(players)), dtype=bool),
                       "ball_x": np.full(n, np.nan, dtype=np.float32),
                       "ball_y": np.full(n, np.nan, dtype=np.float32),
                       "ball_seen": np.zeros(n, dtype=bool),
                       "tracked": np.zeros(n, dtype=bool),
                       "labels": np.full(n, "D", dtype="<U1")}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            frame = json.loads(line)
            period = frame.get("period")
            if period not in spans:
                continue
            i = int(frame["frame"]) - spans[period][0]
            arrays = out[period]
            if not 0 <= i < len(arrays["labels"]):
                continue
            arrays["labels"][i] = GROUPS.get((frame.get("possession") or {}).get("group"), "D")
            ball = frame.get("ball_data") or {}
            if ball.get("x") is not None and ball.get("y") is not None:
                arrays["ball_x"][i], arrays["ball_y"][i] = ball["x"], ball["y"]
                arrays["ball_seen"][i] = bool(ball.get("is_detected"))
            entries = frame.get("player_data") or []
            arrays["tracked"][i] = bool(entries)
            for entry in entries:
                if entry.get("x") is None or entry.get("y") is None:
                    continue
                j = column[entry["player_id"]]
                arrays["x"][i, j], arrays["y"][i, j] = entry["x"], entry["y"]
                arrays["seen"][i, j] = bool(entry.get("is_detected"))
    return players, out


def frames_at_fps(n_source):
    """How many 25 fps frames span n_source 10 fps samples, first sample to last."""
    return (n_source - 1) * FPS // SOURCE_FPS + 1 if n_source else 0


def resampled(values, n_out):
    """values (samples, ...) at 10 fps, linearly interpolated to n_out frames at 25 fps.

    Frame k sits at sample 2k/5. On a sample it takes the sample's value; between two it mixes
    them, so a frame next to a missing sample stays missing. Integer arithmetic places the frames,
    so the result is the same on any machine.
    """
    values = np.asarray(values)
    k = np.arange(n_out)
    low = k * SOURCE_FPS // FPS
    weight = (k * SOURCE_FPS % FPS) / FPS
    high = np.minimum(low + 1, len(values) - 1)
    weight = weight.reshape((-1,) + (1,) * (values.ndim - 1))
    mixed = values[low] * (1.0 - weight) + values[high] * weight
    return np.where(weight == 0, values[low], mixed).astype(values.dtype)


def nearest_sample(values, n_out):
    """values (samples, ...) at 10 fps taken at n_out frames at 25 fps from the nearest sample,
    a tie (never at these rates) to the later one."""
    k = np.arange(n_out)
    index = np.minimum((2 * k * SOURCE_FPS + FPS) // (2 * FPS), len(values) - 1)
    return np.asarray(values)[index]


def at_fps(arrays, team, detected_only=False):
    """One period's 10 fps arrays as a half for possession.features at 25 fps, plus its labels.

    Returns {'half': {'x', 'y', 'ball_x', 'ball_y', 'team'}, 'labels', 'tracked'}.
    detected_only: positions SkillCorner extrapolated are dropped (NaN) before resampling.
    """
    x, y = arrays["x"], arrays["y"]
    ball_x, ball_y = arrays["ball_x"], arrays["ball_y"]
    if detected_only:
        x, y = np.where(arrays["seen"], x, np.nan), np.where(arrays["seen"], y, np.nan)
        ball_x = np.where(arrays["ball_seen"], ball_x, np.nan)
        ball_y = np.where(arrays["ball_seen"], ball_y, np.nan)
    n = frames_at_fps(len(arrays["labels"]))
    half = {"x": resampled(x, n), "y": resampled(y, n), "ball_x": resampled(ball_x, n),
            "ball_y": resampled(ball_y, n), "team": np.asarray(team, dtype=np.int8)}
    return {"half": half, "labels": nearest_sample(arrays["labels"], n),
            "tracked": nearest_sample(arrays["tracked"], n)}


def load_match(match_id, root=SKILLCORNER_DIR, detected_only=False):
    """(info, [one at_fps() dict per period, in period order]) for one match.

    The 10 fps arrays are parsed once and cached under root/cache; resampling is redone on every
    load, so both readings come from one cache.
    """
    match_path, tracking_path = paths(match_id, root)
    info = read_info(match_path)
    cache = os.path.join(root, "cache", "{}.v{}.npz".format(match_id, CACHE_VERSION))
    keys = ("x", "y", "seen", "ball_x", "ball_y", "ball_seen", "tracked", "labels")
    if os.path.isfile(cache):
        with np.load(cache) as data:
            players = data["players"].tolist()
            periods = {int(p): {key: data["{}__{}".format(p, key)] for key in keys}
                       for p in data["periods"]}
    else:
        players, periods = read_tracking(tracking_path, info)
        flat = {"{}__{}".format(p, key): arrays[key] for p, arrays in periods.items()
                for key in keys}
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez(cache, players=np.array(players), periods=np.array(sorted(periods)), **flat)
    team = [info["team"][player] for player in players]
    return info, [at_fps(periods[p], team, detected_only) for p in sorted(periods)]
