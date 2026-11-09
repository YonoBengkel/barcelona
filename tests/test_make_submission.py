"""scripts/make_submission.py end to end on a small made-up match, never the real one.

The real run reads the match on the team's laptop; this catches broken plumbing before that.
"""

import importlib.util
import os
import sys

import numpy as np

import games
from possession.models import DESIGN_NAMES, fit, save

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "make_submission", os.path.join(ROOT, "scripts", "make_submission.py"))
make_submission = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(make_submission)

N_FRAMES = 600
BLOCKS = ((50, 249), (300, 549))  # video frames of the two graded halves
META = """<?xml version="1.0"?>
<Metadata><Teams>
<Team id="T1"><Name>Home</Name><ProviderTeamsParameters><ProviderParameter>
<Name>attack_direction_first_half</Name><Value>left_to_right</Value></ProviderParameter>
</ProviderTeamsParameters></Team>
<Team id="T2"><Name>Away</Name><ProviderTeamsParameters><ProviderParameter>
<Name>attack_direction_first_half</Name><Value>right_to_left</Value></ProviderParameter>
</ProviderTeamsParameters></Team></Teams><Players>
<Player id="P1" teamId="T1"/><Player id="P2" teamId="T1"/><Player id="P3" teamId="T1"/>
<Player id="P4" teamId="T2"/><Player id="P5" teamId="T2"/><Player id="P6" teamId="T2"/>
</Players></Metadata>"""


def raw_tracking(rng):
    """Lines of a made-up ATD export. Each keeper (slots 0 and 3) stands by its own goal, home
    on the left in the first half, so the metadata above is right and the teams swap ends."""
    lines = []
    for frame in range(N_FRAMES):
        second_half = frame >= BLOCKS[1][0]
        keepers = (0.95, 0.05) if second_half else (0.05, 0.95)
        slots = []
        for slot in range(6):
            if slot in (0, 3):
                x = keepers[slot // 3]
            else:
                x = 0.5 + 0.15 * np.sin(frame / 40.0 + slot)
            y = 0.5 + 0.3 * np.cos(frame / 50.0 + slot)
            slots.append("{:.4f},{:.4f},NaN".format(x, y))
        ball = "NaN,NaN,NaN" if rng.random() < 0.3 else "{:.4f},{:.4f},NaN".format(
            0.5 + 0.2 * np.sin(frame / 30.0), 0.5)
        lines.append("{}:{}:{}\n".format(frame + 1, ";".join(slots), ball))
    return "".join(lines)


def test_a_valid_file_and_aggregates_only(tmp_path, monkeypatch, capsys):
    rng = np.random.default_rng(0)
    data = tmp_path / "data" / "tactical"
    data.mkdir(parents=True)
    (data / "BAR-ATH_tactical_atd.xml").write_text(META)
    (data / "BAR-ATH_tactical_atd.txt").write_text(raw_tracking(rng))
    offset = 1000
    monkeypatch.setattr(games, "DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setitem(games.GAMES, "BAR-ATH", dict(
        games.GAMES["BAR-ATH"], n_video_frames=N_FRAMES, offsets={1: offset, 2: offset},
        halves={1: (BLOCKS[0][0] + offset, BLOCKS[0][1] + offset),
                2: (BLOCKS[1][0] + offset, BLOCKS[1][1] + offset)}))

    X = rng.normal(size=(500, len(DESIGN_NAMES)))
    models = {"dead": fit(X, X[:, 0] > 0), "team": fit(X, X[:, 1] > 0),
              "decoder": {"team_switch": 8, "dead_switch": 8, "dead_bias": 0.0},
              "fitted_on": {"practice_range": {"labels": {"spells_per_10min": [0.1, 500.0],
                                                          "mean_spell_s": [0.01, 100.0]}}}}
    models_path = str(tmp_path / "models.json")
    save(models, models_path)
    gt = ["X"] * N_FRAMES
    gt[100:200] = ["H"] * 100
    gt_path = tmp_path / "gt.csv"
    gt_path.write_text("frame,label\n" + "".join("{},{}\n".format(i, l) for i, l in enumerate(gt)))
    monkeypatch.setattr(make_submission, "PUBLIC_GT", str(gt_path))
    out = str(tmp_path / "submission.csv")
    monkeypatch.setattr(sys, "argv", ["make_submission.py", "--out", out, "--models", models_path])

    assert make_submission.run_safely(make_submission.main) == 0

    with open(out, "rb") as handle:
        content = handle.read()
    assert b"\r" not in content
    labels = [line.split(",")[1] for line in content.decode().splitlines()[1:]]
    assert len(labels) == N_FRAMES
    graded = np.zeros(N_FRAMES, dtype=bool)
    for first, last in BLOCKS:
        graded[first:last + 1] = True
    assert all(label in "HAD" for label, g in zip(labels, graded) if g)
    assert all(label == "X" for label, g in zip(labels, graded) if not g)
    printed = capsys.readouterr().out
    assert "first half yes, second half yes" in printed
    assert "gap >15 s" in printed and "true D" not in printed  # no labels on the match
    assert "public five minutes" in printed and "SHA-256" in printed


def test_a_model_file_without_a_decoder_stops_before_reading_the_match(tmp_path, monkeypatch,
                                                                       capsys):
    X = np.random.default_rng(1).normal(size=(200, len(DESIGN_NAMES)))
    models_path = str(tmp_path / "models.json")
    save({"dead": fit(X, X[:, 0] > 0), "team": fit(X, X[:, 1] > 0)}, models_path)
    monkeypatch.setattr(games, "DATA_DIR", str(tmp_path / "nowhere"))
    monkeypatch.setattr(sys, "argv", ["make_submission.py", "--models", models_path])
    assert make_submission.run_safely(make_submission.main) == 1
    assert "no decoder yet" in capsys.readouterr().out
