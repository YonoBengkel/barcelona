"""possession.skillcorner and scripts/lab/cross_provider.py: 10 fps resampled to 25, a synthetic
match file pair read both ways and cached, the digest scheme of the pinned files, and the fast
network against possession.tcn's exact one."""

import hashlib
import importlib.util
import json
import os

import numpy as np

from possession import practice
from possession import skillcorner as sc
from possession import tcn
from possession.models import log_state_probabilities

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _script(name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(ROOT, "scripts", "lab", name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cross_provider = _script("cross_provider")
skillcorner_training = _script("skillcorner_training")
skillcorner_diagnose = _script("skillcorner_diagnose")
pm = practice.pm


def test_ten_fps_samples_are_interpolated_to_twenty_five_and_gaps_stay_missing():
    assert sc.frames_at_fps(3) == 6 and sc.frames_at_fps(2) == 3 and sc.frames_at_fps(0) == 0
    assert np.allclose(sc.resampled(np.array([0.0, 10.0, 20.0]), 6), [0, 4, 8, 12, 16, 20])
    gap = sc.resampled(np.array([0.0, np.nan, 20.0]), 6)
    assert gap[0] == 0 and gap[5] == 20 and np.isnan(gap[1:5]).all()
    two = sc.resampled(np.array([[0.0, 1.0], [10.0, np.nan]]), 3)
    assert np.allclose(two[:, 0], [0, 4, 8])
    assert two[0, 1] == 1 and np.isnan(two[1:, 1]).all()


def test_labels_and_flags_come_from_the_nearest_sample():
    assert "".join(sc.nearest_sample(np.array(list("HAD")), 6)) == "HHAADD"
    assert sc.nearest_sample(np.array([True, False]), 3).tolist() == [True, True, False]


def frame(number, period, group=None, ball=None, players=()):
    """One line of a SkillCorner tracking file."""
    x, y, seen = ball if ball else (None, None, None)
    return {"frame": number, "period": period, "timestamp": None,
            "possession": {"player_id": None, "group": group},
            "ball_data": {"x": x, "y": y, "z": 0.1 if ball else None, "is_detected": seen},
            "player_data": [{"player_id": p, "x": px, "y": py, "is_detected": s}
                            for p, px, py, s in players],
            "image_corners_projection": {}}


def write_match(root, match_id="1"):
    """A home player (id 1) and an away player (id 2); period 1 at frames 2-4, period 2 at 5-6."""
    folder = root / match_id
    folder.mkdir(parents=True)
    match = {"home_team": {"id": 10}, "away_team": {"id": 20}, "pitch_length": 104,
             "pitch_width": 68, "players": [{"id": 2, "team_id": 20}, {"id": 1, "team_id": 10}],
             "match_periods": [{"period": 1, "start_frame": 2, "end_frame": 4},
                               {"period": 2, "start_frame": 5, "end_frame": 6}]}
    lines = [frame(0, None),
             frame(2, 1, "home team", (1.0, 2.0, True), [(1, 0.0, 0.0, True), (2, 5.0, 5.0, False)]),
             frame(3, 1, "away team", (2.0, 2.0, False), [(1, 1.0, 0.0, True), (2, 6.0, 5.0, True)]),
             frame(4, 1),
             frame(5, 2, "home team", (0.0, 0.0, True), [(1, 0.0, 1.0, True)]),
             frame(6, 2, None, (0.0, 1.0, True), [(1, 0.0, 2.0, True)])]
    match_name, tracking_name = sc.names(match_id)
    (folder / match_name).write_text(json.dumps(match), encoding="utf-8")
    (folder / tracking_name).write_text("".join(json.dumps(line) + "\n" for line in lines),
                                        encoding="utf-8")


def test_a_match_is_read_resampled_both_ways_and_cached(tmp_path):
    write_match(tmp_path)
    info, periods = sc.load_match("1", root=str(tmp_path))
    assert info["pitch"] == (104.0, 68.0) and info["team"] == {1: sc.HOME, 2: sc.AWAY}
    assert info["periods"] == [(1, 2, 4), (2, 5, 6)]
    first, second = periods
    assert "".join(first["labels"]) == "HHAADD" and "".join(second["labels"]) == "HHD"
    assert first["tracked"].tolist() == [True, True, True, True, False, False]
    half = first["half"]
    assert half["team"].tolist() == [sc.HOME, sc.AWAY]
    assert np.allclose(half["ball_x"][:3], [1.0, 1.4, 1.8]) and np.isnan(half["ball_x"][3:]).all()
    assert np.allclose(half["x"][:3, 1], [5.0, 5.4, 5.8]) and np.isnan(half["x"][3:, 1]).all()
    assert os.path.isfile(os.path.join(str(tmp_path), "cache", "1.v{}.npz".format(
        sc.CACHE_VERSION)))

    _, detected = sc.load_match("1", root=str(tmp_path), detected_only=True)  # from the cache
    seen = detected[0]["half"]
    assert seen["ball_x"][0] == 1.0 and np.isnan(seen["ball_x"][1:]).all()
    assert np.isnan(seen["x"][:, 1]).all()  # player 2's one detection falls between frames
    assert np.allclose(seen["x"][:3, 0], half["x"][:3, 0])
    assert "".join(detected[1]["labels"]) == "HHD"


def test_the_digests_follow_the_pinned_scheme(tmp_path):
    path = tmp_path / "hello.txt"
    path.write_bytes(b"hello\n")
    assert sc.digest(str(path), 6, "ce013625030ba8dba906f756967f9e9ca394464a") \
        == "ce013625030ba8dba906f756967f9e9ca394464a"  # git hash-object of "hello\n"
    expected = hashlib.sha256(b"hello\n").hexdigest()
    assert sc.intact(str(path), 6, expected) and not sc.intact(str(path), 7, expected)
    assert len(sc.MATCHES) == 20
    for (match_size, match_digest), (tracking_size, tracking_digest) in sc.FILES.values():
        assert match_size > 0 and len(match_digest) == 40
        assert tracking_size > 0 and len(tracking_digest) == 64


def random_network(rng, columns, channels=5, dilations=(1, 2, 4)):
    """A network with random weights, shaped as scripts/train_tcn.py exports one."""
    def layer(outputs, inputs, taps):
        return {"weight": rng.normal(0, 0.4, (outputs, inputs, taps)).tolist(),
                "bias": rng.normal(0, 0.1, outputs).tolist()}

    return {"kind": "tcn", "mean": [0.0] * columns, "scale": [1.0] * columns,
            "input": layer(channels, 2 * columns, 1),
            "blocks": [{"dilation": d, "dilated": layer(channels, channels, 3),
                        "pointwise": layer(channels, channels, 1)} for d in dilations],
            "output": layer(2, channels, 1)}


def test_the_fast_network_equals_the_exact_one():
    rng = np.random.default_rng(0)
    model = random_network(rng, 4)
    X = rng.normal(size=(60, 4))
    X[3, 1] = np.nan
    fast, exact = cross_provider.fast_scores(model, X), tcn.scores(model, X)
    for a, b in zip(fast, exact):
        assert np.allclose(a, b, rtol=0, atol=1e-10)
    assert np.allclose(cross_provider.evidence({"tcn": model}, X),
                       log_state_probabilities({"tcn": model}, X), rtol=0, atol=1e-10)


def test_the_folds_hold_every_match_out_once():
    matches = [str(m) for m in range(20)]
    folds = skillcorner_training.folds_of(matches, 5)
    assert [len(fold) for fold in folds] == [4] * 5
    assert sorted(m for fold in folds for m in fold) == sorted(matches)
    assert folds[0] == ["0", "5", "10", "15"]


def test_the_gap_diagnostic_makes_only_untracked_frames_a_sure_dead_ball():
    log_p = np.log(np.array([[0.6, 0.3, 0.1], [0.2, 0.7, 0.1], [0.5, 0.4, 0.1]]))
    out = skillcorner_training.gaps_dead(log_p, np.array([True, False, True]))
    assert np.array_equal(out[[0, 2]], log_p[[0, 2]])
    assert out[1].tolist() == list(skillcorner_training.GAP_EVIDENCE)
    assert out[1].argmax() == pm.D


def test_labels_shift_within_a_half_and_the_confusion_counts_by_row():
    codes = np.array([pm.H, pm.A, pm.D], dtype=np.int8)
    assert skillcorner_diagnose.shifted(codes, 1).tolist() == [pm.X, pm.H, pm.A]
    assert skillcorner_diagnose.shifted(codes, -1).tolist() == [pm.A, pm.D, pm.X]
    gt, pred = np.array([pm.H, pm.H, pm.A, pm.D]), np.array([pm.H, pm.A, pm.A, pm.H])
    table = skillcorner_diagnose.confusion(gt, pred, np.array([True, True, True, False]))
    assert table.tolist() == [[1, 1, 0], [0, 1, 0], [0, 0, 0]]


def test_untracked_frames_go_ungraded_only_when_asked():
    halves = [{"labels": np.array(list("HHD")), "tracked": np.array([True, False, True])},
              {"labels": np.array(list("AD")), "tracked": np.array([True, True])}]
    every = cross_provider.truth(halves, "every frame")
    tracked = cross_provider.truth(halves, "tracked frames")
    assert every.tolist() == [pm.H, pm.H, pm.D, pm.X, pm.A, pm.D]
    assert tracked.tolist() == [pm.H, pm.X, pm.D, pm.X, pm.A, pm.D]
