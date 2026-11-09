"""The ATD reader on a synthetic three-slot export: placement, units, and quiet errors."""

import numpy as np
import pytest

from possession.atd import halves, metadata_attack_directions, read_atd, team_of_slots

META = """<?xml version="1.0"?>
<Metadata><Teams><Team id="T1"><Name>Home</Name></Team><Team id="T2"><Name>Away</Name></Team>
</Teams><Players><Player id="P1" teamId="T1"/><Player id="P2" teamId="T2"/>
<Player id="P3" teamId="T1"/></Players></Metadata>"""

RAW = ("1:0.5,0.5,1.0;0.25,0.75,2.0;NaN,NaN,NaN:0.5,0.5,0.0\n"
       "2:0.6,0.5,1.0;0.25,0.75,2.0;0.1,0.1,0.5:NaN,NaN,NaN\n"
       "4:0.5,0.5,1.0;0.25,0.75,2.0;0.1,0.1,0.5:0.75,0.25,3.0\n")
PITCH = (100.0, 50.0)


@pytest.fixture
def files(tmp_path):
    meta, raw = tmp_path / "meta.xml", tmp_path / "raw.txt"
    meta.write_text(META)
    raw.write_text(RAW)
    return str(meta), str(raw)


def test_teams_follow_the_first_team_in_the_metadata(files):
    assert team_of_slots(files[0]).tolist() == [0, 1, 0]


def test_positions_are_metres_from_the_centre(files):
    atd = read_atd(*files, n_frames=4, pitch=PITCH)
    assert atd["ball_x"][0] == 0.0 and np.isnan(atd["ball_x"][1])
    assert atd["x"][1, 0] == pytest.approx(10.0)          # (0.6 - 0.5) * 100
    assert atd["y"][0, 1] == pytest.approx(12.5)          # (0.75 - 0.5) * 50
    assert np.isnan(atd["x"][0, 2])
    assert np.isnan(atd["x"][2]).all()                    # frame_id 3 is absent from the file
    assert atd["ball_x"][3] == pytest.approx(25.0)


def test_halves_slice_and_remember_their_first_frame(files):
    atd = read_atd(*files, n_frames=4, pitch=PITCH)
    first, second = halves(atd, [(1, 0, 1), (2, 3, 3)])
    assert len(first["ball_x"]) == 2 and second["first_frame"] == 3


def test_attack_directions_are_read_from_the_team_parameters(tmp_path, files):
    assert metadata_attack_directions(files[0]) == (0, 0)  # no parameter: unknown
    parameter = ("<ProviderTeamsParameters><ProviderParameter><Name>attack_direction_first_half"
                 "</Name><Value>{}</Value></ProviderParameter></ProviderTeamsParameters>")
    meta = tmp_path / "directions.xml"
    meta.write_text(META.replace("<Name>Home</Name>", "<Name>Home</Name>" + parameter.format(
        "right_to_left")).replace("<Name>Away</Name>", "<Name>Away</Name>" + parameter.format(
            "left_to_right")))
    assert metadata_attack_directions(str(meta)) == (-1, 1)


def test_a_malformed_line_fails_without_quoting_it(tmp_path, files):
    bad = tmp_path / "bad.txt"
    bad.write_text("1:0.123,0.5,1.0;0.25,0.75,2.0:0.5,0.5,0.0\n")  # two slots, not three
    with pytest.raises(ValueError) as error:
        read_atd(files[0], str(bad), n_frames=2, pitch=PITCH)
    assert "0.123" not in str(error.value)
