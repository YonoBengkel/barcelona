"""The submission writer: X outside the graded halves, LF line endings, the organisers' format."""

import numpy as np
import pytest

from possession import practice
from possession.submission import assemble, write

pm = practice.pm


def test_parts_land_on_their_frames_and_the_rest_is_x():
    labels = assemble(8, [(1, np.array(["H", "A"])), (5, np.array(["D", "D", "H"]))])
    assert "".join(labels) == "XHAXXDDH"
    with pytest.raises(ValueError):
        assemble(4, [(3, np.array(["H", "A"]))])


def test_the_file_has_lf_endings_and_passes_the_organisers_validation(tmp_path):
    path = str(tmp_path / "submission.csv")
    write(path, assemble(5, [(1, np.array(["H", "A", "D"]))]))
    with open(path, "rb") as handle:
        assert handle.read() == b"frame,label\n0,X\n1,H\n2,A\n3,D\n4,X\n"
    assert pm.validate_submission(pm.read_label_csv(path), 5) is None
