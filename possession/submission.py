"""The submission file: one frame,label row per video frame of the tactical recording.

Rows inside the graded halves carry the decoder's H, A or D; rows outside them carry X, which the
scorer never grades. The file is written with LF line endings on every platform, because the
organisers regenerate it from this repository and compare bytes (issue #11), and Python on
Windows would otherwise write CRLF.
"""

import numpy as np


def assemble(n_frames, parts):
    """(n_frames,) labels: each (first_frame, labels) part placed at its frames, X elsewhere."""
    out = np.full(n_frames, "X", dtype="<U1")
    for first, labels in parts:
        if first < 0 or first + len(labels) > n_frames:
            raise ValueError("a part reaches outside the {} frames".format(n_frames))
        out[first:first + len(labels)] = labels
    return out


def write(path, labels):
    """Write labels as frame,label rows from frame 0, LF line endings; return the path."""
    with open(path, "w", newline="\n", encoding="ascii") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, label) for frame, label in enumerate(labels))
    return path
