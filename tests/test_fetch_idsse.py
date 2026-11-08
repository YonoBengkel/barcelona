"""The checks scripts/fetch_idsse.py runs on every file it downloads."""

import importlib.util
import os

from possession.external import MATCHES, paths

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "fetch_idsse", os.path.join(ROOT, "scripts", "fetch_idsse.py"))
fetch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fetch)

HELLO = b"hello\n"
HELLO_SHA256 = "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03"
HELLO_BLOB = "ce013625030ba8dba906f756967f9e9ca394464a"  # git hash-object of the same bytes


def test_intact_checks_size_and_both_digest_forms(tmp_path):
    path = str(tmp_path / "file")
    with open(path, "wb") as f:
        f.write(HELLO)
    assert fetch.intact(path, len(HELLO), HELLO_SHA256)
    assert fetch.intact(path, len(HELLO), HELLO_BLOB)
    assert not fetch.intact(path, len(HELLO) + 1, HELLO_BLOB)
    assert not fetch.intact(path, len(HELLO), "0" * 64)
    assert not fetch.intact(str(tmp_path / "missing"), len(HELLO), HELLO_SHA256)


def test_every_file_the_reader_needs_is_pinned():
    needed = {os.path.basename(p) for match in MATCHES for p in paths(match).values()}
    assert set(fetch.FILES) == needed
