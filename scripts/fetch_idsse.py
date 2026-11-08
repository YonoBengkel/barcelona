"""Download the seven IDSSE matches from the Hugging Face mirror, pinned and checked.

The files land in external/idsse/ under their own names, where possession.external reads them.
REVISION pins the mirror's commit, and every file is checked against the size and digest the
mirror lists for it, so a run on another machine (Kaggle) fits on exactly the bytes used on the
team's laptop. Files already present and intact are kept. Standard library only.

    python scripts/fetch_idsse.py [--dir external/idsse]
"""

import argparse
import hashlib
import os
import shutil
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = "https://huggingface.co/datasets/pysport/idsse-data/resolve/{}/{}"
REVISION = "a715a38dfbaf5f58e431727c2b78d174101a703c"  # last changed 2026-03-04

# name -> (bytes, digest). The position files are Git LFS objects, listed by their SHA-256; the
# others are plain Git blobs, listed by their Git object id: the SHA-1 of "blob <bytes>\0" and
# the content.
FILES = {
    "DFL_02_01_matchinformation_DFL-COM-000001_DFL-MAT-J03WMX.xml":
        (11930, "8f2a82d84428c28e0e01f0a908c8395cc8d23900"),
    "DFL_02_01_matchinformation_DFL-COM-000001_DFL-MAT-J03WN1.xml":
        (11722, "45d2f112aed6c4212e66db8a91e8da1b8d728090"),
    "DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WOH.xml":
        (11745, "a693a0cacc2b597357e65a04eec5febfdaff79aa"),
    "DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WOY.xml":
        (11952, "bce56e5f518216e62d1e93a38322d4b9e357e5ab"),
    "DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WPY.xml":
        (11922, "46f1ffe15e1b96b12aab10594b12886c503c2f68"),
    "DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WQQ.xml":
        (11932, "7ec8acbff5a0c6167b632cc230d2a8b0e7b44b1c"),
    "DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WR9.xml":
        (11823, "7a8eb84881a1631da510982c01c90a3afd1b2e23"),
    "DFL_03_02_events_raw_DFL-COM-000001_DFL-MAT-J03WMX.xml":
        (776578, "bc2f5265763786ac8c05cd68595acc614deee50c"),
    "DFL_03_02_events_raw_DFL-COM-000001_DFL-MAT-J03WN1.xml":
        (612114, "67b982007b8e3f5de83dcd104779acc92b76a813"),
    "DFL_03_02_events_raw_DFL-COM-000002_DFL-MAT-J03WOH.xml":
        (635747, "c651487e6d0defbc033f87fadb1b5ebfcad75037"),
    "DFL_03_02_events_raw_DFL-COM-000002_DFL-MAT-J03WOY.xml":
        (683018, "9d2bde37ff79a5a9f993882bc1d83aef70a51d8e"),
    "DFL_03_02_events_raw_DFL-COM-000002_DFL-MAT-J03WPY.xml":
        (698898, "c8065cd1464a67a0312a7788bc6da6442084d141"),
    "DFL_03_02_events_raw_DFL-COM-000002_DFL-MAT-J03WQQ.xml":
        (732186, "081d5489af3437a7e3dbe7219226136bf61f7fd4"),
    "DFL_03_02_events_raw_DFL-COM-000002_DFL-MAT-J03WR9.xml":
        (681050, "1c9cc3875850b25837d37d7789554f3112ed1b76"),
    "DFL_04_03_positions_raw_observed_DFL-COM-000001_DFL-MAT-J03WMX.xml":
        (418524833, "d3dd92e509421a6466acb0a083c009035f1751bc5fd9ef42ffa8f20d01cf7d49"),
    "DFL_04_03_positions_raw_observed_DFL-COM-000001_DFL-MAT-J03WN1.xml":
        (391722803, "56ef948961a18e150dba0b7b16869d0008ff66f3050a0c355d31fa5b8ec2fcec"),
    "DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WOH.xml":
        (348945715, "ec4dcf30200aad694dea263d5289485fcd0ca42badddaeb7c81f5cd5a4229d4f"),
    "DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WOY.xml":
        (362567859, "a228ef0e676d009cefff1580fcb721b53f598454eb120ec347e0a628a3a4646b"),
    "DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WPY.xml":
        (371688499, "845202c29ad93d696653436da426a7cdaefb848298e07b7343de6529581671e1"),
    "DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WQQ.xml":
        (356219423, "cbefcb0bd7f39f3b94ee69a7bc97aac0f936425062c1cc1b144733df36804d6f"),
    "DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WR9.xml":
        (372934140, "70b63069189019066fb7e3d9db891122edfe787f3a43ba522ebe45ed89ea0269"),
}


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
    request = urllib.request.Request(url, headers={"User-Agent": "fetch_idsse"})
    with urllib.request.urlopen(request, timeout=60) as response, open(part, "wb") as out:
        shutil.copyfileobj(response, out, 1 << 20)
    os.replace(part, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", default=os.path.join(ROOT, "external", "idsse"))
    args = parser.parse_args()
    os.makedirs(args.dir, exist_ok=True)

    fetched = 0
    for name, (size, expected) in FILES.items():
        path = os.path.join(args.dir, name)
        if intact(path, size, expected):
            continue
        for _ in range(3):
            try:
                download(URL.format(REVISION, name), path)
            except OSError as error:
                print("retrying {}: {}".format(name, type(error).__name__))
                continue
            if intact(path, size, expected):
                break
            print("retrying {}: size or digest does not match".format(name))
        else:
            print("FAILED: {}".format(name))
            return 1
        fetched += 1
    print("IDSSE mirror at {}: {} files fetched, {} already intact".format(
        REVISION[:12], fetched, len(FILES) - fetched))
    return 0


if __name__ == "__main__":
    sys.exit(main())
