"""Properties of the match bundle: its structure and format, never its content.

What kind of data each file holds, not what it says (AGENTS.md):

- every file: its size, and with --checksums its SHA-256 (so the bundle can be shown to match the
  organisers' copy);
- videos: container and stream descriptors from ffprobe -- codec, resolution, frame rate, frame
  count, duration, start time and time base, audio format -- and none of the metadata tags;
- the tracking metadata: the values of the parameters that describe the format only (half
  boundaries, data type, tracking source, version, field size); never team, player or match ones;
- XML files: element paths with their counts and attribute names; values only for a whitelist of
  format descriptors (channel ids, units, frame rate, separators, register names), with digits
  masked in per-player channel references;
- the tracking text: how a line is laid out -- parts, groups, fields per group, NaN shares -- and,
  for a field that is not numeric, its labels only when there are at most MAX_LABELS of them;
- the homography CSV: header and row count.

    python scripts/data_properties.py [--checksums]
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from games import asset_paths  # noqa: E402
from possession.safety import run_safely  # noqa: E402

FFPROBE = shutil.which("ffprobe") or r"C:\ffmpeg\bin\ffprobe.exe"
FFPROBE_ENTRIES = ("format=format_name,duration,size,bit_rate,nb_streams,start_time:"
                   "stream=index,codec_type,codec_name,profile,width,height,pix_fmt,"
                   "r_frame_rate,avg_frame_rate,nb_frames,duration,bit_rate,sample_rate,"
                   "channels,channel_layout,start_time,time_base")
# Provider parameters that describe the export, not the match: printed with their values.
FORMAT_PARAMETERS = {"first_half_start", "first_half_end", "second_half_start", "second_half_end",
                     "first_extra_half_start", "first_extra_half_end", "second_extra_half_start",
                     "second_extra_half_end", "data_type", "tracking_source", "version"}
TEXT_VALUES = {"FrameRate", "Unit", "TrackingType", "ProviderName"}
NAME_UNDER = {"Channel", "Sensor", "ProviderParameter"}
VALUE_ATTRS = {"id", "channelId", "playerChannelId", "separator", "startFrame", "endFrame",
               "name"}
MAX_VALUES = 12
MAX_LABELS = 10


def describes_format(tag, attr):
    """Attribute values that name a column, a unit or a separator, never a person or an event."""
    return attr in VALUE_ATTRS and any(key in tag for key in ("Channel", "Register", "DataFormat",
                                                              "Sensor"))


def xml_structure(path):
    counts, attrs, values, texts = Counter(), defaultdict(set), defaultdict(set), defaultdict(set)
    stack = []
    for event, elem in ET.iterparse(path, events=("start", "end")):
        if event == "start":
            stack.append(elem.tag)
            where = "/".join(stack)
            counts[where] += 1
            for name, value in elem.attrib.items():
                attrs[where].add(name)
                if describes_format(elem.tag, name):
                    if name not in ("separator", "startFrame", "endFrame"):
                        value = re.sub(r"\d+", "#", value)  # ids may embed a track number
                    values[(where, name)].add(value)
        else:
            where = "/".join(stack)
            text = (elem.text or "").strip()
            parent = stack[-2] if len(stack) > 1 else ""
            if text and (elem.tag in TEXT_VALUES or (elem.tag == "Name" and parent in NAME_UNDER)):
                texts[where].add(text)
            stack.pop()
            elem.clear()
    for where in sorted(counts):
        line = "  {:>7} x {}".format(counts[where], where)
        if attrs[where]:
            line += "  [" + ", ".join(sorted(attrs[where])) + "]"
        print(line)
        for (place, name), found in sorted(values.items()):
            if place == where:
                print("            {} = {}".format(name, _few(found)))
        if where in texts:
            print("            text = {}".format(_few(texts[where])))


def _few(found):
    found = sorted(found)
    head = ", ".join(found[:MAX_VALUES])
    return head + (" ... ({} distinct)".format(len(found)) if len(found) > MAX_VALUES else "")


def _numeric(token):
    try:
        float(token)
        return True
    except ValueError:
        return False


def text_layout(path):
    """Line layout of the raw tracking: counts and shares, labels only for small categories."""
    parts, groups, player_fields, ball_fields = Counter(), Counter(), Counter(), Counter()
    nan = defaultdict(lambda: [0, 0])          # (where, index) -> [nan, total]
    labels = defaultdict(Counter)              # (where, index) -> non-numeric tokens
    lines = 0
    with open(path) as handle:
        for line in handle:
            pieces = line.rstrip("\n").split(":")
            parts[len(pieces)] += 1
            if len(pieces) != 3:
                continue
            lines += 1
            slots = pieces[1].split(";")
            groups[len(slots)] += 1
            for slot in slots:
                fields = slot.split(",")
                player_fields[len(fields)] += 1
                _tally("player", fields, nan, labels)
            ball = pieces[2].split(",")
            ball_fields[len(ball)] += 1
            _tally("ball", ball, nan, labels)
    print("  lines with 3 ':'-separated parts: {} (parts per line: {})".format(lines, dict(parts)))
    print("  player groups per line: {}".format(dict(groups)))
    print("  fields per player group: {}".format(dict(player_fields)))
    print("  fields per ball group: {}".format(dict(ball_fields)))
    for (where, index), (missing, total) in sorted(nan.items()):
        found = labels[(where, index)]
        kind = "numeric" if not found else (
            "non-numeric labels {}".format(sorted(found)) if len(found) <= MAX_LABELS
            else "non-numeric, more than {} distinct (not shown)".format(MAX_LABELS))
        print("  {} field {}: NaN {:.1%}, {}".format(where, index, missing / total, kind))


def _tally(where, fields, nan, labels):
    for index, token in enumerate(fields):
        cell = nan[(where, index)]
        cell[1] += 1
        if token == "NaN":
            cell[0] += 1
        elif not _numeric(token):
            if len(labels[(where, index)]) <= MAX_LABELS:
                labels[(where, index)][token] += 1


def format_parameters(path):
    """{name: value} for FORMAT_PARAMETERS and the field size; nothing about teams or players."""
    root = ET.parse(path).getroot()
    out = {}
    for parameter in root.iter("ProviderParameter"):
        name = (parameter.findtext("Name") or "").strip()
        if name in FORMAT_PARAMETERS:
            out[name] = (parameter.findtext("Value") or "").strip()
    size = root.find(".//FieldSize")
    if size is not None:
        out["field_size"] = "{} x {}".format((size.findtext("Width") or "").strip(),
                                             (size.findtext("Height") or "").strip())
    return out


def sha256(path, chunk=8 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def video_properties(path):
    result = subprocess.run([FFPROBE, "-v", "error", "-show_entries", FFPROBE_ENTRIES, "-of",
                             "json", path], capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)
    print("  format: {}".format(info.get("format", {})))
    for stream in info.get("streams", []):
        print("  stream: {}".format(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checksums", action="store_true",
                        help="also print SHA-256 per file (reads all 19 GB of video)")
    args = parser.parse_args()
    for source in ("tactical", "broadcast"):
        print("=" * 78)
        print(source)
        for asset, path in sorted(asset_paths(source=source).items()):
            if not os.path.isfile(path):
                print("\n{}: missing".format(asset))
                continue
            print("\n{}: {:.1f} MB".format(asset, os.path.getsize(path) / 1e6))
            if args.checksums:
                print("  sha256: {}".format(sha256(path)))
            if asset == "video":
                video_properties(path)
            elif path.endswith(".xml"):
                xml_structure(path)
                if asset == "atd_meta":
                    print("  format parameters:")
                    for name, value in sorted(format_parameters(path).items()):
                        print("    {} = {}".format(name, value))
            elif asset == "atd_raw":
                text_layout(path)
            elif asset == "homography":
                with open(path) as handle:
                    header = handle.readline().strip()
                    rows = sum(1 for _ in handle)
                print("  header: {}\n  rows: {}".format(header, rows))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(run_safely(main))
