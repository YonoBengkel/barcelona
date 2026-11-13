"""Run one job of this repository on Kaggle and pack what it made into kaggle-results.zip.

kaggle/cell.py clones the repository to /tmp/barcelona and calls this file with the job: a script
and its arguments, relative to the repository root. The steps:

1. build the pinned environment, Python 3.12.4 with requirements.txt, using uv, plus the job's
   own extra requirements when EXTRA_REQUIREMENTS names some (PyTorch for scripts/train_tcn.py,
   kept within requirements.txt's pins);
2. fetch the IDSSE files, pinned and checked (scripts/fetch_idsse.py);
3. run the tests, so a broken environment stops the run here;
4. run the job;
5. zip every file the job created or changed, at its path in the repository, together with the
   log and a manifest under runs/<time>-<script>/. Unzipped at the repository root on the
   laptop, everything lands where it belongs.

The zip is written even when a step fails, so the log always comes back. Only public data is used
here: the match data never comes to Kaggle. Standard library only, so Kaggle's own Python runs it.

    python kaggle/run.py "scripts/train_models.py --seeds 2"
"""

import hashlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
import zipfile

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV = "/tmp/venv"
PYTHON = os.path.join(VENV, "bin", "python")
PYTHON_VERSION = "3.12.4"
RESULTS = "/kaggle/working/kaggle-results.zip"
# Requirements a job needs beyond requirements.txt, which the submission alone needs (M11l).
EXTRA_REQUIREMENTS = {"scripts/train_tcn.py": "requirements-torch.txt",
                      "scripts/train_tcn_ensemble.py": "requirements-torch.txt",
                      "scripts/train_tcn_variants.py": "requirements-torch.txt"}


class Log:
    """Echo to the notebook and keep a copy for the zip."""

    def __init__(self):
        self.parts = []

    def write(self, text):
        sys.stdout.write(text)
        sys.stdout.flush()
        self.parts.append(text)


def run(log, argv, env):
    """Run argv in the repository, echoing and logging its output; return its exit code."""
    log.write("\n$ {}\n".format(shlex.join(argv)))
    child = subprocess.Popen(argv, cwd=CODE, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, errors="replace")
    for line in child.stdout:
        log.write(line)
    return child.wait()


def capture(argv):
    """A command's output, or '' if it cannot run."""
    try:
        return subprocess.run(argv, cwd=CODE, capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def outputs():
    """Files the job created or changed: modified tracked files, and new files git keeps."""
    listed = capture(["git", "ls-files", "-z", "--modified", "--others", "--exclude-standard"])
    return sorted({path for path in listed.split("\0")
                   if path and os.path.isfile(os.path.join(CODE, path))})


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    command = sys.argv[1]
    job = shlex.split(command)
    if job[0] in ("python", "python3"):
        job = job[1:]
    started = time.gmtime()
    folder = "runs/{}-{}".format(time.strftime("%Y%m%dT%H%MZ", started),
                                 os.path.splitext(os.path.basename(job[0]))[0])
    # The children get no Kaggle credentials and no Python settings from Kaggle's own environment.
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("PYTHON", "KAGGLE_"))}
    env["PYTHONUNBUFFERED"] = "1"

    log = Log()
    manifest = {"command": command, "commit": capture(["git", "rev-parse", "HEAD"]),
                "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", started),
                "cpus": os.cpu_count(), "platform": platform.platform(),
                "gpu": capture(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]),
                "steps": []}
    log.write("commit {}\njob {}\n{} CPUs, {:.0f} GB free in /tmp, GPU {}\n".format(
        manifest["commit"], command, os.cpu_count(), shutil.disk_usage("/tmp").free / 1e9,
        manifest["gpu"] or "none"))

    install = [sys.executable, "-m", "uv", "pip", "install", "--quiet", "--python", PYTHON]
    steps = [
        ("install uv", [sys.executable, "-m", "pip", "install", "--quiet", "uv"]),
        ("create the environment", [sys.executable, "-m", "uv", "venv", "--quiet",
                                    "--python", PYTHON_VERSION, VENV]),
        ("install the requirements", install + ["-r", "requirements.txt"]),
    ]
    if job[0] in EXTRA_REQUIREMENTS:
        steps.append(("install the job's extra requirements",
                      install + ["-r", EXTRA_REQUIREMENTS[job[0]], "-c", "requirements.txt"]))
    steps += [
        ("fetch IDSSE", [PYTHON, "scripts/fetch_idsse.py"]),
        ("run the tests", [PYTHON, "-m", "pytest", "-q", "-p", "no:cacheprovider"]),
        ("run the job", [PYTHON] + job),
    ]
    shutil.rmtree(VENV, ignore_errors=True)  # left over when the cell is run twice in a session
    code = 1
    try:
        for name, argv in steps:
            start = time.time()
            code = run(log, argv, env)
            manifest["steps"].append({"step": name, "exit_code": code,
                                      "seconds": round(time.time() - start, 1)})
            if code != 0:
                log.write("\n{}: failed with exit code {}\n".format(name, code))
                break
    except Exception as error:  # still pack the log below
        log.write("\nrunner error: {}: {}\n".format(type(error).__name__, error))
        code = 1
    finally:
        manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        manifest["exit_code"] = code
        manifest["uv"] = capture([sys.executable, "-m", "uv", "--version"])
        manifest["python"] = capture([PYTHON, "-c", "import sys; print(sys.version.split()[0])"])
        manifest["packages"] = capture([sys.executable, "-m", "uv", "pip", "freeze",
                                        "--python", PYTHON]).split()
        files = outputs()
        manifest["outputs"] = {path: {"bytes": os.path.getsize(os.path.join(CODE, path)),
                                      "sha256": sha256(os.path.join(CODE, path))}
                               for path in files}
        with zipfile.ZipFile(RESULTS, "w", zipfile.ZIP_DEFLATED) as results:
            for path in files:
                results.write(os.path.join(CODE, path), path)
            results.writestr(folder + "/log.txt", "".join(log.parts))
            results.writestr(folder + "/manifest.json",
                             json.dumps(manifest, indent=1, sort_keys=True) + "\n")
        print("\nwrote {}: {} output file(s), log and manifest under {}/".format(
            RESULTS, len(files), folder))
    return code


if __name__ == "__main__":
    sys.exit(main())
