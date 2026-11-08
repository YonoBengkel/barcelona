# Barcelona on Kaggle: paste this whole file into the only cell of a Kaggle notebook.
#
# Notebook settings, once (right-hand panel): Accelerator None; Internet on; under Add-ons >
# Secrets, a secret named GITHUB_TOKEN attached to this notebook, holding a fine-grained GitHub
# token that can only read YonoBengkel/barcelona. Each run: set REF and COMMAND, Save Version >
# Save & Run All, then download kaggle-results.zip from that version's Output. kaggle/README.md
# has the details.
#
# This cell only fetches the code and hands over to kaggle/run.py, so it never needs updating.
# The token goes to a single git command as a one-off header: it is never written to disk or
# printed, and the commands after the clone run without it.

REF = "main"                          # branch, tag or commit of YonoBengkel/barcelona
COMMAND = "scripts/train_models.py"   # the job: a script and its arguments, from the repo root

import base64
import shutil
import subprocess
import sys

from kaggle_secrets import UserSecretsClient

CODE = "/tmp/barcelona"  # outside /kaggle/working, so the code is not saved as output

token = UserSecretsClient().get_secret("GITHUB_TOKEN")
header = "Authorization: Basic " + base64.b64encode(
    ("YonoBengkel:" + token.strip()).encode()).decode()
shutil.rmtree(CODE, ignore_errors=True)
clone = subprocess.run(["git", "-c", "http.https://github.com/.extraheader=" + header, "clone",
                        "--quiet", "https://github.com/YonoBengkel/barcelona.git", CODE],
                       capture_output=True, text=True)
del token, header
if clone.returncode != 0:
    print(clone.stderr)
    raise RuntimeError("git clone failed: is Internet on, and is GITHUB_TOKEN attached and valid?")
checkout = subprocess.run(["git", "-C", CODE, "checkout", "--quiet", REF],
                          capture_output=True, text=True)
if checkout.returncode != 0:
    print(checkout.stderr)
    raise RuntimeError("no branch, tag or commit named " + REF)

job = subprocess.Popen([sys.executable, "-u", CODE + "/kaggle/run.py", COMMAND],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in job.stdout:
    print(line, end="")
if job.wait() != 0:
    raise RuntimeError("the job failed: the log above is also in kaggle-results.zip")
