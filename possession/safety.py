"""Run code that reads the match data without letting its contents reach the console.

The terms of the competition forbid disclosing the match data, or any extract of it, to a hosted
LLM, and the console output of a script is exactly what gets pasted into one. A traceback is the
easy way to break that by accident: a parse error quotes the offending line of tracking, and a
KeyError names the value it could not find. Every script that reads the match data therefore runs
its entry point through run_safely(), which reports the type of an exception and where it
happened, never its message.

Set MTAH_SHOW_ERRORS=1 to see the full traceback on your own screen while debugging, and do not
paste that one anywhere.
"""

import os
import sys
import traceback

# Opt-in for the full traceback, local debugging only.
SHOW_ERRORS_ENV = "MTAH_SHOW_ERRORS"


def describe(error):
    """'ValueError at io.py:42 in parse_line <- run.py:7 in main': type and location only.

    File names are reduced to their base name: enough to find the line, and a path can carry
    nothing worth hiding anyway.
    """
    frames = traceback.extract_tb(error.__traceback__)
    where = " <- ".join("{}:{} in {}".format(os.path.basename(frame.filename), frame.lineno,
                                             frame.name)
                        for frame in reversed(frames))
    return "{} at {}".format(type(error).__name__, where or "?")


def run_safely(main):
    """Call main() and return its exit code; on an exception print describe() and return 1.

    Catches Exception, not BaseException, so Ctrl+C and sys.exit() behave as usual.
    """
    try:
        return main() or 0
    except Exception as error:  # any exception: every one of them may carry data in its message
        if os.environ.get(SHOW_ERRORS_ENV) == "1":
            raise
        print("FAILED  {}".format(describe(error)), file=sys.stderr)
        print("        message withheld because it may quote the match data; set {}=1 to see it "
              "on your own screen".format(SHOW_ERRORS_ENV), file=sys.stderr)
        return 1
