"""A scripted engine for the WP-13.4 witnesses (selected only through the explicit, uncertified
`EngineOverride`). It records the argv it received and plays back a fixed output.

Environment (inherited, as Pi's spawn inherits it):
    FAKE_ENGINE_LOG      append one JSON line {"argv": [...]} per invocation
    FAKE_ENGINE_STDOUT_FILE  a file whose exact bytes are written to stdout (default: none)
    FAKE_ENGINE_STDERR   text to write to stderr (default: none)
    FAKE_ENGINE_EXIT     exit code (default: 0)
    FAKE_ENGINE_SLEEP    seconds to sleep after writing stdout, before exiting (default: 0)
"""

import json
import os
import sys
import time

log = os.environ.get("FAKE_ENGINE_LOG")
if log:
    with open(log, "a", encoding="utf-8") as out:
        out.write(json.dumps({"argv": sys.argv[1:]}, ensure_ascii=False) + "\n")
stdout_file = os.environ.get("FAKE_ENGINE_STDOUT_FILE")
with open(stdout_file, "rb") if stdout_file else open(os.devnull, "rb") as source:
    data = source.read()
sys.stdout.buffer.write(data)
sys.stdout.buffer.flush()
sys.stderr.write(os.environ.get("FAKE_ENGINE_STDERR", ""))
sys.stderr.flush()
time.sleep(float(os.environ.get("FAKE_ENGINE_SLEEP", "0")))
sys.exit(int(os.environ.get("FAKE_ENGINE_EXIT", "0")))
