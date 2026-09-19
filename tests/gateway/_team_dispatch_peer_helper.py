"""Boot the actual CLI with only synthetic external HTTP responses."""
import json
import os
from pathlib import Path
import runpy
import socket
import sys
import faulthandler

import pytest

capture = Path(sys.argv.pop(1))
mode = sys.argv.pop(1)
# Install HTTP fixtures after selecting the target, as the CLI does before
# importing profile-bound stores. Do not bind caches to the caller profile.
os.environ["HERMES_HOME"] = str(Path(os.environ["HOME"]) / ".hermes" / "profiles" / "product")
from tests.gateway.test_team_dispatch_runtime import model
faulthandler.dump_traceback_later(8)
patches = pytest.MonkeyPatch()
def no_network(*a, **kw):
    raise AssertionError("offline peer test forbids external sockets")
patches.setattr(socket.socket, "connect", no_network)
calls = model.__wrapped__(patches)
calls.final = "PUBLIC_PEER_RESULT"
if mode == "read":
    calls.tool = ("read_file", {"path": str(capture.parent / "public-brief.txt")})
elif mode == "approval":
    calls.tool = ("terminal", {"command": "rm -rf " + str(capture.parent / "approval-canary"), "timeout": 5})
sys.argv[0] = "hermes"
try:
    runpy.run_module("hermes_cli.main", run_name="__main__")
finally:
    capture.write_text(json.dumps({"calls": calls, "profile_home": os.environ.get("HERMES_HOME"),
                                   "pid": os.getpid()}, indent=2))
