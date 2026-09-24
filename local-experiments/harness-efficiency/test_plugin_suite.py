"""Run standalone plugin tests through Hermes's hermetic test runner.

The standalone directory has a hyphenated plugin identifier and a relative
package entrypoint. Pytest package collection is not its discovery contract;
its stdlib suite exercises real PluginManager discovery instead.
"""
import os
from pathlib import Path
import re
import subprocess
import sys


def test_standalone_plugin_suite():
    root = Path(__file__).resolve().parents[2]
    plugin = root / 'local-plugins' / 'task-accounting'
    for directory in ('.test-runtime', 'evidence'):
        (plugin / directory).mkdir(exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE='1')
    result = subprocess.run(
        [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
        cwd=plugin, env=env, text=True, capture_output=True, timeout=60,
    )
    evidence = result.stdout + result.stderr
    print(evidence)
    assert result.returncode == 0, evidence
    count = re.search(r'Ran (\d+) tests?', evidence)
    assert count and int(count.group(1)) > 0, 'No standalone tests ran'
