"""Pytest bootstrap: make the repository root and the published reference importable.

``original/`` holds the files as published on this branch; ``tests/test_branches.py``
imports them to check that the library reproduces the original implementation
branch by branch, so the directory has to be on ``sys.path``.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for path in (ROOT, ROOT / "original"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
