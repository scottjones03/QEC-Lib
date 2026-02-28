"""Shared process-management utilities for the trapped-ion compiler pipeline.

Extracted from best_effort_compilation_WISE.py and qccd_SAT_WISE_odd_even_sorter.py
(H10: de-duplicate _get_child_pids).
"""

from __future__ import annotations

import subprocess
from typing import Set


def get_child_pids(parent_pid: int) -> Set[int]:
    """Return the set of direct child PIDs of *parent_pid*.

    Uses ``pgrep -P`` which works on both macOS and Linux.
    Returns an empty set on any error (process not found, timeout, etc.).
    """
    children: Set[int] = set()
    try:
        result = subprocess.run(
            ["pgrep", "-P", str(parent_pid)],
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                stripped = line.strip()
                if stripped:
                    try:
                        children.add(int(stripped))
                    except ValueError:
                        pass
    except Exception:
        pass
    return children
