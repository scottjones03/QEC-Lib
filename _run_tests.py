#!/usr/bin/env python3
"""Run test suites in clean subprocesses immune to stale SIGINT."""
import signal, subprocess, sys, os

signal.signal(signal.SIGINT, signal.SIG_IGN)

env = os.environ.copy()
env["PYTHONPATH"] = "src"

suites = [
    ("INTEGRATION",
     "src/qectostim/experiments/hardware_simulation/trapped_ion/demo/test_integration_layer.py"),
    ("E2E",
     "src/qectostim/experiments/hardware_simulation/trapped_ion/demo/test_e2e.py"),
]

for label, path in suites:
    print(f"\n{'='*60}")
    print(f"  {label}: {path}")
    print(f"{'='*60}")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", path, "--tb=short", "-q"],
        env=env,
        timeout=600,
        text=True,
        capture_output=True,
        preexec_fn=os.setsid,
    )
    out = result.stdout
    print(out[-2000:] if len(out) > 2000 else out)
    if result.stderr:
        print("STDERR:", result.stderr[-500:])
    print(f"EXIT CODE: {result.returncode}")
    if result.returncode != 0:
        print(f"*** {label} FAILED ***")
        break
