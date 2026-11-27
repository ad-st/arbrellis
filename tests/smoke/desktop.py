#!/usr/bin/env python3
"""Launch the packaged desktop and wait for its real workspace to become ready."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=Path)
    args = parser.parse_args()
    executable = args.executable.resolve()
    if not executable.is_file():
        raise SystemExit(f"Desktop executable not found: {executable}")

    with tempfile.TemporaryDirectory(prefix="arbrellis-desktop-smoke-") as temporary:
        root = Path(temporary)
        ready_file = root / "ready.json"
        environment = {
            **os.environ,
            "HOME": str(root),
            "XDG_CONFIG_HOME": str(root / "config"),
            "XDG_CACHE_HOME": str(root / "cache"),
            "ARBRELLIS_SMOKE_READY_FILE": str(ready_file),
            "ARBRELLIS_SMOKE_AUTO_QUIT": "1",
        }
        process = subprocess.Popen(
            [str(executable), "--enable-logging=stderr"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
        )
        try:
            output, _ = process.communicate(timeout=120)
        except subprocess.TimeoutExpired:
            process.kill()
            output, _ = process.communicate()
            raise RuntimeError(f"Desktop smoke test timed out.\n{output}")

        if process.returncode != 0:
            raise RuntimeError(
                f"Desktop exited with status {process.returncode}.\n{output}"
            )
        if not ready_file.is_file():
            raise RuntimeError(f"Desktop never reported a ready workspace.\n{output}")
        payload = json.loads(ready_file.read_text())
        if payload.get("ready") is not True:
            raise RuntimeError(f"Invalid desktop readiness payload: {payload}")
        print("Packaged desktop smoke test passed.")


if __name__ == "__main__":
    main()
