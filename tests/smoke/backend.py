#!/usr/bin/env python3
"""Exercise the frozen Arbrellis service without loading model weights."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def request(url: str, token: str | None = None) -> tuple[int, bytes]:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, headers=headers), timeout=2
        ) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=Path)
    args = parser.parse_args()
    executable = args.executable.resolve()
    if not executable.is_file():
        raise SystemExit(f"Frozen service not found: {executable}")

    with tempfile.TemporaryDirectory(prefix="arbrellis-backend-smoke-") as temporary:
        root = Path(temporary)
        port = free_port()
        token = "release-smoke-token"
        command = [
            str(executable),
            "--host", "127.0.0.1",
            "--port", str(port),
            "--auth-token", token,
            "--db", str(root / "smoke.sqlite3"),
        ]
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, "HOME": str(root)},
        )
        base = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 90
        try:
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    output = process.stdout.read() if process.stdout else ""
                    raise RuntimeError(f"Frozen service exited early.\n{output}")
                try:
                    status, body = request(f"{base}/api/health", token)
                    if status == 200 and json.loads(body)["status"] == "ok":
                        break
                except (OSError, ValueError, KeyError):
                    pass
                time.sleep(0.2)
            else:
                raise RuntimeError("Frozen service did not become healthy within 90 seconds.")

            assert request(f"{base}/api/health")[0] == 401
            assert request(f"{base}/api/graph", token)[0] == 200
            assert request(f"{base}/api/documents", token)[0] == 200
            assert request(f"{base}/api/citations?summary=1", token)[0] == 200
            print("Frozen backend smoke test passed.")
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


if __name__ == "__main__":
    main()
