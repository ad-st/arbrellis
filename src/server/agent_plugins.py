"""Replaceable conversational-agent providers for the knowledge graph experiment."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TextIO


DeltaCallback = Callable[[str], None]


@dataclass(frozen=True)
class AgentTurnResult:
    conversation_id: str
    answer: str


class AgentPlugin(ABC):
    """The application-facing contract implemented by every chat provider."""

    id: str
    name: str

    @abstractmethod
    def run_turn(
        self,
        conversation_id: str | None,
        prompt: str,
        on_delta: DeltaCallback,
    ) -> AgentTurnResult:
        """Run one conversational turn and stream visible answer deltas."""

    def metadata(self) -> dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "available": True,
            "capabilities": ["conversation", "streaming", "resume"],
        }


class CodexAgentPlugin(AgentPlugin):
    id = "codex"
    name = "Codex"

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace

    @staticmethod
    def executable() -> str:
        return os.environ.get("ARBRELLIS_CODEX_PATH", "codex")

    def metadata(self) -> dict[str, object]:
        available = shutil.which(self.executable()) is not None
        return {
            "id": self.id,
            "name": self.name,
            "available": available,
            "capabilities": ["conversation", "streaming", "resume"],
            "reason": None if available else "Install and authenticate the Codex CLI to use chat.",
        }

    @staticmethod
    def _write(writer: TextIO, message: dict[str, object]) -> None:
        writer.write(json.dumps(message, separators=(",", ":")) + "\n")
        writer.flush()

    @staticmethod
    def _read_response(reader: TextIO, request_id: int) -> dict[str, object]:
        while line := reader.readline():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if value.get("id") != request_id:
                continue
            if value.get("error"):
                error = value["error"]
                if isinstance(error, dict):
                    raise RuntimeError(str(error.get("message", "Codex rejected the request.")))
                raise RuntimeError(str(error))
            result = value.get("result")
            if not isinstance(result, dict):
                raise RuntimeError("Codex returned no result.")
            return result
        raise RuntimeError("Codex app-server closed unexpectedly.")

    def _start_or_resume(
        self,
        writer: TextIO,
        reader: TextIO,
        conversation_id: str | None,
    ) -> str:
        if conversation_id:
            self._write(
                writer,
                {
                    "id": 2,
                    "method": "thread/resume",
                    "params": {"threadId": conversation_id, "excludeTurns": True},
                },
            )
            try:
                result = self._read_response(reader, 2)
                resumed = result.get("thread")
                if isinstance(resumed, dict) and resumed.get("id"):
                    return str(resumed["id"])
            except RuntimeError:
                # A missing local Codex thread should not strand the locally stored chat.
                pass

        self._write(
            writer,
            {
                "id": 3,
                "method": "thread/start",
                "params": {
                    "cwd": str(self.workspace),
                    "approvalPolicy": "never",
                    "sandbox": "read-only",
                    "developerInstructions": (
                        "You are the conversational research assistant inside Arbrellis, a personal "
                        "knowledge manager. The application may give you a virtual working document and "
                        "a document-command protocol. Treat that protocol as your primary workspace: put "
                        "durable research knowledge in the document and use chat to coordinate and briefly "
                        "report changes. Do not inspect or modify local files, run commands, or use tools; "
                        "the application itself validates and applies your document commands."
                    ),
                },
            },
        )
        result = self._read_response(reader, 3)
        thread = result.get("thread")
        if not isinstance(thread, dict) or not thread.get("id"):
            raise RuntimeError("Codex did not return a conversation ID.")
        return str(thread["id"])

    def run_turn(
        self,
        conversation_id: str | None,
        prompt: str,
        on_delta: DeltaCallback,
    ) -> AgentTurnResult:
        executable = self.executable()
        process = subprocess.Popen(
            [executable, "app-server", "--listen", "stdio://"],
            cwd=self.workspace,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        if process.stdin is None or process.stdout is None:
            process.kill()
            raise RuntimeError("Could not open Codex app-server pipes.")
        writer, reader = process.stdin, process.stdout
        try:
            self._write(
                writer,
                {
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "clientInfo": {
                            "name": "arbrellis",
                            "title": "Arbrellis",
                            "version": "0.1.0",
                        }
                    },
                },
            )
            self._read_response(reader, 1)
            self._write(writer, {"method": "initialized", "params": {}})
            thread_id = self._start_or_resume(writer, reader, conversation_id)
            self._write(
                writer,
                {
                    "id": 4,
                    "method": "turn/start",
                    "params": {
                        "threadId": thread_id,
                        "input": [{"type": "text", "text": prompt}],
                        "approvalPolicy": "never",
                        "sandboxPolicy": {"type": "readOnly", "networkAccess": False},
                    },
                },
            )
            self._read_response(reader, 4)

            answer: list[str] = []
            while line := reader.readline():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                method = value.get("method")
                if method == "item/agentMessage/delta":
                    params = value.get("params")
                    delta = params.get("delta") if isinstance(params, dict) else None
                    if isinstance(delta, str) and delta:
                        answer.append(delta)
                        on_delta(delta)
                elif method == "turn/completed":
                    params = value.get("params")
                    turn = params.get("turn") if isinstance(params, dict) else None
                    status = turn.get("status") if isinstance(turn, dict) else None
                    if status != "completed":
                        detail = turn.get("error") if isinstance(turn, dict) else None
                        raise RuntimeError(str(detail or "Codex turn did not complete."))
                    text = "".join(answer).strip()
                    if not text:
                        raise RuntimeError("Codex completed without returning an answer.")
                    return AgentTurnResult(thread_id, text)
            raise RuntimeError("Codex app-server closed before completing the answer.")
        finally:
            try:
                writer.close()
            except OSError:
                pass
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


class AgentPluginRegistry:
    def __init__(self, plugins: list[AgentPlugin]) -> None:
        self._plugins = {plugin.id: plugin for plugin in plugins}

    def get(self, plugin_id: str) -> AgentPlugin:
        try:
            plugin = self._plugins[plugin_id]
        except KeyError as error:
            raise ValueError(f"Unknown agent plugin: {plugin_id}") from error
        metadata = plugin.metadata()
        if not metadata.get("available", True):
            raise ValueError(str(metadata.get("reason") or f"{plugin.name} is unavailable."))
        return plugin

    def list(self) -> list[dict[str, object]]:
        return [plugin.metadata() for plugin in self._plugins.values()]
