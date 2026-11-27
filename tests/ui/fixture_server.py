#!/usr/bin/env python3
"""Run the real server against a deterministic, disposable UI-test database."""

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "server"))

from document_capture_server import (  # noqa: E402
    BGE_MODEL,
    CaptureApplication,
    ThreadingHTTPServer,
    initialize_database,
    make_handler,
)


def embedding(values: list[float]) -> bytes:
    vector = np.asarray(values, dtype=np.float32)
    vector /= np.linalg.norm(vector)
    return vector.tobytes()


def seed(db_path: Path) -> None:
    initialize_database(db_path)
    content = (
        "# Transformer deployment notes\n\n"
        "Causal attention prevents a token from reading future token positions.\n\n"
        "Weight quantization stores model parameters with reduced numerical precision.\n\n"
        "Residual connections preserve the original hidden state while adding an update."
    )
    attention = "Causal attention prevents a token from reading future token positions."
    quantization = "Weight quantization stores model parameters with reduced numerical precision."
    attention_start = content.index(attention)
    quantization_start = content.index(quantization)

    with sqlite3.connect(db_path) as connection:
        document_id = connection.execute(
            "INSERT INTO documents (name, content) VALUES (?, ?)",
            ("transformer-notes.md", content),
        ).lastrowid
        facts = [
            (
                "transformer-notes.md", attention, attention, "causal attention",
                embedding([1.0, 0.2, 0.0, 0.0]), document_id,
                attention_start, attention_start + len(attention),
            ),
            (
                "simple fact", "Attention heads can specialize in different relationships.",
                "Attention heads can specialize in different relationships.",
                "causal attention", embedding([0.95, 0.25, 0.0, 0.0]), None, None, None,
            ),
            (
                "transformer-notes.md", quantization, quantization, "weight quantization",
                embedding([0.15, 1.0, 0.0, 0.0]), document_id,
                quantization_start, quantization_start + len(quantization),
            ),
            (
                "simple fact", "Residual streams preserve information across transformer blocks.",
                "Residual streams preserve information across transformer blocks.",
                "residual streams", embedding([0.0, 0.2, 1.0, 0.0]), None, None, None,
            ),
        ]
        ids = []
        for source, highlight, context, label, vector, doc_id, start, end in facts:
            cursor = connection.execute(
                """
                INSERT INTO citations (
                    source, highlight, context, label, embedding_model, embedding,
                    integration_state, analysis_state, document_id, document_start,
                    document_end, source_segments
                ) VALUES (?, ?, ?, ?, ?, ?, 'integrated', 'ready', ?, ?, ?, ?)
                """,
                (
                    source, highlight, context, label, BGE_MODEL, vector, doc_id, start, end,
                    json.dumps([[start, end]]) if start is not None else None,
                ),
            )
            ids.append(int(cursor.lastrowid))
        for citation_id, x, y in zip(ids, (0.25, 0.25, 0.63, 0.78), (0.35, 0.35, 0.48, 0.68)):
            connection.execute(
                "INSERT INTO graph_positions (citation_id, x, y) VALUES (?, ?, ?)",
                (citation_id, x, y),
            )
        connection.execute(
            """
            INSERT INTO document_annotations (
                document_id, document_start, document_end, note
            ) VALUES (?, ?, ?, ?)
            """,
            (
                document_id,
                quantization_start,
                quantization_start + len(quantization),
                "Compare post-training and quantization-aware training.",
            ),
        )


def main() -> None:
    temporary = tempfile.TemporaryDirectory(prefix="arbrellis-ui-")
    db_path = Path(temporary.name) / "fixture.sqlite3"
    seed(db_path)
    application = CaptureApplication(db_path)
    server = ThreadingHTTPServer(("127.0.0.1", 18766), make_handler(application))
    print("UI fixture server: http://127.0.0.1:18766", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        temporary.cleanup()


if __name__ == "__main__":
    main()
