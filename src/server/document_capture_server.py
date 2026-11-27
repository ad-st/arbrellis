#!/usr/bin/env python3
"""Serve the local semantic graph and document-capture experiment."""

from __future__ import annotations

import argparse
import json
import math
import mimetypes
import re
import secrets
import sqlite3
import sys
import threading
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from agent_plugins import AgentPluginRegistry, CodexAgentPlugin
from local_knowledge_models import (
    BGE_MODEL,
    CAPTION_VERSION,
    KnowledgeAnalysis,
    LocalKnowledgeModels,
    ProgressCallback,
)


HERE = Path(__file__).resolve().parent
FROZEN_ASSET_ROOT = Path(getattr(sys, "_MEIPASS", HERE))
HTML_PATH = FROZEN_ASSET_ROOT / "document_capture.html"
GRAPH_HTML_PATH = FROZEN_ASSET_ROOT / "graph_view.html"
KATEX_DIST = (
    FROZEN_ASSET_ROOT / "katex"
    if getattr(sys, "frozen", False)
    else HERE.parents[1] / "node_modules" / "katex" / "dist"
)
DEFAULT_DB = Path.home() / ".local" / "share" / "arbrellis" / "knowledge-captures.sqlite3"
LEGACY_BGE_MODEL = "BAAI/bge-base-en-v1.5"
MAX_TEXT_LENGTH = 50_000
MAX_DOCUMENT_LENGTH = 2_000_000
FACT_EDGE_CACHE_VERSION = 1
PERFORMANCE_SAMPLES: list[dict[str, object]] = []
PERFORMANCE_SAMPLES_LOCK = threading.Lock()

LEGACY_SAMPLE_TEXT = """Attention allows a language model to decide which earlier tokens matter most for the current token. Multi-head attention performs several such comparisons in parallel, allowing different heads to represent different relationships.

Model quantization stores weights with lower numerical precision, such as eight-bit or four-bit integers. This reduces memory requirements and may improve local inference speed.

Attention layers can also be quantized. This connects architectural decisions about information flow with deployment decisions about memory, speed, and numerical precision.

Database transaction isolation is a separate concern. Serializable isolation prevents concurrent transactions from observing inconsistent intermediate states."""


SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    content TEXT NOT NULL,
    recovered INTEGER NOT NULL DEFAULT 0,
    is_workspace INTEGER NOT NULL DEFAULT 0,
    revision INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS app_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS document_annotations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL,
    document_start INTEGER NOT NULL,
    document_end INTEGER NOT NULL,
    note TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
    CHECK (document_start >= 0 AND document_end > document_start)
);

CREATE INDEX IF NOT EXISTS document_annotations_lookup
ON document_annotations (document_id, document_start, document_end, id);

CREATE TABLE IF NOT EXISTS chat_threads (
    id TEXT PRIMARY KEY,
    agent_plugin_id TEXT NOT NULL,
    provider_conversation_id TEXT,
    title TEXT NOT NULL,
    workspace_document_id INTEGER,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (workspace_document_id) REFERENCES documents(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id TEXT PRIMARY KEY,
    chat_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (chat_id) REFERENCES chat_threads(id) ON DELETE CASCADE,
    UNIQUE (chat_id, ordinal)
);

CREATE INDEX IF NOT EXISTS chat_messages_thread_lookup
ON chat_messages (chat_id, ordinal);

CREATE TABLE IF NOT EXISTS chat_message_references (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_message_id TEXT NOT NULL,
    document_id INTEGER NOT NULL,
    document_start INTEGER NOT NULL,
    document_end INTEGER NOT NULL,
    quote TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    FOREIGN KEY (chat_message_id) REFERENCES chat_messages(id) ON DELETE CASCADE,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
    CHECK (document_start >= 0 AND document_end > document_start),
    UNIQUE (chat_message_id, ordinal)
);

CREATE TABLE IF NOT EXISTS document_edit_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL,
    from_revision INTEGER NOT NULL,
    to_revision INTEGER NOT NULL,
    commands_json TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS chat_annotations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_message_id TEXT NOT NULL,
    message_start INTEGER NOT NULL,
    message_end INTEGER NOT NULL,
    note TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (chat_message_id) REFERENCES chat_messages(id) ON DELETE CASCADE,
    CHECK (message_start >= 0 AND message_end > message_start)
);

CREATE INDEX IF NOT EXISTS chat_annotations_lookup
ON chat_annotations (chat_message_id, message_start, message_end, id);

CREATE TABLE IF NOT EXISTS citations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    highlight TEXT NOT NULL,
    context TEXT NOT NULL,
    label TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding BLOB NOT NULL,
    fact_caption TEXT,
    caption_version INTEGER,
    draft_highlight TEXT,
    draft_updated_at TEXT,
    integration_state TEXT NOT NULL DEFAULT 'integrated',
    analysis_state TEXT NOT NULL DEFAULT 'ready',
    label_locked INTEGER NOT NULL DEFAULT 0,
    source_segments TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS graph_positions (
    citation_id INTEGER PRIMARY KEY,
    x REAL NOT NULL,
    y REAL NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (citation_id) REFERENCES citations(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS fact_positions (
    citation_id INTEGER PRIMARY KEY,
    x REAL NOT NULL,
    y REAL NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (citation_id) REFERENCES citations(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS graph_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    citation_id INTEGER NOT NULL,
    action TEXT NOT NULL,
    before_state TEXT NOT NULL,
    after_state TEXT NOT NULL,
    applied INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (citation_id) REFERENCES citations(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS concept_similarities (
    embedding_model TEXT NOT NULL,
    label_a TEXT NOT NULL,
    label_b TEXT NOT NULL,
    similarity REAL NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (embedding_model, label_a, label_b)
);

CREATE TABLE IF NOT EXISTS concept_similarity_state (
    embedding_model TEXT PRIMARY KEY,
    initialized INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS concept_similarity_dirty (
    embedding_model TEXT NOT NULL,
    label TEXT NOT NULL,
    PRIMARY KEY (embedding_model, label)
);

CREATE TABLE IF NOT EXISTS fact_similarity_edges (
    embedding_model TEXT NOT NULL,
    label TEXT NOT NULL,
    source_id INTEGER NOT NULL,
    target_id INTEGER NOT NULL,
    similarity REAL NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (embedding_model, label, source_id, target_id)
);

CREATE TABLE IF NOT EXISTS fact_similarity_state (
    embedding_model TEXT NOT NULL,
    label TEXT NOT NULL,
    neighbours INTEGER NOT NULL,
    version INTEGER NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (embedding_model, label)
);

CREATE TABLE IF NOT EXISTS fact_similarity_dirty (
    embedding_model TEXT NOT NULL,
    label TEXT NOT NULL,
    PRIMARY KEY (embedding_model, label)
);

CREATE INDEX IF NOT EXISTS citations_graph_lookup
ON citations (embedding_model, integration_state, analysis_state, label, id);

CREATE TRIGGER IF NOT EXISTS citations_similarity_dirty_insert
AFTER INSERT ON citations
WHEN NEW.integration_state = 'integrated' AND NEW.analysis_state = 'ready'
     AND NEW.embedding_model != '' AND NEW.label != ''
BEGIN
    INSERT OR IGNORE INTO concept_similarity_dirty (embedding_model, label)
    VALUES (NEW.embedding_model, NEW.label);
END;

CREATE TRIGGER IF NOT EXISTS citations_similarity_dirty_delete
AFTER DELETE ON citations
WHEN OLD.integration_state = 'integrated' AND OLD.analysis_state = 'ready'
     AND OLD.embedding_model != '' AND OLD.label != ''
BEGIN
    INSERT OR IGNORE INTO concept_similarity_dirty (embedding_model, label)
    VALUES (OLD.embedding_model, OLD.label);
END;

CREATE TRIGGER IF NOT EXISTS citations_similarity_dirty_update
AFTER UPDATE OF label, embedding_model, embedding, integration_state, analysis_state ON citations
BEGIN
    INSERT OR IGNORE INTO concept_similarity_dirty (embedding_model, label)
    SELECT OLD.embedding_model, OLD.label
    WHERE OLD.integration_state = 'integrated' AND OLD.analysis_state = 'ready'
      AND OLD.embedding_model != '' AND OLD.label != '';
    INSERT OR IGNORE INTO concept_similarity_dirty (embedding_model, label)
    SELECT NEW.embedding_model, NEW.label
    WHERE NEW.integration_state = 'integrated' AND NEW.analysis_state = 'ready'
      AND NEW.embedding_model != '' AND NEW.label != '';
END;

CREATE TRIGGER IF NOT EXISTS citations_fact_similarity_dirty_insert
AFTER INSERT ON citations
WHEN NEW.integration_state = 'integrated' AND NEW.analysis_state = 'ready'
     AND NEW.embedding_model != '' AND NEW.label != ''
BEGIN
    INSERT OR IGNORE INTO fact_similarity_dirty (embedding_model, label)
    VALUES (NEW.embedding_model, NEW.label);
END;

CREATE TRIGGER IF NOT EXISTS citations_fact_similarity_dirty_delete
AFTER DELETE ON citations
WHEN OLD.integration_state = 'integrated' AND OLD.analysis_state = 'ready'
     AND OLD.embedding_model != '' AND OLD.label != ''
BEGIN
    INSERT OR IGNORE INTO fact_similarity_dirty (embedding_model, label)
    VALUES (OLD.embedding_model, OLD.label);
END;

CREATE TRIGGER IF NOT EXISTS citations_fact_similarity_dirty_update
AFTER UPDATE OF label, embedding_model, embedding, integration_state, analysis_state ON citations
BEGIN
    INSERT OR IGNORE INTO fact_similarity_dirty (embedding_model, label)
    SELECT OLD.embedding_model, OLD.label
    WHERE OLD.integration_state = 'integrated' AND OLD.analysis_state = 'ready'
      AND OLD.embedding_model != '' AND OLD.label != '';
    INSERT OR IGNORE INTO fact_similarity_dirty (embedding_model, label)
    SELECT NEW.embedding_model, NEW.label
    WHERE NEW.integration_state = 'integrated' AND NEW.analysis_state = 'ready'
      AND NEW.embedding_model != '' AND NEW.label != '';
END;
"""


def connect(db_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path, timeout=5.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    connection.execute("PRAGMA synchronous = NORMAL")
    return connection


def initialize_database(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with connect(db_path) as connection:
        # WAL lets graph/fact reads continue while a background integration writes.
        # journal_mode persists in the database; the remaining connection settings
        # are deliberately applied by connect() to every worker connection.
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA wal_autocheckpoint = 1000")
        connection.executescript(SCHEMA)
        document_columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(documents)")
        }
        if "revision" not in document_columns:
            connection.execute(
                "ALTER TABLE documents ADD COLUMN revision INTEGER NOT NULL DEFAULT 0"
            )
        if "updated_at" not in document_columns:
            connection.execute("ALTER TABLE documents ADD COLUMN updated_at TEXT")
        connection.execute("UPDATE documents SET updated_at = CURRENT_TIMESTAMP WHERE updated_at IS NULL")
        if "is_workspace" not in document_columns:
            connection.execute(
                "ALTER TABLE documents ADD COLUMN is_workspace INTEGER NOT NULL DEFAULT 0"
            )
        chat_columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(chat_threads)")
        }
        if "workspace_document_id" not in chat_columns:
            connection.execute("ALTER TABLE chat_threads ADD COLUMN workspace_document_id INTEGER")
        connection.execute(
            """UPDATE documents SET is_workspace = 1
               WHERE id IN (SELECT workspace_document_id FROM chat_threads
                            WHERE workspace_document_id IS NOT NULL)"""
        )
        columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(citations)")
        }
        if "fact_caption" not in columns:
            connection.execute("ALTER TABLE citations ADD COLUMN fact_caption TEXT")
        if "caption_version" not in columns:
            connection.execute("ALTER TABLE citations ADD COLUMN caption_version INTEGER")
        if "draft_highlight" not in columns:
            connection.execute("ALTER TABLE citations ADD COLUMN draft_highlight TEXT")
        if "draft_updated_at" not in columns:
            connection.execute("ALTER TABLE citations ADD COLUMN draft_updated_at TEXT")
        if "integration_state" not in columns:
            connection.execute(
                "ALTER TABLE citations ADD COLUMN integration_state TEXT NOT NULL DEFAULT 'integrated'"
            )
        if "analysis_state" not in columns:
            connection.execute(
                "ALTER TABLE citations ADD COLUMN analysis_state TEXT NOT NULL DEFAULT 'ready'"
            )
        if "label_locked" not in columns:
            connection.execute(
                "ALTER TABLE citations ADD COLUMN label_locked INTEGER NOT NULL DEFAULT 0"
            )
        if "document_id" not in columns:
            connection.execute("ALTER TABLE citations ADD COLUMN document_id INTEGER")
        if "document_start" not in columns:
            connection.execute("ALTER TABLE citations ADD COLUMN document_start INTEGER")
        if "document_end" not in columns:
            connection.execute("ALTER TABLE citations ADD COLUMN document_end INTEGER")
        if "chat_message_id" not in columns:
            connection.execute("ALTER TABLE citations ADD COLUMN chat_message_id TEXT")
        if "chat_start" not in columns:
            connection.execute("ALTER TABLE citations ADD COLUMN chat_start INTEGER")
        if "chat_end" not in columns:
            connection.execute("ALTER TABLE citations ADD COLUMN chat_end INTEGER")
        if "source_segments" not in columns:
            connection.execute("ALTER TABLE citations ADD COLUMN source_segments TEXT")
        connection.execute(
            "CREATE INDEX IF NOT EXISTS citations_document_lookup ON citations (document_id, id)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS citations_chat_lookup ON citations (chat_message_id, id)"
        )
        _migrate_legacy_documents(connection)
        _repair_recovered_documents(connection)


def _migrate_legacy_documents(connection: sqlite3.Connection) -> None:
    """Recover the document material retained by the original selection-only UI."""
    sources = connection.execute(
        """
        SELECT DISTINCT source FROM citations
        WHERE document_id IS NULL AND source != 'simple fact'
        ORDER BY source
        """
    ).fetchall()
    for source_row in sources:
        source = str(source_row["source"])
        rows = connection.execute(
            """
            SELECT id, highlight, context FROM citations
            WHERE source = ? AND document_id IS NULL ORDER BY id
            """,
            (source,),
        ).fetchall()
        if not rows:
            continue
        recovered = source != "Sample document"
        if recovered:
            contexts: list[str] = []
            for row in rows:
                context = str(row["context"] or row["highlight"]).strip()
                if context and context not in contexts:
                    contexts.append(context)
            content = "\n\n--- recovered excerpt ---\n\n".join(contexts)
        else:
            content = LEGACY_SAMPLE_TEXT
        cursor = connection.execute(
            "INSERT INTO documents (name, content, recovered) VALUES (?, ?, ?)",
            (source, content, int(recovered)),
        )
        document_id = int(cursor.lastrowid)
        search_from = 0
        for row in rows:
            highlight = str(row["highlight"])
            start = content.find(highlight, search_from)
            if start < 0:
                start = content.find(highlight)
            end = start + len(highlight) if start >= 0 else None
            connection.execute(
                """
                UPDATE citations
                SET document_id = ?, document_start = ?, document_end = ?
                WHERE id = ?
                """,
                (document_id, start if start >= 0 else None, end, int(row["id"])),
            )
            if end is not None:
                search_from = end


def _repair_recovered_documents(connection: sqlite3.Connection) -> None:
    """Keep legacy excerpt documents searchable even when Markdown was stripped."""
    documents = connection.execute(
        "SELECT id FROM documents WHERE recovered = 1 ORDER BY id"
    ).fetchall()
    separator = "\n\n--- recovered excerpt ---\n\n"
    for document in documents:
        document_id = int(document["id"])
        facts = connection.execute(
            "SELECT id, highlight FROM citations WHERE document_id = ? ORDER BY id",
            (document_id,),
        ).fetchall()
        if not facts:
            continue
        parts = [str(fact["highlight"]) for fact in facts]
        content = separator.join(parts)
        connection.execute(
            "UPDATE documents SET content = ? WHERE id = ?", (content, document_id)
        )
        cursor = 0
        for index, fact in enumerate(facts):
            highlight = parts[index]
            start = cursor
            end = start + len(highlight)
            connection.execute(
                """
                UPDATE citations SET document_start = ?, document_end = ?
                WHERE id = ?
                """,
                (start, end, int(fact["id"])),
            )
            cursor = end + (len(separator) if index + 1 < len(facts) else 0)


class CaptureStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def get_setting(self, key: str) -> str | None:
        with connect(self.db_path) as connection:
            row = connection.execute(
                "SELECT value FROM app_settings WHERE key = ?", (key,)
            ).fetchone()
        return str(row["value"]) if row is not None else None

    def set_setting(self, key: str, value: str) -> None:
        with connect(self.db_path) as connection:
            connection.execute(
                """INSERT INTO app_settings (key, value) VALUES (?, ?)
                   ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
                (key, value),
            )

    @staticmethod
    def _fact_with_segments(row: sqlite3.Row) -> dict:
        fact = dict(row)
        raw = fact.get("source_segments")
        try:
            fact["source_segments"] = json.loads(raw) if raw else None
        except (TypeError, json.JSONDecodeError):
            fact["source_segments"] = None
        return fact

    def list(self, limit: int = 10_000) -> list[dict]:
        with connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT id, source, highlight, context, label, embedding_model,
                       fact_caption, caption_version, draft_highlight,
                       draft_updated_at, integration_state, analysis_state,
                       label_locked, document_id, document_start, document_end,
                       chat_message_id, chat_start, chat_end,
                       (SELECT chat_id FROM chat_messages
                        WHERE chat_messages.id = citations.chat_message_id) AS chat_id,
                       created_at
                FROM citations ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_summaries(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        query: str = "",
        citation_ids: list[int] | None = None,
        origin: str = "all",
    ) -> tuple[list[dict], int]:
        pattern = f"%{query}%"
        clauses: list[str] = []
        parameters: list[object] = []
        if origin == "standalone":
            clauses.append("document_id IS NULL AND chat_message_id IS NULL")
        elif origin == "document":
            clauses.append("document_id IS NOT NULL")
        elif origin == "chat":
            clauses.append("chat_message_id IS NOT NULL")
        elif origin != "all":
            raise ValueError("Invalid fact origin.")
        if query:
            clauses.append(
                "(label LIKE ? OR source LIKE ? OR highlight LIKE ? "
                "OR COALESCE(draft_highlight, '') LIKE ?)"
            )
            parameters.extend([pattern, pattern, pattern, pattern])
        if citation_ids is not None:
            if citation_ids:
                placeholders = ",".join("?" for _ in citation_ids)
                clauses.append(f"id IN ({placeholders})")
                parameters.extend(citation_ids)
            else:
                clauses.append("0")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with connect(self.db_path) as connection:
            total = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM citations {where}", parameters
                ).fetchone()[0]
            )
            rows = connection.execute(
                f"""
                SELECT id, source, label,
                       SUBSTR(COALESCE(draft_highlight, highlight), 1, 240) AS preview,
                       CASE WHEN draft_highlight IS NULL THEN 0 ELSE 1 END AS dirty,
                       integration_state, analysis_state, label_locked,
                       document_id, document_start, document_end,
                       chat_message_id, chat_start, chat_end, created_at
                FROM citations {where}
                ORDER BY id DESC LIMIT ? OFFSET ?
                """,
                (*parameters, limit, offset),
            ).fetchall()
        return [dict(row) for row in rows], total

    def get(self, citation_id: int) -> dict:
        with connect(self.db_path) as connection:
            row = connection.execute(
                """
                SELECT id, source, highlight, context, label, embedding_model,
                       fact_caption, caption_version, draft_highlight,
                       draft_updated_at, integration_state, analysis_state,
                       label_locked, document_id, document_start, document_end,
                       chat_message_id, chat_start, chat_end,
                       (SELECT chat_id FROM chat_messages
                        WHERE chat_messages.id = citations.chat_message_id) AS chat_id,
                       created_at
                FROM citations WHERE id = ?
                """,
                (citation_id,),
            ).fetchone()
        if row is None:
            raise KeyError(citation_id)
        return dict(row)

    def count(self, origin: str = "all") -> int:
        if origin == "standalone":
            where = " WHERE document_id IS NULL AND chat_message_id IS NULL"
        elif origin == "document":
            where = " WHERE document_id IS NOT NULL"
        elif origin == "chat":
            where = " WHERE chat_message_id IS NOT NULL"
        elif origin == "all":
            where = ""
        else:
            raise ValueError("Invalid fact origin.")
        with connect(self.db_path) as connection:
            return int(connection.execute(f"SELECT COUNT(*) FROM citations{where}").fetchone()[0])

    def list_documents(self, query: str = "", concept: str = "") -> list[dict]:
        clauses: list[str] = ["documents.is_workspace = 0"]
        parameters: list[object] = []
        if query:
            pattern = f"%{query}%"
            clauses.append("(documents.name LIKE ? OR documents.content LIKE ?)")
            parameters.extend([pattern, pattern])
        if concept:
            clauses.append(
                "EXISTS (SELECT 1 FROM citations AS concept_facts "
                "WHERE concept_facts.document_id = documents.id "
                "AND concept_facts.label = ?)"
            )
            parameters.append(concept)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with connect(self.db_path) as connection:
            rows = connection.execute(
                f"""
                SELECT documents.id, documents.name, documents.recovered,
                       documents.created_at, LENGTH(documents.content) AS character_count,
                       COUNT(citations.id) AS fact_count
                FROM documents
                LEFT JOIN citations ON citations.document_id = documents.id
                {where}
                GROUP BY documents.id
                ORDER BY documents.created_at DESC, documents.id DESC
                """,
                parameters,
            ).fetchall()
        return [dict(row) for row in rows]

    def count_documents(self) -> int:
        with connect(self.db_path) as connection:
            return int(connection.execute("SELECT COUNT(*) FROM documents WHERE is_workspace = 0").fetchone()[0])

    def document_concepts(self) -> list[str]:
        with connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT label FROM citations
                WHERE document_id IS NOT NULL AND label != ''
                  AND document_id IN (SELECT id FROM documents WHERE is_workspace = 0)
                ORDER BY label COLLATE NOCASE
                """
            ).fetchall()
        return [str(row["label"]) for row in rows]

    def get_document(self, document_id: int) -> dict:
        with connect(self.db_path) as connection:
            row = connection.execute(
                """
                SELECT id, name, content, recovered, revision, created_at, updated_at
                FROM documents WHERE id = ?
                """,
                (document_id,),
            ).fetchone()
            if row is None:
                raise KeyError(document_id)
            facts = connection.execute(
                """
                SELECT id, label, highlight, document_start, document_end,
                       source_segments,
                       integration_state, analysis_state
                FROM citations WHERE document_id = ? ORDER BY document_start, id
                """,
                (document_id,),
            ).fetchall()
            annotations = connection.execute(
                """
                SELECT id, document_id, document_start, document_end, note,
                       created_at, updated_at
                FROM document_annotations
                WHERE document_id = ?
                ORDER BY document_start, document_end, id
                """,
                (document_id,),
            ).fetchall()
        document = dict(row)
        document["facts"] = [self._fact_with_segments(fact) for fact in facts]
        document["annotations"] = [dict(annotation) for annotation in annotations]
        return document

    def add_document_annotation(
        self,
        document_id: int,
        document_start: int,
        document_end: int,
        note: str,
    ) -> dict:
        with connect(self.db_path) as connection:
            document = connection.execute(
                "SELECT content FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
            if document is None:
                raise KeyError(document_id)
            content = str(document["content"])
            if document_start < 0 or document_end <= document_start or document_end > len(content):
                raise ValueError("Invalid annotation range.")
            overlap = connection.execute(
                """
                SELECT 1 FROM document_annotations
                WHERE document_id = ? AND document_start < ? AND document_end > ?
                LIMIT 1
                """,
                (document_id, document_end, document_start),
            ).fetchone()
            if overlap is not None:
                raise ValueError("Annotations cannot overlap another annotation.")
            cursor = connection.execute(
                """
                INSERT INTO document_annotations
                    (document_id, document_start, document_end, note)
                VALUES (?, ?, ?, ?)
                """,
                (document_id, document_start, document_end, note),
            )
            annotation_id = int(cursor.lastrowid)
            row = connection.execute(
                "SELECT * FROM document_annotations WHERE id = ?", (annotation_id,)
            ).fetchone()
        return dict(row)

    def update_document_annotation(self, annotation_id: int, note: str) -> dict:
        with connect(self.db_path) as connection:
            cursor = connection.execute(
                """
                UPDATE document_annotations
                SET note = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (note, annotation_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(annotation_id)
            row = connection.execute(
                "SELECT * FROM document_annotations WHERE id = ?", (annotation_id,)
            ).fetchone()
        return dict(row)

    def list_chats(self) -> list[dict]:
        with connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT chat_threads.id, chat_threads.agent_plugin_id,
                       chat_threads.title, chat_threads.created_at,
                       chat_threads.updated_at, COUNT(chat_messages.id) AS message_count
                FROM chat_threads
                LEFT JOIN chat_messages ON chat_messages.chat_id = chat_threads.id
                GROUP BY chat_threads.id
                ORDER BY chat_threads.updated_at DESC, chat_threads.created_at DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def count_chats(self) -> int:
        with connect(self.db_path) as connection:
            return int(connection.execute("SELECT COUNT(*) FROM chat_threads").fetchone()[0])

    def create_chat(self, agent_plugin_id: str) -> dict:
        chat_id = uuid.uuid4().hex
        with connect(self.db_path) as connection:
            document_cursor = connection.execute(
                "INSERT INTO documents (name, content, is_workspace) VALUES (?, '', 1)",
                ("Untitled knowledge.md",),
            )
            document_id = int(document_cursor.lastrowid)
            connection.execute(
                """
                INSERT INTO chat_threads (id, agent_plugin_id, title, workspace_document_id)
                VALUES (?, ?, 'New chat', ?)
                """,
                (chat_id, agent_plugin_id, document_id),
            )
        return self.get_chat(chat_id)

    def get_chat(self, chat_id: str) -> dict:
        with connect(self.db_path) as connection:
            thread = connection.execute(
                """
                SELECT id, agent_plugin_id, provider_conversation_id, title,
                       workspace_document_id, created_at, updated_at
                FROM chat_threads WHERE id = ?
                """,
                (chat_id,),
            ).fetchone()
            if thread is None:
                raise KeyError(chat_id)
            message_rows = connection.execute(
                """
                SELECT id, chat_id, role, content, ordinal, created_at
                FROM chat_messages WHERE chat_id = ? ORDER BY ordinal
                """,
                (chat_id,),
            ).fetchall()
            messages: list[dict] = []
            all_facts: list[dict] = []
            for message_row in message_rows:
                message = dict(message_row)
                references = connection.execute(
                    """
                    SELECT id, document_id, document_start, document_end, quote, ordinal
                    FROM chat_message_references WHERE chat_message_id = ? ORDER BY ordinal
                    """,
                    (message["id"],),
                ).fetchall()
                facts = connection.execute(
                    """
                    SELECT id, label, highlight, chat_message_id, chat_start, chat_end,
                           source_segments,
                           integration_state, analysis_state
                    FROM citations WHERE chat_message_id = ? ORDER BY chat_start, id
                    """,
                    (message["id"],),
                ).fetchall()
                annotations = connection.execute(
                    """
                    SELECT id, chat_message_id, message_start, message_end, note,
                           created_at, updated_at
                    FROM chat_annotations WHERE chat_message_id = ?
                    ORDER BY message_start, message_end, id
                    """,
                    (message["id"],),
                ).fetchall()
                message["facts"] = [self._fact_with_segments(fact) for fact in facts]
                message["annotations"] = [dict(annotation) for annotation in annotations]
                message["references"] = [dict(reference) for reference in references]
                all_facts.extend(message["facts"])
                messages.append(message)
        chat = dict(thread)
        chat["messages"] = messages
        chat["facts"] = all_facts
        document_id = chat.get("workspace_document_id")
        chat["document"] = self.get_document(int(document_id)) if document_id else None
        return chat

    def chat_provider_state(self, chat_id: str) -> tuple[str, str | None]:
        with connect(self.db_path) as connection:
            row = connection.execute(
                """
                SELECT agent_plugin_id, provider_conversation_id
                FROM chat_threads WHERE id = ?
                """,
                (chat_id,),
            ).fetchone()
        if row is None:
            raise KeyError(chat_id)
        return str(row["agent_plugin_id"]), row["provider_conversation_id"]

    def add_chat_message(
        self, chat_id: str, role: str, content: str, references: list[dict] | None = None
    ) -> dict:
        message_id = uuid.uuid4().hex
        with connect(self.db_path) as connection:
            exists = connection.execute(
                "SELECT 1 FROM chat_threads WHERE id = ?", (chat_id,)
            ).fetchone()
            if exists is None:
                raise KeyError(chat_id)
            ordinal = int(
                connection.execute(
                    "SELECT COALESCE(MAX(ordinal), -1) + 1 FROM chat_messages WHERE chat_id = ?",
                    (chat_id,),
                ).fetchone()[0]
            )
            connection.execute(
                """
                INSERT INTO chat_messages (id, chat_id, role, content, ordinal)
                VALUES (?, ?, ?, ?, ?)
                """,
                (message_id, chat_id, role, content, ordinal),
            )
            for index, reference in enumerate(references or []):
                document_id = int(reference["document_id"])
                start = int(reference["document_start"])
                end = int(reference["document_end"])
                document = connection.execute(
                    "SELECT content FROM documents WHERE id = ?", (document_id,)
                ).fetchone()
                if document is None or start < 0 or end <= start or end > len(document["content"]):
                    raise ValueError("A document reference is no longer valid.")
                quote = str(document["content"])[start:end]
                connection.execute(
                    """
                    INSERT INTO chat_message_references
                        (chat_message_id, document_id, document_start, document_end, quote, ordinal)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (message_id, document_id, start, end, quote, index),
                )
            row = connection.execute(
                "SELECT * FROM chat_messages WHERE id = ?", (message_id,)
            ).fetchone()
        return dict(row)

    def chat_workspace(self, chat_id: str) -> dict:
        with connect(self.db_path) as connection:
            row = connection.execute(
                """
                SELECT documents.id, documents.name, documents.content, documents.revision
                FROM chat_threads JOIN documents
                  ON documents.id = chat_threads.workspace_document_id
                WHERE chat_threads.id = ?
                """,
                (chat_id,),
            ).fetchone()
            if row is None:
                raise KeyError(chat_id)
            protected = connection.execute(
                """
                SELECT document_start AS start, document_end AS end, 'fact' AS kind, id
                FROM citations WHERE document_id = ?
                UNION ALL
                SELECT document_start, document_end, 'annotation', id
                FROM document_annotations WHERE document_id = ?
                ORDER BY start, end
                """,
                (row["id"], row["id"]),
            ).fetchall()
        result = dict(row)
        result["protected_ranges"] = [dict(item) for item in protected]
        return result

    def apply_document_commands(
        self, document_id: int, expected_revision: int, commands: list[dict]
    ) -> dict:
        if not commands:
            return {"applied": 0, "revision": expected_revision}
        with connect(self.db_path) as connection:
            row = connection.execute(
                "SELECT content, revision FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
            if row is None:
                raise KeyError(document_id)
            revision = int(row["revision"] or 0)
            if revision != expected_revision:
                raise ValueError(f"Document changed: expected revision {expected_revision}, found {revision}.")
            content = str(row["content"])
            protected = [
                (int(item["start"]), int(item["end"]), str(item["kind"]))
                for item in connection.execute(
                    """
                    SELECT document_start AS start, document_end AS end, 'fact' AS kind
                    FROM citations WHERE document_id = ?
                    UNION ALL
                    SELECT document_start, document_end, 'annotation'
                    FROM document_annotations WHERE document_id = ?
                    """,
                    (document_id, document_id),
                )
            ]
            normalized: list[dict] = []
            for command in commands:
                operation = str(command.get("op", ""))
                if operation == "insert":
                    start = end = int(command.get("at", -1))
                elif operation in {"replace", "delete"}:
                    start, end = int(command.get("start", -1)), int(command.get("end", -1))
                else:
                    raise ValueError(f"Unsupported document command: {operation or 'missing op'}.")
                text = "" if operation == "delete" else str(command.get("text", ""))
                if start < 0 or end < start or end > len(content):
                    raise ValueError("Document command range is outside the current document.")
                for protected_start, protected_end, kind in protected:
                    intersects = start < protected_end and end > protected_start
                    insert_inside = start == end and protected_start < start < protected_end
                    if intersects or insert_inside:
                        raise ValueError(
                            f"Document command touches a protected {kind} range "
                            f"({protected_start}:{protected_end})."
                        )
                normalized.append({"op": operation, "start": start, "end": end, "text": text})
            normalized.sort(key=lambda command: (command["start"], command["end"]), reverse=True)
            for previous, current in zip(normalized, normalized[1:]):
                if current["end"] > previous["start"]:
                    raise ValueError("Document commands overlap each other.")
            for command in normalized:
                start, end, text = command["start"], command["end"], command["text"]
                delta = len(text) - (end - start)
                content = content[:start] + text + content[end:]
                connection.execute(
                    """UPDATE citations SET document_start = document_start + ?,
                       document_end = document_end + ?
                       WHERE document_id = ? AND document_start >= ?""",
                    (delta, delta, document_id, end),
                )
                connection.execute(
                    """UPDATE document_annotations SET document_start = document_start + ?,
                       document_end = document_end + ?
                       WHERE document_id = ? AND document_start >= ?""",
                    (delta, delta, document_id, end),
                )
                segment_rows = connection.execute(
                    "SELECT id, source_segments FROM citations WHERE document_id = ? AND source_segments IS NOT NULL",
                    (document_id,),
                ).fetchall()
                for segment_row in segment_rows:
                    segments = json.loads(segment_row["source_segments"] or "[]")
                    shifted = [[a + delta, b + delta] if a >= end else [a, b] for a, b in segments]
                    connection.execute(
                        "UPDATE citations SET source_segments = ? WHERE id = ?",
                        (json.dumps(shifted), segment_row["id"]),
                    )
            next_revision = revision + 1
            connection.execute(
                """UPDATE documents SET content = ?, revision = ?, updated_at = CURRENT_TIMESTAMP
                   WHERE id = ?""",
                (content, next_revision, document_id),
            )
            connection.execute(
                """INSERT INTO document_edit_events
                   (document_id, from_revision, to_revision, commands_json, status)
                   VALUES (?, ?, ?, ?, 'applied')""",
                (document_id, revision, next_revision, json.dumps(commands)),
            )
        return {"applied": len(normalized), "revision": next_revision}

    def complete_chat_turn(
        self,
        chat_id: str,
        provider_conversation_id: str,
        answer: str,
        first_prompt: str,
    ) -> dict:
        message = self.add_chat_message(chat_id, "assistant", answer)
        title_words = first_prompt.split()[:6]
        title = " ".join(title_words) or "New chat"
        if len(first_prompt.split()) > len(title_words):
            title += "…"
        with connect(self.db_path) as connection:
            current = connection.execute(
                "SELECT title FROM chat_threads WHERE id = ?", (chat_id,)
            ).fetchone()
            next_title = title if current and current["title"] == "New chat" else current["title"]
            connection.execute(
                """
                UPDATE chat_threads
                SET provider_conversation_id = ?, title = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (provider_conversation_id, next_title, chat_id),
            )
        return message

    def add_chat_annotation(
        self, chat_message_id: str, message_start: int, message_end: int, note: str
    ) -> dict:
        with connect(self.db_path) as connection:
            message = connection.execute(
                "SELECT content FROM chat_messages WHERE id = ?", (chat_message_id,)
            ).fetchone()
            if message is None:
                raise KeyError(chat_message_id)
            content = str(message["content"])
            if message_start < 0 or message_end <= message_start or message_end > len(content):
                raise ValueError("Invalid annotation range.")
            overlap = connection.execute(
                """
                SELECT 1 FROM chat_annotations
                WHERE chat_message_id = ? AND message_start < ? AND message_end > ?
                LIMIT 1
                """,
                (chat_message_id, message_end, message_start),
            ).fetchone()
            if overlap is not None:
                raise ValueError("Annotations cannot overlap another annotation.")
            cursor = connection.execute(
                """
                INSERT INTO chat_annotations
                    (chat_message_id, message_start, message_end, note)
                VALUES (?, ?, ?, ?)
                """,
                (chat_message_id, message_start, message_end, note),
            )
            row = connection.execute(
                "SELECT * FROM chat_annotations WHERE id = ?", (cursor.lastrowid,)
            ).fetchone()
        return dict(row)

    def update_chat_annotation(self, annotation_id: int, note: str) -> dict:
        with connect(self.db_path) as connection:
            cursor = connection.execute(
                """
                UPDATE chat_annotations SET note = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (note, annotation_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(annotation_id)
            row = connection.execute(
                "SELECT * FROM chat_annotations WHERE id = ?", (annotation_id,)
            ).fetchone()
        return dict(row)

    def add_document(self, name: str, content: str) -> dict:
        with connect(self.db_path) as connection:
            existing = connection.execute(
                "SELECT id FROM documents WHERE name = ? AND content = ? ORDER BY id LIMIT 1",
                (name, content),
            ).fetchone()
            if existing is not None:
                return self.get_document(int(existing["id"]))
            cursor = connection.execute(
                "INSERT INTO documents (name, content) VALUES (?, ?)",
                (name, content),
            )
            document_id = int(cursor.lastrowid)
        return self.get_document(document_id)

    @staticmethod
    def _record_graph_action(
        connection: sqlite3.Connection,
        citation_id: int,
        action: str,
        before_state: str,
        after_state: str,
    ) -> int:
        # A new action after undo starts a new linear branch.
        connection.execute("DELETE FROM graph_history WHERE applied = 0")
        cursor = connection.execute(
            """
            INSERT INTO graph_history
                (citation_id, action, before_state, after_state, applied)
            VALUES (?, ?, ?, ?, 1)
            """,
            (citation_id, action, before_state, after_state),
        )
        return int(cursor.lastrowid)

    @staticmethod
    def _apply_integration_state(
        connection: sqlite3.Connection, citation_id: int, state: str
    ) -> None:
        if state not in {"integrated", "not_integrated"}:
            raise ValueError(f"Invalid integration state: {state}")
        row = connection.execute(
            """
            SELECT citations.label, graph_positions.x, graph_positions.y
            FROM citations
            LEFT JOIN graph_positions ON graph_positions.citation_id = citations.id
            WHERE citations.id = ?
            """,
            (citation_id,),
        ).fetchone()
        if row is None:
            raise KeyError(citation_id)
        representative = connection.execute(
            """
            SELECT id FROM citations
            WHERE label = ? AND integration_state = 'integrated'
            ORDER BY id LIMIT 1
            """,
            (row["label"],),
        ).fetchone()
        is_representative = (
            representative is not None and int(representative["id"]) == citation_id
        )
        if (
            state == "not_integrated"
            and is_representative
            and row["x"] is not None
            and row["y"] is not None
        ):
            replacement = connection.execute(
                """
                SELECT id FROM citations
                WHERE label = ? AND id != ? AND integration_state = 'integrated'
                ORDER BY id LIMIT 1
                """,
                (row["label"], citation_id),
            ).fetchone()
            if replacement is not None:
                connection.execute(
                    """
                    INSERT INTO graph_positions (citation_id, x, y)
                    VALUES (?, ?, ?)
                    ON CONFLICT(citation_id) DO UPDATE SET
                        x = excluded.x, y = excluded.y, updated_at = CURRENT_TIMESTAMP
                    """,
                    (int(replacement["id"]), float(row["x"]), float(row["y"])),
                )
        connection.execute(
            "UPDATE citations SET integration_state = ? WHERE id = ?",
            (state, citation_id),
        )

    def history_status(self) -> dict:
        with connect(self.db_path) as connection:
            undo = connection.execute(
                """
                SELECT graph_history.id, graph_history.citation_id, graph_history.action,
                       citations.label
                FROM graph_history JOIN citations ON citations.id = graph_history.citation_id
                WHERE applied = 1 ORDER BY graph_history.id DESC LIMIT 1
                """
            ).fetchone()
            redo = connection.execute(
                """
                SELECT graph_history.id, graph_history.citation_id, graph_history.action,
                       citations.label
                FROM graph_history JOIN citations ON citations.id = graph_history.citation_id
                WHERE applied = 0 ORDER BY graph_history.id LIMIT 1
                """
            ).fetchone()
        return {
            "can_undo": undo is not None,
            "can_redo": redo is not None,
            "undo": dict(undo) if undo else None,
            "redo": dict(redo) if redo else None,
        }

    def set_integration(self, citation_id: int, integrated: bool) -> dict:
        state = "integrated" if integrated else "not_integrated"
        with connect(self.db_path) as connection:
            row = connection.execute(
                """
                SELECT integration_state, analysis_state FROM citations WHERE id = ?
                """,
                (citation_id,),
            ).fetchone()
            if row is None:
                raise KeyError(citation_id)
            if integrated and str(row["analysis_state"]) != "ready":
                raise ValueError("This fact must be analyzed before integration.")
            before = str(row["integration_state"])
            changed = before != state
            if changed:
                self._apply_integration_state(connection, citation_id, state)
                self._record_graph_action(
                    connection,
                    citation_id,
                    "integrate" if integrated else "remove",
                    before,
                    state,
                )
        return {"changed": changed, "citation": self.get(citation_id), "history": self.history_status()}

    def move_history(self, direction: str) -> dict:
        if direction not in {"undo", "redo"}:
            raise ValueError("History direction must be undo or redo.")
        with connect(self.db_path) as connection:
            if direction == "undo":
                row = connection.execute(
                    "SELECT * FROM graph_history WHERE applied = 1 ORDER BY id DESC LIMIT 1"
                ).fetchone()
                if row is None:
                    return {"changed": False, "history": self.history_status()}
                self._apply_integration_state(
                    connection, int(row["citation_id"]), str(row["before_state"])
                )
                connection.execute("UPDATE graph_history SET applied = 0 WHERE id = ?", (row["id"],))
            else:
                row = connection.execute(
                    "SELECT * FROM graph_history WHERE applied = 0 ORDER BY id LIMIT 1"
                ).fetchone()
                if row is None:
                    return {"changed": False, "history": self.history_status()}
                self._apply_integration_state(
                    connection, int(row["citation_id"]), str(row["after_state"])
                )
                connection.execute("UPDATE graph_history SET applied = 1 WHERE id = ?", (row["id"],))
            result = {
                "changed": True,
                "direction": direction,
                "citation_id": int(row["citation_id"]),
                "action": str(row["action"]),
            }
        result["history"] = self.history_status()
        return result

    @staticmethod
    def _refresh_concept_similarity_cache(
        connection: sqlite3.Connection,
        embedding_model: str,
        affected_labels: set[str] | None,
    ) -> int:
        rows = connection.execute(
            """
            SELECT label, embedding FROM citations
            WHERE embedding_model = ? AND integration_state = 'integrated'
              AND analysis_state = 'ready'
            ORDER BY label, id
            """,
            (embedding_model,),
        ).fetchall()
        vectors: dict[str, list[np.ndarray]] = {}
        for row in rows:
            vectors.setdefault(str(row["label"]), []).append(
                np.frombuffer(row["embedding"], dtype=np.float32)
            )
        labels = sorted(vectors)

        if affected_labels is None:
            connection.execute(
                "DELETE FROM concept_similarities WHERE embedding_model = ?",
                (embedding_model,),
            )
            pairs = [
                (labels[left], labels[right])
                for left in range(len(labels))
                for right in range(left + 1, len(labels))
            ]
        else:
            dirty = sorted(affected_labels)
            if dirty:
                placeholders = ",".join("?" for _ in dirty)
                connection.execute(
                    f"""
                    DELETE FROM concept_similarities
                    WHERE embedding_model = ?
                      AND (label_a IN ({placeholders}) OR label_b IN ({placeholders}))
                    """,
                    (embedding_model, *dirty, *dirty),
                )
            active_dirty = set(dirty) & set(labels)
            pairs = sorted(
                {
                    tuple(sorted((label, other)))
                    for label in active_dirty
                    for other in labels
                    if label != other
                }
            )

        cached = []
        for label_a, label_b in pairs:
            left = np.stack(vectors[label_a])
            right = np.stack(vectors[label_b])
            similarity = float(np.max(left @ right.T))
            cached.append((embedding_model, label_a, label_b, similarity))
        connection.executemany(
            """
            INSERT INTO concept_similarities
                (embedding_model, label_a, label_b, similarity)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(embedding_model, label_a, label_b) DO UPDATE SET
                similarity = excluded.similarity,
                updated_at = CURRENT_TIMESTAMP
            """,
            cached,
        )
        connection.execute(
            """
            INSERT INTO concept_similarity_state (embedding_model, initialized)
            VALUES (?, 1)
            ON CONFLICT(embedding_model) DO UPDATE SET
                initialized = 1, updated_at = CURRENT_TIMESTAMP
            """,
            (embedding_model,),
        )
        if affected_labels is None:
            connection.execute(
                "DELETE FROM concept_similarity_dirty WHERE embedding_model = ?",
                (embedding_model,),
            )
        elif affected_labels:
            dirty = sorted(affected_labels)
            placeholders = ",".join("?" for _ in dirty)
            connection.execute(
                f"""
                DELETE FROM concept_similarity_dirty
                WHERE embedding_model = ? AND label IN ({placeholders})
                """,
                (embedding_model, *dirty),
            )
        stale_fact_labels = (
            set(
                row["label"]
                for row in connection.execute(
                    """
                    SELECT label FROM fact_similarity_state WHERE embedding_model = ?
                    """,
                    (embedding_model,),
                )
            )
            - set(labels)
        )
        if stale_fact_labels:
            stale = sorted(stale_fact_labels)
            placeholders = ",".join("?" for _ in stale)
            for table in (
                "fact_similarity_edges",
                "fact_similarity_state",
                "fact_similarity_dirty",
            ):
                connection.execute(
                    f"""
                    DELETE FROM {table}
                    WHERE embedding_model = ? AND label IN ({placeholders})
                    """,
                    (embedding_model, *stale),
                )
        return len(pairs)

    def _ensure_concept_similarity_cache(
        self, connection: sqlite3.Connection, embedding_model: str
    ) -> dict[str, object]:
        state = connection.execute(
            """
            SELECT initialized FROM concept_similarity_state
            WHERE embedding_model = ?
            """,
            (embedding_model,),
        ).fetchone()
        if state is None or not int(state["initialized"]):
            recomputed = self._refresh_concept_similarity_cache(
                connection, embedding_model, None
            )
            return {"mode": "full", "pairs": recomputed}
        dirty = {
            str(row["label"])
            for row in connection.execute(
                """
                SELECT label FROM concept_similarity_dirty WHERE embedding_model = ?
                """,
                (embedding_model,),
            )
        }
        if dirty:
            recomputed = self._refresh_concept_similarity_cache(
                connection, embedding_model, dirty
            )
            return {"mode": "incremental", "labels": len(dirty), "pairs": recomputed}
        return {"mode": "hit", "pairs": 0}

    def graph(self, neighbours: int = 3) -> dict:
        """Build the graph from cached exact concept similarities."""
        with connect(self.db_path) as connection:
            model_counts = connection.execute(
                """
                SELECT embedding_model, COUNT(*) AS count FROM citations
                WHERE integration_state = 'integrated' AND analysis_state = 'ready'
                GROUP BY embedding_model
                """
            ).fetchall()
            total = sum(int(row["count"]) for row in model_counts)
            large_count = next(
                (int(row["count"]) for row in model_counts if row["embedding_model"] == BGE_MODEL),
                0,
            )
            graph_model = BGE_MODEL if total and large_count == total else LEGACY_BGE_MODEL
            rows = connection.execute(
                """
                SELECT citations.id, label,
                       graph_positions.x AS position_x,
                       graph_positions.y AS position_y
                FROM citations
                LEFT JOIN graph_positions ON graph_positions.citation_id = citations.id
                WHERE embedding_model = ? AND integration_state = 'integrated'
                  AND analysis_state = 'ready'
                ORDER BY citations.id
                """,
                (graph_model,),
            ).fetchall()
            cache = self._ensure_concept_similarity_cache(connection, graph_model)
            similarity_rows = connection.execute(
                """
                SELECT label_a, label_b, similarity FROM concept_similarities
                WHERE embedding_model = ?
                """,
                (graph_model,),
            ).fetchall()

        concepts: dict[str, list[sqlite3.Row]] = {}
        for row in rows:
            concepts.setdefault(str(row["label"]), []).append(row)

        nodes = []
        for label, facts in concepts.items():
            representative = facts[0]
            fact_nodes = [{"id": row["id"]} for row in facts]
            nodes.append(
                {
                    "id": representative["id"],
                    "label": label,
                    "facts": fact_nodes,
                    "fact_edges": [],
                    "fact_count": len(fact_nodes),
                    "density": 0.0,
                    "degree": 0,
                    "position": (
                        {"x": representative["position_x"], "y": representative["position_y"]}
                        if representative["position_x"] is not None
                        and representative["position_y"] is not None
                        else None
                    ),
                }
            )

        if len(nodes) < 2:
            return {
                "nodes": nodes,
                "edges": [],
                "neighbours": 0,
                "baseline": 0.0,
                "cache": cache,
            }

        count = len(nodes)
        label_indexes = {node["label"]: index for index, node in enumerate(nodes)}
        similarities = np.eye(count, dtype=np.float32)
        for row in similarity_rows:
            left = label_indexes.get(str(row["label_a"]))
            right = label_indexes.get(str(row["label_b"]))
            if left is None or right is None:
                continue
            similarities[left, right] = similarities[right, left] = float(row["similarity"])
        k = min(neighbours, count - 1)
        nearest = [
            [int(item) for item in np.argsort(-similarities[index]) if int(item) != index][
                :k
            ]
            for index in range(count)
        ]
        off_diagonal = similarities[~np.eye(count, dtype=bool)]
        baseline = float(np.median(off_diagonal))
        edges = []
        for left in range(count):
            for right in nearest[left]:
                if left >= right or left not in nearest[right]:
                    continue
                similarity = float(similarities[left, right])
                strength = max(
                    0.0, (similarity - baseline) / max(1e-9, 1.0 - baseline)
                )
                edges.append(
                    {
                        "source": nodes[left]["id"],
                        "target": nodes[right]["id"],
                        "similarity": round(similarity, 4),
                        "strength": round(strength, 4),
                    }
                )
                nodes[left]["density"] += strength
                nodes[right]["density"] += strength
                nodes[left]["degree"] += 1
                nodes[right]["degree"] += 1

        for node in nodes:
            node["density"] = round(node["density"], 4)
        return {
            "nodes": nodes,
            "edges": edges,
            "neighbours": k,
            "baseline": round(baseline, 4),
            "cache": cache,
        }

    @staticmethod
    def _refresh_fact_similarity_cache(
        connection: sqlite3.Connection,
        embedding_model: str,
        label: str,
        neighbours: int,
    ) -> int:
        rows = connection.execute(
            """
            SELECT id, embedding FROM citations
            WHERE label = ? AND embedding_model = ?
              AND integration_state = 'integrated' AND analysis_state = 'ready'
            ORDER BY id
            """,
            (label, embedding_model),
        ).fetchall()
        connection.execute(
            """
            DELETE FROM fact_similarity_edges
            WHERE embedding_model = ? AND label = ?
            """,
            (embedding_model, label),
        )
        pairs: set[tuple[int, int]] = set()
        similarities: np.ndarray | None = None
        if len(rows) >= 2:
            vectors = np.stack(
                [np.frombuffer(row["embedding"], dtype=np.float32) for row in rows]
            )
            similarities = vectors @ vectors.T
            count = len(rows)
            k = min(max(1, neighbours), count - 1)
            for index in range(count):
                nearest = np.argsort(-similarities[index])
                added_count = 0
                for other in nearest:
                    other = int(other)
                    if other == index:
                        continue
                    pairs.add(tuple(sorted((index, other))))
                    added_count += 1
                    if added_count >= k:
                        break

            # A maximum spanning tree prevents sparse nearest-neighbour islands.
            visited = {0}
            best = similarities[0].copy()
            parent = np.zeros(count, dtype=np.int32)
            best[0] = -np.inf
            while len(visited) < count:
                candidates = [index for index in range(count) if index not in visited]
                added = max(candidates, key=lambda index: float(best[index]))
                pairs.add(tuple(sorted((int(parent[added]), added))))
                visited.add(added)
                for candidate in candidates:
                    if (
                        candidate not in visited
                        and similarities[added, candidate] > best[candidate]
                    ):
                        best[candidate] = similarities[added, candidate]
                        parent[candidate] = added

            connection.executemany(
                """
                INSERT INTO fact_similarity_edges
                    (embedding_model, label, source_id, target_id, similarity)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        embedding_model,
                        label,
                        int(rows[left]["id"]),
                        int(rows[right]["id"]),
                        float(similarities[left, right]),
                    )
                    for left, right in sorted(pairs)
                ],
            )

        if rows:
            connection.execute(
                """
                INSERT INTO fact_similarity_state
                    (embedding_model, label, neighbours, version)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(embedding_model, label) DO UPDATE SET
                    neighbours = excluded.neighbours,
                    version = excluded.version,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (embedding_model, label, neighbours, FACT_EDGE_CACHE_VERSION),
            )
        else:
            connection.execute(
                """
                DELETE FROM fact_similarity_state
                WHERE embedding_model = ? AND label = ?
                """,
                (embedding_model, label),
            )
        connection.execute(
            """
            DELETE FROM fact_similarity_dirty
            WHERE embedding_model = ? AND label = ?
            """,
            (embedding_model, label),
        )
        return len(pairs)

    def _ensure_fact_similarity_cache(
        self,
        connection: sqlite3.Connection,
        embedding_model: str,
        label: str,
        neighbours: int,
    ) -> dict[str, object]:
        state = connection.execute(
            """
            SELECT neighbours, version FROM fact_similarity_state
            WHERE embedding_model = ? AND label = ?
            """,
            (embedding_model, label),
        ).fetchone()
        dirty = connection.execute(
            """
            SELECT 1 FROM fact_similarity_dirty
            WHERE embedding_model = ? AND label = ?
            """,
            (embedding_model, label),
        ).fetchone()
        valid = (
            state is not None
            and int(state["neighbours"]) == neighbours
            and int(state["version"]) == FACT_EDGE_CACHE_VERSION
        )
        if valid and dirty is None:
            return {"mode": "hit", "edges": 0}
        edges = self._refresh_fact_similarity_cache(
            connection, embedding_model, label, neighbours
        )
        return {"mode": "refresh" if state is not None else "build", "edges": edges}

    def concept_facts(self, concept_id: int, neighbours: int = 3) -> dict:
        """Load one concept's details using its cached sparse relationship graph."""
        with connect(self.db_path) as connection:
            concept = connection.execute(
                """
                SELECT label, embedding_model FROM citations
                WHERE id = ? AND integration_state = 'integrated'
                  AND analysis_state = 'ready'
                """,
                (concept_id,),
            ).fetchone()
            if concept is None:
                raise KeyError(concept_id)
            cache = self._ensure_fact_similarity_cache(
                connection,
                str(concept["embedding_model"]),
                str(concept["label"]),
                neighbours,
            )
            rows = connection.execute(
                """
                SELECT citations.id, source, highlight, fact_caption, caption_version,
                       created_at,
                       fact_positions.x AS position_x,
                       fact_positions.y AS position_y
                FROM citations
                LEFT JOIN fact_positions ON fact_positions.citation_id = citations.id
                WHERE label = ? AND embedding_model = ?
                  AND integration_state = 'integrated' AND analysis_state = 'ready'
                ORDER BY citations.id
                """,
                (concept["label"], concept["embedding_model"]),
            ).fetchall()
            edge_rows = connection.execute(
                """
                SELECT source_id, target_id, similarity FROM fact_similarity_edges
                WHERE embedding_model = ? AND label = ?
                ORDER BY source_id, target_id
                """,
                (concept["embedding_model"], concept["label"]),
            ).fetchall()

        facts = [
            {
                "id": int(row["id"]),
                "source": row["source"],
                "highlight": row["highlight"],
                "caption": row["fact_caption"],
                "caption_version": row["caption_version"],
                "created_at": row["created_at"],
                "position": (
                    {"x": row["position_x"], "y": row["position_y"]}
                    if row["position_x"] is not None and row["position_y"] is not None
                    else None
                ),
            }
            for row in rows
        ]
        edges = [
            {
                "source": int(row["source_id"]),
                "target": int(row["target_id"]),
                "similarity": round(float(row["similarity"]), 4),
            }
            for row in edge_rows
        ]
        return {
            "id": concept_id,
            "label": concept["label"],
            "facts": facts,
            "fact_edges": edges,
            "cache": cache,
        }

    def save_graph_positions(self, positions: list[tuple[int, float, float]]) -> None:
        citation_ids = {citation_id for citation_id, _, _ in positions}
        with connect(self.db_path) as connection:
            placeholders = ",".join("?" for _ in citation_ids)
            existing = {
                int(row["id"])
                for row in connection.execute(
                    f"SELECT id FROM citations WHERE id IN ({placeholders})",
                    tuple(citation_ids),
                )
            }
            missing = citation_ids - existing
            if missing:
                raise ValueError(f"Unknown citation IDs: {sorted(missing)}")
            connection.executemany(
                """
                INSERT INTO graph_positions (citation_id, x, y)
                VALUES (?, ?, ?)
                ON CONFLICT(citation_id) DO UPDATE SET
                    x = excluded.x,
                    y = excluded.y,
                    updated_at = CURRENT_TIMESTAMP
                """,
                positions,
            )

    def save_fact_positions(self, positions: list[tuple[int, float, float]]) -> None:
        citation_ids = {citation_id for citation_id, _, _ in positions}
        with connect(self.db_path) as connection:
            placeholders = ",".join("?" for _ in citation_ids)
            existing = {
                int(row["id"])
                for row in connection.execute(
                    f"SELECT id FROM citations WHERE id IN ({placeholders})",
                    tuple(citation_ids),
                )
            }
            missing = citation_ids - existing
            if missing:
                raise ValueError(f"Unknown citation IDs: {sorted(missing)}")
            connection.executemany(
                """
                INSERT INTO fact_positions (citation_id, x, y)
                VALUES (?, ?, ?)
                ON CONFLICT(citation_id) DO UPDATE SET
                    x = excluded.x,
                    y = excluded.y,
                    updated_at = CURRENT_TIMESTAMP
                """,
                positions,
            )

    def nearest(self, vector: np.ndarray, embedding_model: str, limit: int = 5) -> list[dict]:
        with connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT id, source, highlight, label, embedding
                FROM citations
                WHERE embedding_model = ? AND integration_state = 'integrated'
                """,
                (embedding_model,),
            ).fetchall()
        scored = []
        for row in rows:
            candidate = np.frombuffer(row["embedding"], dtype=np.float32)
            if candidate.shape != vector.shape:
                continue
            scored.append(
                {
                    "id": row["id"],
                    "source": row["source"],
                    "highlight": row["highlight"],
                    "label": row["label"],
                    "similarity": round(float(vector @ candidate), 4),
                }
            )
        scored.sort(key=lambda item: item["similarity"], reverse=True)
        return scored[:limit]

    def ensure_embedding_model(self, embedding_model: str, embed: object) -> int:
        """Atomically migrate every stored citation to one embedding space."""
        with connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT id, highlight FROM citations
                WHERE embedding_model != ? AND analysis_state = 'ready'
                ORDER BY id
                """,
                (embedding_model,),
            ).fetchall()
        if not rows:
            return 0
        vectors = embed([str(row["highlight"]) for row in rows])
        with connect(self.db_path) as connection:
            connection.executemany(
                "UPDATE citations SET embedding_model = ?, embedding = ? WHERE id = ?",
                [
                    (embedding_model, vector.astype(np.float32).tobytes(), int(row["id"]))
                    for row, vector in zip(rows, vectors, strict=True)
                ],
            )
        return len(rows)

    def add(
        self,
        source: str,
        highlight: str,
        context: str,
        analysis: KnowledgeAnalysis,
        *,
        document_id: int | None = None,
        document_start: int | None = None,
        document_end: int | None = None,
        chat_message_id: str | None = None,
        chat_start: int | None = None,
        chat_end: int | None = None,
        source_segments: list[list[int]] | None = None,
    ) -> int:
        with connect(self.db_path) as connection:
            cursor = connection.execute(
                """
                INSERT INTO citations
                    (source, highlight, context, label, embedding_model, embedding,
                     document_id, document_start, document_end,
                     chat_message_id, chat_start, chat_end, source_segments)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source,
                    highlight,
                    context,
                    analysis.label,
                    BGE_MODEL,
                    analysis.vector.tobytes(),
                    document_id,
                    document_start,
                    document_end,
                    chat_message_id,
                    chat_start,
                    chat_end,
                    json.dumps(source_segments) if source_segments else None,
                ),
            )
            citation_id = int(cursor.lastrowid)
            self._record_graph_action(
                connection, citation_id, "integrate", "not_integrated", "integrated"
            )
            return citation_id

    def add_unintegrated(self, source: str, highlight: str, context: str) -> int:
        with connect(self.db_path) as connection:
            cursor = connection.execute(
                """
                INSERT INTO citations
                    (source, highlight, context, label, embedding_model, embedding,
                     integration_state, analysis_state)
                VALUES (?, ?, ?, '', '', ?, 'not_integrated', 'unprocessed')
                """,
                (source, highlight, context, b""),
            )
            return int(cursor.lastrowid)

    @staticmethod
    def _prepare_label_change(
        connection: sqlite3.Connection, citation_ids: set[int], new_label: str
    ) -> None:
        placeholders = ",".join("?" for _ in citation_ids)
        rows = connection.execute(
            f"SELECT id, label FROM citations WHERE id IN ({placeholders})",
            tuple(citation_ids),
        ).fetchall()
        old_labels = {str(row["label"]) for row in rows if str(row["label"]) != new_label}
        for old_label in old_labels:
            representative = connection.execute(
                """
                SELECT citations.id, graph_positions.x, graph_positions.y
                FROM citations
                LEFT JOIN graph_positions ON graph_positions.citation_id = citations.id
                WHERE citations.label = ? AND citations.integration_state = 'integrated'
                ORDER BY citations.id LIMIT 1
                """,
                (old_label,),
            ).fetchone()
            if representative is None or int(representative["id"]) not in citation_ids:
                continue
            remaining = connection.execute(
                f"""
                SELECT id FROM citations
                WHERE label = ? AND integration_state = 'integrated'
                  AND id NOT IN ({placeholders})
                ORDER BY id LIMIT 1
                """,
                (old_label, *citation_ids),
            ).fetchone()
            if remaining is None:
                continue
            if representative["x"] is not None and representative["y"] is not None:
                connection.execute(
                    """
                    INSERT INTO graph_positions (citation_id, x, y)
                    VALUES (?, ?, ?)
                    ON CONFLICT(citation_id) DO UPDATE SET
                        x = excluded.x, y = excluded.y, updated_at = CURRENT_TIMESTAMP
                    """,
                    (
                        int(remaining["id"]),
                        float(representative["x"]),
                        float(representative["y"]),
                    ),
                )
            connection.execute(
                "DELETE FROM graph_positions WHERE citation_id = ?",
                (int(representative["id"]),),
            )

    def rename_labels(self, citation_ids: list[int], label: str) -> list[dict]:
        ids = set(citation_ids)
        if not ids:
            raise ValueError("Choose at least one fact to rename.")
        placeholders = ",".join("?" for _ in ids)
        with connect(self.db_path) as connection:
            existing = connection.execute(
                f"SELECT id FROM citations WHERE id IN ({placeholders})", tuple(ids)
            ).fetchall()
            if len(existing) != len(ids):
                raise KeyError("Unknown fact")
            self._prepare_label_change(connection, ids, label)
            connection.execute(
                f"""
                UPDATE citations SET label = ?, label_locked = 1
                WHERE id IN ({placeholders})
                """,
                (label, *ids),
            )
        return [self.get(citation_id) for citation_id in sorted(ids)]

    def update(
        self,
        citation_id: int,
        highlight: str,
        analysis: KnowledgeAnalysis,
        *,
        label_locked: bool,
    ) -> None:
        with connect(self.db_path) as connection:
            self._prepare_label_change(connection, {citation_id}, analysis.label)
            cursor = connection.execute(
                """
                UPDATE citations SET
                    highlight = ?, label = ?, embedding_model = ?, embedding = ?,
                    fact_caption = NULL, caption_version = NULL,
                    draft_highlight = NULL, draft_updated_at = NULL,
                    integration_state = 'integrated', analysis_state = 'ready',
                    label_locked = ?
                WHERE id = ?
                """,
                (
                    highlight,
                    analysis.label,
                    BGE_MODEL,
                    analysis.vector.astype(np.float32).tobytes(),
                    int(label_locked),
                    citation_id,
                ),
            )
            if cursor.rowcount != 1:
                raise KeyError(citation_id)
            self._record_graph_action(
                connection, citation_id, "integrate", "not_integrated", "integrated"
            )

    def save_draft(self, citation_id: int, draft_highlight: str) -> dict:
        draft_highlight = draft_highlight.strip()
        with connect(self.db_path) as connection:
            row = connection.execute(
                "SELECT highlight, document_id, chat_message_id FROM citations WHERE id = ?",
                (citation_id,),
            ).fetchone()
            if row is None:
                raise KeyError(citation_id)
            if (row["document_id"] is not None or row["chat_message_id"] is not None) and draft_highlight != str(row["highlight"]):
                raise ValueError("Document and chat excerpts are read-only.")
            dirty = bool(draft_highlight and draft_highlight != str(row["highlight"]))
            connection.execute(
                """
                UPDATE citations SET
                    draft_highlight = ?,
                    draft_updated_at = CASE WHEN ? IS NULL THEN NULL ELSE CURRENT_TIMESTAMP END
                WHERE id = ?
                """,
                (draft_highlight if dirty else None, draft_highlight if dirty else None, citation_id),
            )
        return {
            "id": citation_id,
            "dirty": dirty,
            "draft_highlight": draft_highlight if dirty else None,
        }

    def delete(self, citation_id: int) -> dict:
        with connect(self.db_path) as connection:
            connection.execute("DELETE FROM graph_history WHERE applied = 0")
            row = connection.execute(
                """
                SELECT citations.id, citations.label, graph_positions.x, graph_positions.y
                FROM citations
                LEFT JOIN graph_positions ON graph_positions.citation_id = citations.id
                WHERE citations.id = ?
                """,
                (citation_id,),
            ).fetchone()
            if row is None:
                raise KeyError(citation_id)
            replacement = connection.execute(
                """
                SELECT id FROM citations
                WHERE label = ? AND id != ? AND integration_state = 'integrated'
                ORDER BY id LIMIT 1
                """,
                (row["label"], citation_id),
            ).fetchone()
            if replacement is not None and row["x"] is not None and row["y"] is not None:
                connection.execute(
                    """
                    INSERT INTO graph_positions (citation_id, x, y)
                    VALUES (?, ?, ?)
                    ON CONFLICT(citation_id) DO UPDATE SET
                        x = excluded.x, y = excluded.y, updated_at = CURRENT_TIMESTAMP
                    """,
                    (int(replacement["id"]), float(row["x"]), float(row["y"])),
                )
            connection.execute("DELETE FROM graph_positions WHERE citation_id = ?", (citation_id,))
            connection.execute("DELETE FROM fact_positions WHERE citation_id = ?", (citation_id,))
            connection.execute("DELETE FROM graph_history WHERE citation_id = ?", (citation_id,))
            connection.execute("DELETE FROM citations WHERE id = ?", (citation_id,))
        return {"deleted": citation_id, "replacement": int(replacement["id"]) if replacement else None}

    def caption_targets(self, label: str) -> list[dict[str, object]]:
        """Return uncaptioned facts only after a concept gains a second fact."""
        with connect(self.db_path) as connection:
            count = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM citations
                    WHERE label = ? AND integration_state = 'integrated'
                    """,
                    (label,),
                ).fetchone()[0]
            )
            if count < 2:
                return []
            rows = connection.execute(
                """
                SELECT id, highlight FROM citations
                WHERE label = ? AND fact_caption IS NULL
                  AND integration_state = 'integrated'
                ORDER BY id
                """,
                (label,),
            ).fetchall()
        return [dict(row) for row in rows]

    def save_caption(self, citation_id: int, caption: str) -> None:
        with connect(self.db_path) as connection:
            connection.execute(
                """
                UPDATE citations SET fact_caption = ?, caption_version = ?
                WHERE id = ?
                """,
                (caption, CAPTION_VERSION, citation_id),
            )


class CaptureApplication:
    def __init__(self, db_path: Path) -> None:
        self.store = CaptureStore(db_path)
        saved_profile = self.store.get_setting("model_profile")
        self.models = LocalKnowledgeModels(
            t5_enabled=(saved_profile == "enhanced") if saved_profile else None
        )
        agent_workspace = db_path.parent / "agent-workspace"
        agent_workspace.mkdir(parents=True, exist_ok=True)
        self.agent_plugins = AgentPluginRegistry([CodexAgentPlugin(agent_workspace)])
        self._capture_lock = threading.Lock()
        self._jobs_lock = threading.Lock()
        self._jobs: dict[str, dict[str, object]] = {}
        self._model_setup_lock = threading.Lock()
        self._model_setup_jobs: dict[str, dict[str, object]] = {}
        self._chat_jobs_lock = threading.Lock()
        self._chat_jobs: dict[str, dict[str, object]] = {}
        self._chat_job_by_thread: dict[str, str] = {}

    def model_setup_status(self) -> dict[str, object]:
        status = self.models.cache_status()
        configured = self.store.get_setting("model_profile") in {"standard", "enhanced"}
        status["configured"] = configured
        status["ready"] = bool(status["ready"] and configured)
        return status

    def start_model_setup(self, payload: dict) -> dict[str, object]:
        profile = str(payload.get("profile", "standard")).strip().lower()
        if profile not in {"standard", "enhanced"}:
            raise ValueError("Model profile must be standard or enhanced.")
        cached = self.models.cache_status()
        cached_models = cached.get("models", {})
        t5_cached = bool(
            isinstance(cached_models, dict)
            and isinstance(cached_models.get("t5gemma"), dict)
            and cached_models["t5gemma"].get("cached")
        )
        if profile == "enhanced" and not t5_cached:
            access = self.models.check_t5_access()
            if not access["available"]:
                raise ValueError(str(access["message"]))
        with self._model_setup_lock:
            active = next(
                (
                    dict(job)
                    for job in self._model_setup_jobs.values()
                    if job["state"] in {"queued", "running"}
                ),
                None,
            )
            if active is not None:
                return active
            job_id = uuid.uuid4().hex
            stages = {
                "qwen": "queued",
                "bge": "queued",
                "t5gemma": "queued" if profile == "enhanced" else "disabled",
            }
            job: dict[str, object] = {
                "id": job_id,
                "profile": profile,
                "state": "queued",
                "stages": stages,
                "error": None,
            }
            self._model_setup_jobs[job_id] = job

        def update(stage: str, state: str) -> None:
            with self._model_setup_lock:
                stages = self._model_setup_jobs[job_id]["stages"]
                assert isinstance(stages, dict)
                stages[stage] = state

        def work() -> None:
            with self._model_setup_lock:
                self._model_setup_jobs[job_id]["state"] = "running"
            try:
                self.models.prepare(profile == "enhanced", update)
                self.store.set_setting("model_profile", profile)
                with self._model_setup_lock:
                    self._model_setup_jobs[job_id]["state"] = "complete"
                    self._model_setup_jobs[job_id]["status"] = self.model_setup_status()
            except Exception as error:
                print(f"Model setup failed: {error!r}", flush=True)
                with self._model_setup_lock:
                    self._model_setup_jobs[job_id].update(
                        state="failed", error=str(error)
                    )

        threading.Thread(
            target=work, name=f"model-setup-{job_id[:8]}", daemon=True
        ).start()
        return dict(job)

    def get_model_setup_job(self, job_id: str) -> dict[str, object]:
        with self._model_setup_lock:
            job = self._model_setup_jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            return dict(job)

    def list_agent_plugins(self) -> list[dict[str, object]]:
        return self.agent_plugins.list()

    def create_chat(self, payload: dict) -> dict:
        plugin_id = str(payload.get("agent_plugin_id", "codex")).strip() or "codex"
        self.agent_plugins.get(plugin_id)
        return {"chat": self.store.create_chat(plugin_id)}

    def get_chat(self, chat_id: str) -> dict:
        chat = self.store.get_chat(chat_id)
        with self._chat_jobs_lock:
            job_id = self._chat_job_by_thread.get(chat_id)
            job = dict(self._chat_jobs[job_id]) if job_id in self._chat_jobs else None
        chat["active_turn"] = job
        return {"chat": chat}

    def start_chat_turn(self, chat_id: str, payload: dict) -> dict:
        content = str(payload.get("content", "")).strip()
        mode = str(payload.get("mode", "document_first")).strip()
        if mode not in {"document_first", "chat_only"}:
            raise ValueError("Invalid chat workspace mode.")
        if not content:
            raise ValueError("Type a message first.")
        if len(content) > MAX_TEXT_LENGTH:
            raise ValueError(f"Message must be shorter than {MAX_TEXT_LENGTH:,} characters.")
        with self._chat_jobs_lock:
            existing_id = self._chat_job_by_thread.get(chat_id)
            existing = self._chat_jobs.get(existing_id) if existing_id else None
            if existing and existing.get("state") in {"queued", "running"}:
                raise ValueError("This chat already has a turn in progress.")

        references = payload.get("references", [])
        if not isinstance(references, list) or len(references) > 12:
            raise ValueError("A message may contain at most 12 document references.")
        workspace = self.store.chat_workspace(chat_id)
        for reference in references:
            if not isinstance(reference, dict) or int(reference.get("document_id", -1)) != int(workspace["id"]):
                raise ValueError("Chat references must point into this chat's working document.")
        plugin_id, conversation_id = self.store.chat_provider_state(chat_id)
        plugin = self.agent_plugins.get(plugin_id)
        user_message = self.store.add_chat_message(chat_id, "user", content, references)
        reference_text = "\n\n".join(
            f"REFERENCE {index + 1} [{int(reference['document_start'])}:{int(reference['document_end'])}]:\n"
            f"{workspace['content'][int(reference['document_start']):int(reference['document_end'])]}"
            for index, reference in enumerate(references)
        ) or "(none)"
        protected_text = ", ".join(
            f"{item['kind']}:{item['start']}..{item['end']}"
            for item in workspace["protected_ranges"]
        ) or "(none)"
        mode_contract = (
            "DOCUMENT-FIRST MODE IS ACTIVE. Every substantive explanation, synthesis, or research answer "
            "must be written into the working document. Initialize an empty document automatically. "
            "Do not place the full knowledge answer in chat: chat should contain only a short description "
            "of what you changed or a necessary clarification question. You MUST return a document command "
            "block unless no durable knowledge can reasonably be added."
            if mode == "document_first" else
            "CHAT-ONLY MODE IS ACTIVE. Answer in chat and do not return document commands."
        )
        agent_prompt = f"""The user is discussing a working knowledge document with you.
{mode_contract}
Use the document and references to answer them. When changing the document, return an all-or-nothing
command block at the END of your short chat reply using exactly this format:
<document_commands>{{"revision":{workspace['revision']},"commands":[{{"op":"insert","at":0,"text":"..."}},{{"op":"replace","start":0,"end":1,"text":"..."}},{{"op":"delete","start":0,"end":1}}]}}</document_commands>
Use character offsets from the exact document below. In chat-only mode, omit the block.
Never edit through a protected range. A rejected command will be reported in chat.

DOCUMENT id={workspace['id']} revision={workspace['revision']} name={workspace['name']}
PROTECTED RANGES: {protected_text}
--- document begins ---
{workspace['content']}
--- document ends ---

ATTACHED REFERENCES:
{reference_text}

USER MESSAGE:
{content}"""
        job_id = uuid.uuid4().hex
        job: dict[str, object] = {
            "id": job_id,
            "chat_id": chat_id,
            "state": "queued",
            "partial": "",
            "error": None,
            "user_message_id": user_message["id"],
        }
        with self._chat_jobs_lock:
            self._chat_jobs[job_id] = job
            self._chat_job_by_thread[chat_id] = job_id

        def on_delta(delta: str) -> None:
            with self._chat_jobs_lock:
                current = self._chat_jobs.get(job_id)
                if current is not None:
                    current["partial"] = str(current.get("partial", "")) + delta

        def work() -> None:
            with self._chat_jobs_lock:
                self._chat_jobs[job_id]["state"] = "running"
            try:
                result = plugin.run_turn(conversation_id, agent_prompt, on_delta)
                edit_status = ""
                try:
                    visible_answer, command_payload = self._parse_document_commands(result.answer)
                except Exception as parse_error:
                    visible_answer, command_payload = result.answer, None
                    edit_status = f"\n\n_Document edit rejected: invalid command block ({parse_error})._"
                if mode == "document_first" and command_payload is None:
                    correction = (
                        "Your previous response violated document-first mode because it put the knowledge "
                        "in chat without returning a document command. Correct it now: update the working "
                        f"document at revision {workspace['revision']} using the required "
                        "<document_commands> JSON block. Keep the visible chat reply to one short sentence."
                    )
                    with self._chat_jobs_lock:
                        self._chat_jobs[job_id]["partial"] = ""
                    result = plugin.run_turn(result.conversation_id, correction, on_delta)
                    visible_answer, command_payload = self._parse_document_commands(result.answer)
                    if command_payload is None:
                        raise RuntimeError("The agent did not produce the required document update.")
                if mode == "chat_only" and command_payload is not None:
                    command_payload = None
                    edit_status = "\n\n_Document commands were ignored because chat-only mode was active._"
                if command_payload is not None:
                    try:
                        outcome = self.store.apply_document_commands(
                            int(workspace["id"]),
                            int(command_payload.get("revision", -1)),
                            command_payload.get("commands", []),
                        )
                        edit_status = (
                            f"\n\n_Document updated to revision {outcome['revision']} "
                            f"({outcome['applied']} command{'s' if outcome['applied'] != 1 else ''})._"
                        )
                    except Exception as edit_error:
                        edit_status = f"\n\n_Document edit rejected: {edit_error}_"
                visible_answer = (visible_answer.strip() + edit_status).strip()
                assistant = self.store.complete_chat_turn(
                    chat_id, result.conversation_id, visible_answer, content
                )
                with self._chat_jobs_lock:
                    self._chat_jobs[job_id].update(
                        state="complete", partial=visible_answer,
                        assistant_message_id=assistant["id"],
                    )
            except Exception as error:
                print(f"Chat turn failed: {error!r}", flush=True)
                with self._chat_jobs_lock:
                    self._chat_jobs[job_id].update(state="failed", error=str(error))

        threading.Thread(target=work, name=f"chat-{job_id[:8]}", daemon=True).start()
        return {"turn": dict(job), "message": user_message}

    @staticmethod
    def _parse_document_commands(answer: str) -> tuple[str, dict | None]:
        pattern = re.compile(r"\s*<document_commands>(.*?)</document_commands>\s*", re.DOTALL)
        matches = list(pattern.finditer(answer))
        if not matches:
            return answer, None
        if len(matches) != 1:
            return answer, {"revision": -1, "commands": []}
        payload = json.loads(matches[0].group(1))
        if not isinstance(payload, dict) or not isinstance(payload.get("commands"), list):
            raise ValueError("The agent returned an invalid document command block.")
        visible = answer[: matches[0].start()] + answer[matches[0].end() :]
        return visible.strip(), payload

    def save_document(self, payload: dict) -> dict:
        name = Path(str(payload.get("name", "untitled"))).name.strip() or "untitled"
        content = str(payload.get("content", ""))
        if not content.strip():
            raise ValueError("The document is empty.")
        if len(name) > 255:
            raise ValueError("Document name is too long.")
        if len(content) > MAX_DOCUMENT_LENGTH:
            raise ValueError(
                f"Document must be shorter than {MAX_DOCUMENT_LENGTH:,} characters."
            )
        return {"document": self.store.add_document(name, content)}

    def _document_selection(
        self, payload: dict, highlight: str
    ) -> tuple[int | None, int | None, int | None, list[list[int]] | None]:
        raw_document_id = payload.get("document_id")
        if raw_document_id is None:
            return None, None, None, None
        try:
            document_id = int(raw_document_id)
        except (TypeError, ValueError) as error:
            raise ValueError("Invalid document ID.") from error
        document = self.store.get_document(document_id)
        content = str(document["content"])
        segments = self._source_segments(payload.get("source_segments"), content, highlight)
        if segments:
            return document_id, segments[0][0], segments[-1][1], segments
        if "document_start" not in payload or "document_end" not in payload:
            start = content.find(highlight)
            end = start + len(highlight) if start >= 0 else -1
        else:
            try:
                start = int(payload["document_start"])
                end = int(payload["document_end"])
            except (TypeError, ValueError) as error:
                raise ValueError("Document selections need numeric start and end offsets.") from error
        if document_id < 1 or start < 0 or end <= start or end > len(content):
            raise ValueError("Invalid document selection range.")
        if content[start:end] != highlight:
            raise ValueError("The selected text no longer matches the stored document.")
        return document_id, start, end, None

    def _chat_selection(
        self, payload: dict, highlight: str
    ) -> tuple[str | None, int | None, int | None, list[list[int]] | None]:
        raw_message_id = payload.get("chat_message_id")
        if raw_message_id is None:
            return None, None, None, None
        message_id = str(raw_message_id).strip()
        if not message_id:
            raise ValueError("Invalid chat message ID.")
        with connect(self.store.db_path) as connection:
            message = connection.execute(
                "SELECT content FROM chat_messages WHERE id = ?", (message_id,)
            ).fetchone()
        if message is None:
            raise ValueError("Chat message not found.")
        content = str(message["content"])
        segments = self._source_segments(payload.get("source_segments"), content, highlight)
        if segments:
            return message_id, segments[0][0], segments[-1][1], segments
        try:
            start = int(payload["chat_start"])
            end = int(payload["chat_end"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Chat selections need numeric start and end offsets.") from error
        if start < 0 or end <= start or end > len(content):
            raise ValueError("Invalid chat selection range.")
        if content[start:end] != highlight:
            raise ValueError("The selected text no longer matches the stored chat message.")
        return message_id, start, end, None

    @staticmethod
    def _source_segments(
        raw_segments: object, content: str, highlight: str
    ) -> list[list[int]] | None:
        if raw_segments is None:
            return None
        if not isinstance(raw_segments, list) or not raw_segments or len(raw_segments) > 256:
            raise ValueError("Invalid source segments.")
        segments: list[list[int]] = []
        previous_end = -1
        for raw in raw_segments:
            if not isinstance(raw, list) or len(raw) != 2:
                raise ValueError("Invalid source segment.")
            try:
                start, end = int(raw[0]), int(raw[1])
            except (TypeError, ValueError) as error:
                raise ValueError("Source segments need numeric offsets.") from error
            if start < 0 or end <= start or end > len(content) or start < previous_end:
                raise ValueError("Invalid source segment range.")
            segments.append([start, end])
            previous_end = end
        selected = "\n\n".join(content[start:end].strip() for start, end in segments).strip()
        if selected != highlight:
            raise ValueError("The selected text no longer matches its source segments.")
        return segments

    def save_unintegrated(self, payload: dict) -> dict:
        highlight = str(payload.get("highlight", "")).strip()
        context = str(payload.get("context", "")).strip()
        source = str(payload.get("source", "untitled")).strip() or "untitled"
        if not highlight:
            raise ValueError("Enter some text first.")
        if len(highlight) > MAX_TEXT_LENGTH or len(context) > MAX_TEXT_LENGTH:
            raise ValueError(f"Text must be shorter than {MAX_TEXT_LENGTH:,} characters.")
        citation_id = self.store.add_unintegrated(source, highlight, context)
        return {
            "citation": {
                "id": citation_id,
                "source": source,
                "highlight": highlight,
                "context": context,
                "label": "",
                "integration_state": "not_integrated",
                "analysis_state": "unprocessed",
            }
        }

    def capture(
        self, payload: dict, progress: ProgressCallback | None = None
    ) -> dict:
        highlight = str(payload.get("highlight", "")).strip()
        context = str(payload.get("context", "")).strip()
        source = str(payload.get("source", "untitled")).strip() or "untitled"
        if not highlight:
            raise ValueError("Select some text first.")
        if len(highlight) > MAX_TEXT_LENGTH or len(context) > MAX_TEXT_LENGTH:
            raise ValueError(f"Text must be shorter than {MAX_TEXT_LENGTH:,} characters.")
        document_id, document_start, document_end, document_segments = self._document_selection(
            payload, highlight
        )
        chat_message_id, chat_start, chat_end, chat_segments = self._chat_selection(payload, highlight)
        if document_id is not None and chat_message_id is not None:
            raise ValueError("A fact cannot belong to both a document and a chat.")

        # The evidence itself determines graph similarity. Context is preserved
        # for later evaluation but deliberately does not alter this vector.
        with self._capture_lock:
            if progress:
                progress("bge", "running")
            migrated = self.store.ensure_embedding_model(BGE_MODEL, self.models.embed)
            if progress:
                progress("bge", "complete")
            analysis = self.models.analyze(highlight, progress)
            nearest = self.store.nearest(analysis.vector, BGE_MODEL)
            citation_id = self.store.add(
                source,
                highlight,
                context,
                analysis,
                document_id=document_id,
                document_start=document_start,
                document_end=document_end,
                chat_message_id=chat_message_id,
                chat_start=chat_start,
                chat_end=chat_end,
                source_segments=document_segments or chat_segments,
            )
            caption_targets = self.store.caption_targets(analysis.label)
            caption_model = "t5gemma" if self.models.t5_enabled else "qwen"
            for target in caption_targets:
                caption_stage = f"{caption_model}_caption_{target['id']}"
                if progress:
                    progress(caption_stage, "queued")
            for target in caption_targets:
                caption_stage = f"{caption_model}_caption_{target['id']}"
                if progress:
                    progress(caption_stage, "running")
                caption = self.models.caption_fact(str(target["highlight"]))
                self.store.save_caption(int(target["id"]), caption)
                if progress:
                    progress(caption_stage, "complete")
        return {
            "citation": {
                "id": citation_id,
                "source": source,
                "highlight": highlight,
                "context": context,
                "label": analysis.label,
                "embedding_model": BGE_MODEL,
                "document_id": document_id,
                "document_start": document_start,
                "document_end": document_end,
                "chat_message_id": chat_message_id,
                "chat_start": chat_start,
                "chat_end": chat_end,
            },
            "nearest": nearest,
            "selection": {
                "agreement": analysis.agreement,
                "candidates": analysis.candidates,
                "timings_ms": analysis.timings_ms,
            },
            "migrated_citations": migrated,
            "captions_generated": len(caption_targets),
        }

    def start_capture(self, payload: dict) -> dict:
        """Queue capture work and return without waiting for local inference."""
        highlight = str(payload.get("highlight", "")).strip()
        context = str(payload.get("context", "")).strip()
        if not highlight:
            raise ValueError("Enter some text first.")
        if len(highlight) > MAX_TEXT_LENGTH or len(context) > MAX_TEXT_LENGTH:
            raise ValueError(f"Text must be shorter than {MAX_TEXT_LENGTH:,} characters.")

        job_id = uuid.uuid4().hex
        job: dict[str, object] = {
            "id": job_id,
            "state": "queued",
            "stages": {
                "t5gemma": "queued" if self.models.t5_enabled else "disabled",
                "qwen": "queued",
                "bge": "queued",
            },
        }
        with self._jobs_lock:
            self._jobs[job_id] = job

        def update(model: str, state: str) -> None:
            with self._jobs_lock:
                stages = self._jobs[job_id]["stages"]
                assert isinstance(stages, dict)
                stages[model] = state

        def work() -> None:
            with self._jobs_lock:
                self._jobs[job_id]["state"] = "running"
            try:
                result = self.capture(payload, update)
                with self._jobs_lock:
                    self._jobs[job_id].update(state="complete", result=result)
            except Exception as error:
                print(f"Background capture failed: {error!r}", flush=True)
                with self._jobs_lock:
                    self._jobs[job_id].update(state="failed", error=str(error))

        threading.Thread(target=work, name=f"capture-{job_id[:8]}", daemon=True).start()
        return {"id": job_id, "state": "queued"}

    def update_fact(
        self,
        citation_id: int,
        payload: dict,
        progress: ProgressCallback | None = None,
    ) -> dict:
        highlight = str(payload.get("highlight", "")).strip()
        if not highlight:
            raise ValueError("Fact text cannot be empty.")
        if len(highlight) > MAX_TEXT_LENGTH:
            raise ValueError(f"Text must be shorter than {MAX_TEXT_LENGTH:,} characters.")
        existing = self.store.get(citation_id)
        if (existing["document_id"] is not None or existing["chat_message_id"] is not None) and highlight != str(existing["highlight"]):
            raise ValueError("Document and chat excerpts are read-only.")
        rediscover_label = payload.get("rediscover_label", False)
        if not isinstance(rediscover_label, bool):
            raise ValueError("rediscover_label must be boolean.")
        discover_label = rediscover_label or not str(existing["label"]).strip()
        with self._capture_lock:
            if progress:
                progress("bge", "running")
            migrated = self.store.ensure_embedding_model(BGE_MODEL, self.models.embed)
            if progress:
                progress("bge", "complete")
            analysis = (
                self.models.analyze(highlight, progress)
                if discover_label
                else self.models.embed_with_label(highlight, str(existing["label"]), progress)
            )
            nearest = self.store.nearest(analysis.vector, BGE_MODEL)
            self.store.update(
                citation_id,
                highlight,
                analysis,
                label_locked=(False if discover_label else bool(existing["label_locked"])),
            )
            caption_targets = self.store.caption_targets(analysis.label)
            caption_model = "t5gemma" if self.models.t5_enabled else "qwen"
            for target in caption_targets:
                if progress:
                    progress(f"{caption_model}_caption_{target['id']}", "queued")
            for target in caption_targets:
                stage = f"{caption_model}_caption_{target['id']}"
                if progress:
                    progress(stage, "running")
                caption = self.models.caption_fact(str(target["highlight"]))
                self.store.save_caption(int(target["id"]), caption)
                if progress:
                    progress(stage, "complete")
        return {
            "citation": {
                "id": citation_id,
                "source": existing["source"],
                "highlight": highlight,
                "context": existing["context"],
                "label": analysis.label,
                "label_locked": False if discover_label else bool(existing["label_locked"]),
                "embedding_model": BGE_MODEL,
            },
            "nearest": nearest,
            "selection": {
                "agreement": analysis.agreement,
                "candidates": analysis.candidates,
                "timings_ms": analysis.timings_ms,
            },
            "migrated_citations": migrated,
            "captions_generated": len(caption_targets),
        }

    def start_update_fact(self, citation_id: int, payload: dict) -> dict:
        highlight = str(payload.get("highlight", "")).strip()
        if not highlight:
            raise ValueError("Fact text cannot be empty.")
        existing = self.store.get(citation_id)
        if (existing["document_id"] is not None or existing["chat_message_id"] is not None) and highlight != str(existing["highlight"]):
            raise ValueError("Document and chat excerpts are read-only.")
        rediscover_label = payload.get("rediscover_label", False)
        if not isinstance(rediscover_label, bool):
            raise ValueError("rediscover_label must be boolean.")
        discover_label = rediscover_label or not str(existing["label"]).strip()
        job_id = uuid.uuid4().hex
        job: dict[str, object] = {
            "id": job_id,
            "state": "queued",
            "stages": (
                {
                    "t5gemma": "queued" if self.models.t5_enabled else "disabled",
                    "qwen": "queued",
                    "bge": "queued",
                }
                if discover_label
                else {"bge": "queued"}
            ),
        }
        with self._jobs_lock:
            self._jobs[job_id] = job

        def update(model: str, state: str) -> None:
            with self._jobs_lock:
                stages = self._jobs[job_id]["stages"]
                assert isinstance(stages, dict)
                stages[model] = state

        def work() -> None:
            with self._jobs_lock:
                self._jobs[job_id]["state"] = "running"
            try:
                result = self.update_fact(citation_id, payload, update)
                with self._jobs_lock:
                    self._jobs[job_id].update(state="complete", result=result)
            except Exception as error:
                print(f"Background update failed: {error!r}", flush=True)
                with self._jobs_lock:
                    self._jobs[job_id].update(state="failed", error=str(error))

        threading.Thread(target=work, name=f"update-{job_id[:8]}", daemon=True).start()
        return {"id": job_id, "state": "queued"}

    def get_capture_job(self, job_id: str) -> dict:
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            # JSON-compatible data provides a sufficient detached snapshot.
            return json.loads(json.dumps(job))

    def save_graph_positions(self, payload: dict) -> dict:
        raw_positions = payload.get("positions")
        if not isinstance(raw_positions, list) or not raw_positions:
            raise ValueError("positions must be a non-empty list.")
        if len(raw_positions) > 10_000:
            raise ValueError("Too many graph positions.")

        positions: list[tuple[int, float, float]] = []
        for raw in raw_positions:
            if not isinstance(raw, dict):
                raise ValueError("Each graph position must be an object.")
            try:
                citation_id = int(raw["id"])
                x = float(raw["x"])
                y = float(raw["y"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("Each graph position needs numeric id, x, and y.") from error
            if citation_id < 1 or not math.isfinite(x) or not math.isfinite(y):
                raise ValueError("Invalid graph position.")
            if not -10 <= x <= 10 or not -10 <= y <= 10:
                raise ValueError("Graph position is outside the supported range.")
            positions.append((citation_id, x, y))

        self.store.save_graph_positions(positions)
        return {"saved": len(positions)}

    def save_fact_positions(self, payload: dict) -> dict:
        raw_positions = payload.get("positions")
        if not isinstance(raw_positions, list) or not raw_positions:
            raise ValueError("positions must be a non-empty list.")
        if len(raw_positions) > 10_000:
            raise ValueError("Too many fact positions.")

        positions: list[tuple[int, float, float]] = []
        for raw in raw_positions:
            if not isinstance(raw, dict):
                raise ValueError("Each fact position must be an object.")
            try:
                citation_id = int(raw["id"])
                x = float(raw["x"])
                y = float(raw["y"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("Each fact position needs numeric id, x, and y.") from error
            if citation_id < 1 or not math.isfinite(x) or not math.isfinite(y):
                raise ValueError("Invalid fact position.")
            if not -10 <= x <= 10 or not -10 <= y <= 10:
                raise ValueError("Fact position is outside the supported range.")
            positions.append((citation_id, x, y))

        self.store.save_fact_positions(positions)
        return {"saved": len(positions)}


def make_handler(
    application: CaptureApplication, auth_token: str | None = None
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "Arbrellis/0.1"

        def log_message(self, format: str, *args: object) -> None:
            print(f"{self.address_string()} - {format % args}", flush=True)

        def add_security_headers(self) -> None:
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
                "font-src 'self'; connect-src 'self'; object-src 'none'; "
                "base-uri 'none'; frame-ancestors 'none'",
            )

        def is_authorized(self, parsed=None) -> bool:
            if not auth_token:
                return True
            authorization = self.headers.get("Authorization", "")
            if authorization.startswith("Bearer ") and secrets.compare_digest(
                authorization.removeprefix("Bearer "), auth_token
            ):
                return True
            cookies = self.headers.get("Cookie", "").split(";")
            for cookie in cookies:
                name, separator, value = cookie.strip().partition("=")
                if name == "arbrellis_session" and separator and secrets.compare_digest(
                    value, auth_token
                ):
                    return True
            if parsed is not None:
                supplied = parse_qs(parsed.query).get("session", [""])[0]
                return bool(supplied) and secrets.compare_digest(supplied, auth_token)
            return False

        def require_authorization(self, parsed=None) -> bool:
            if self.is_authorized(parsed):
                return True
            self.send_json({"error": "Unauthorized"}, HTTPStatus.UNAUTHORIZED)
            return False

        def send_json(self, value: object, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(value).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.add_security_headers()
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            parsed = urlparse(self.path)
            path = parsed.path
            if not self.require_authorization(parsed if path == "/" else None):
                return
            if path == "/":
                body = GRAPH_HTML_PATH.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.add_security_headers()
                if auth_token and parse_qs(parsed.query).get("session", [""])[0]:
                    self.send_header(
                        "Set-Cookie",
                        f"arbrellis_session={auth_token}; HttpOnly; SameSite=Strict; Path=/",
                    )
                self.end_headers()
                self.wfile.write(body)
            elif path.startswith("/vendor/katex/"):
                relative = path.removeprefix("/vendor/katex/")
                asset = (KATEX_DIST / relative).resolve()
                try:
                    asset.relative_to(KATEX_DIST.resolve())
                except ValueError:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                if not asset.is_file():
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                body = asset.read_bytes()
                content_type = mimetypes.guess_type(asset.name)[0] or "application/octet-stream"
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "public, max-age=86400")
                self.add_security_headers()
                self.end_headers()
                self.wfile.write(body)
            elif path == "/capture":
                body = HTML_PATH.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.add_security_headers()
                self.end_headers()
                self.wfile.write(body)
            elif path == "/api/agent-plugins":
                self.send_json({"plugins": application.list_agent_plugins()})
            elif path == "/api/chats":
                self.send_json(
                    {
                        "chats": application.store.list_chats(),
                        "total": application.store.count_chats(),
                    }
                )
            elif path.startswith("/api/chats/"):
                try:
                    chat_id = path.removeprefix("/api/chats/")
                    if not chat_id or "/" in chat_id:
                        raise ValueError("Invalid chat ID.")
                    self.send_json(application.get_chat(chat_id))
                except KeyError:
                    self.send_json({"error": "Chat not found"}, HTTPStatus.NOT_FOUND)
                except ValueError as error:
                    self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            elif path == "/api/citations":
                parameters = parse_qs(parsed.query)
                if parameters.get("summary", [""])[0] == "1":
                    try:
                        limit = int(parameters.get("limit", ["50"])[0])
                        offset = int(parameters.get("offset", ["0"])[0])
                        query = parameters.get("q", [""])[0].strip()
                        origin = parameters.get("origin", ["all"])[0].strip()
                        raw_ids = parameters.get("ids", [None])[0]
                        citation_ids = None
                        if raw_ids is not None:
                            parts = [part.strip() for part in raw_ids.split(",") if part.strip()]
                            if len(parts) > 10_000:
                                raise ValueError("Too many fact IDs.")
                            citation_ids = [int(part) for part in parts]
                            if any(citation_id < 1 for citation_id in citation_ids):
                                raise ValueError("Invalid fact ID filter.")
                        if not 1 <= limit <= 100 or offset < 0:
                            raise ValueError("Invalid pagination.")
                        if len(query) > 200:
                            raise ValueError("Search is too long.")
                        citations, filtered_total = application.store.list_summaries(
                            limit=limit,
                            offset=offset,
                            query=query,
                            citation_ids=citation_ids,
                            origin=origin,
                        )
                        self.send_json(
                            {
                                "citations": citations,
                                "total": application.store.count(origin),
                                "filtered_total": filtered_total,
                                "limit": limit,
                                "offset": offset,
                                "query": query,
                                "filter_ids": citation_ids,
                                "origin": origin,
                            }
                        )
                    except ValueError as error:
                        self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
                else:
                    self.send_json(
                        {
                            "citations": application.store.list(),
                            "total": application.store.count(),
                        }
                    )
            elif path == "/api/documents":
                parameters = parse_qs(parsed.query)
                query = parameters.get("q", [""])[0].strip()
                concept = parameters.get("concept", [""])[0].strip()
                if len(query) > 200 or len(concept) > 200:
                    self.send_json({"error": "Document filter is too long."}, HTTPStatus.BAD_REQUEST)
                    return
                documents = application.store.list_documents(query=query, concept=concept)
                self.send_json(
                    {
                        "documents": documents,
                        "total": application.store.count_documents(),
                        "filtered_total": len(documents),
                        "query": query,
                        "concept": concept,
                        "concepts": application.store.document_concepts(),
                    }
                )
            elif path.startswith("/api/documents/"):
                try:
                    document_id = int(path.removeprefix("/api/documents/"))
                    if document_id < 1:
                        raise ValueError("Invalid document ID.")
                    self.send_json({"document": application.store.get_document(document_id)})
                except KeyError:
                    self.send_json({"error": "Document not found"}, HTTPStatus.NOT_FOUND)
                except ValueError as error:
                    self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            elif path.startswith("/api/citations/"):
                try:
                    citation_id = int(path.removeprefix("/api/citations/"))
                    if citation_id < 1:
                        raise ValueError("Invalid fact ID.")
                    self.send_json({"citation": application.store.get(citation_id)})
                except KeyError:
                    self.send_json({"error": "Fact not found"}, HTTPStatus.NOT_FOUND)
                except ValueError as error:
                    self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            elif path == "/api/graph":
                self.send_json(application.store.graph())
            elif path.startswith("/api/concepts/") and path.endswith("/facts"):
                try:
                    raw_id = path.removeprefix("/api/concepts/").removesuffix("/facts")
                    concept_id = int(raw_id)
                    if concept_id < 1:
                        raise ValueError("Invalid concept ID.")
                    self.send_json(application.store.concept_facts(concept_id))
                except KeyError:
                    self.send_json({"error": "Concept not found"}, HTTPStatus.NOT_FOUND)
                except ValueError as error:
                    self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
                except Exception as error:
                    print(f"Concept facts failed: {error!r}", flush=True)
                    self.send_json(
                        {"error": "Could not load concept facts."},
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                    )
            elif path == "/api/graph/history":
                self.send_json(application.store.history_status())
            elif path == "/api/health":
                self.send_json(
                    {
                        "status": "ok",
                        "embedding_model": BGE_MODEL,
                        "t5gemma_enabled": application.models.t5_enabled,
                    }
                )
            elif path == "/api/model-setup":
                self.send_json(application.model_setup_status())
            elif path.startswith("/api/model-setup/jobs/"):
                job_id = path.removeprefix("/api/model-setup/jobs/")
                try:
                    self.send_json(application.get_model_setup_job(job_id))
                except KeyError:
                    self.send_json({"error": "Model setup job not found"}, HTTPStatus.NOT_FOUND)
            elif path == "/api/performance-samples":
                with PERFORMANCE_SAMPLES_LOCK:
                    self.send_json({"samples": list(PERFORMANCE_SAMPLES)})
            elif path.startswith("/api/integrations/"):
                job_id = path.removeprefix("/api/integrations/")
                try:
                    self.send_json(application.get_capture_job(job_id))
                except KeyError:
                    self.send_json({"error": "Integration job not found"}, HTTPStatus.NOT_FOUND)
            else:
                self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if not self.require_authorization():
                return
            path = urlparse(self.path).path
            if path == "/api/performance-samples":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if length <= 0 or length > 200_000:
                        raise ValueError("Invalid performance sample size.")
                    payload = json.loads(self.rfile.read(length))
                    if not isinstance(payload, dict):
                        raise ValueError("Performance sample must be an object.")
                    with PERFORMANCE_SAMPLES_LOCK:
                        PERFORMANCE_SAMPLES.append(payload)
                        del PERFORMANCE_SAMPLES[:-20]
                    print(
                        "PERF_SAMPLE " + json.dumps(payload, separators=(",", ":")),
                        flush=True,
                    )
                    self.send_json({"stored": True}, HTTPStatus.CREATED)
                except (ValueError, json.JSONDecodeError) as error:
                    self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
                return
            if path == "/api/model-setup/t5-access":
                self.send_json(application.models.check_t5_access())
                return
            if path == "/api/model-setup":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if length <= 0 or length > 10_000:
                        raise ValueError("Invalid request.")
                    payload = json.loads(self.rfile.read(length))
                    if not isinstance(payload, dict):
                        raise ValueError("Request body must be a JSON object.")
                    self.send_json(application.start_model_setup(payload), HTTPStatus.ACCEPTED)
                except (ValueError, json.JSONDecodeError) as error:
                    self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
                except Exception as error:
                    print(f"Model setup failed to start: {error!r}", flush=True)
                    self.send_json(
                        {"error": "Could not start local model setup."},
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                    )
                return
            if path == "/api/chats":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    payload = json.loads(self.rfile.read(length)) if length > 0 else {}
                    if not isinstance(payload, dict):
                        raise ValueError("Request body must be a JSON object.")
                    self.send_json(application.create_chat(payload), HTTPStatus.CREATED)
                except (ValueError, json.JSONDecodeError) as error:
                    self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
                except Exception as error:
                    print(f"Chat creation failed: {error!r}", flush=True)
                    self.send_json({"error": "Could not create chat."}, HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            if path.startswith("/api/chats/") and path.endswith("/messages"):
                try:
                    chat_id = path.removeprefix("/api/chats/").removesuffix("/messages")
                    length = int(self.headers.get("Content-Length", "0"))
                    if not chat_id or "/" in chat_id or length <= 0 or length > 200_000:
                        raise ValueError("Invalid request.")
                    payload = json.loads(self.rfile.read(length))
                    if not isinstance(payload, dict):
                        raise ValueError("Request body must be a JSON object.")
                    self.send_json(application.start_chat_turn(chat_id, payload), HTTPStatus.ACCEPTED)
                except KeyError:
                    self.send_json({"error": "Chat not found"}, HTTPStatus.NOT_FOUND)
                except (ValueError, json.JSONDecodeError) as error:
                    self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
                except Exception as error:
                    print(f"Chat turn failed to start: {error!r}", flush=True)
                    self.send_json({"error": "Could not start chat turn."}, HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            if path.startswith("/api/chat-messages/") and path.endswith("/annotations"):
                try:
                    message_id = path.removeprefix("/api/chat-messages/").removesuffix("/annotations")
                    length = int(self.headers.get("Content-Length", "0"))
                    if not message_id or "/" in message_id or length <= 0 or length > 100_000:
                        raise ValueError("Invalid request.")
                    payload = json.loads(self.rfile.read(length))
                    if not isinstance(payload, dict):
                        raise ValueError("Request body must be a JSON object.")
                    start = int(payload.get("message_start"))
                    end = int(payload.get("message_end"))
                    note = str(payload.get("note", "")).strip()
                    if not note or len(note) > 20_000:
                        raise ValueError("Annotation must contain between 1 and 20,000 characters.")
                    self.send_json(
                        application.store.add_chat_annotation(message_id, start, end, note),
                        HTTPStatus.CREATED,
                    )
                except KeyError:
                    self.send_json({"error": "Chat message not found"}, HTTPStatus.NOT_FOUND)
                except (TypeError, ValueError, json.JSONDecodeError) as error:
                    self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
                except Exception as error:
                    print(f"Chat annotation creation failed: {error!r}", flush=True)
                    self.send_json({"error": "Could not create annotation."}, HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            if path in {"/api/graph/history/undo", "/api/graph/history/redo"}:
                direction = path.rsplit("/", 1)[-1]
                try:
                    self.send_json(application.store.move_history(direction))
                except Exception as error:
                    print(f"Graph history failed: {error!r}", flush=True)
                    self.send_json({"error": "Could not change graph history."}, HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            if path.startswith("/api/citations/") and path.endswith("/integration"):
                try:
                    raw_id = path.removeprefix("/api/citations/").removesuffix("/integration")
                    citation_id = int(raw_id)
                    length = int(self.headers.get("Content-Length", "0"))
                    if citation_id < 1 or length <= 0 or length > 200_000:
                        raise ValueError("Invalid request.")
                    payload = json.loads(self.rfile.read(length))
                    if not isinstance(payload, dict) or not isinstance(payload.get("integrated"), bool):
                        raise ValueError("The integrated field must be boolean.")
                    self.send_json(
                        application.store.set_integration(citation_id, payload["integrated"])
                    )
                except KeyError:
                    self.send_json({"error": "Fact not found"}, HTTPStatus.NOT_FOUND)
                except (ValueError, json.JSONDecodeError) as error:
                    self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
                except Exception as error:
                    print(f"Fact integration change failed: {error!r}", flush=True)
                    self.send_json({"error": "Could not change fact integration."}, HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            if path.startswith("/api/documents/") and path.endswith("/annotations"):
                try:
                    raw_id = path.removeprefix("/api/documents/").removesuffix("/annotations")
                    document_id = int(raw_id)
                    length = int(self.headers.get("Content-Length", "0"))
                    if document_id < 1 or length <= 0 or length > 100_000:
                        raise ValueError("Invalid request.")
                    payload = json.loads(self.rfile.read(length))
                    if not isinstance(payload, dict):
                        raise ValueError("Request body must be a JSON object.")
                    start = int(payload.get("document_start"))
                    end = int(payload.get("document_end"))
                    note = str(payload.get("note", "")).strip()
                    if not note or len(note) > 20_000:
                        raise ValueError("Annotation must contain between 1 and 20,000 characters.")
                    self.send_json(
                        application.store.add_document_annotation(
                            document_id, start, end, note
                        ),
                        HTTPStatus.CREATED,
                    )
                except KeyError:
                    self.send_json({"error": "Document not found"}, HTTPStatus.NOT_FOUND)
                except (TypeError, ValueError, json.JSONDecodeError) as error:
                    self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
                except Exception as error:
                    print(f"Annotation creation failed: {error!r}", flush=True)
                    self.send_json({"error": "Could not create annotation."}, HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            if path not in {
                "/api/citations",
                "/api/documents",
                "/api/graph/positions",
                "/api/graph/fact-positions",
                "/api/integrations",
            }:
                self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                maximum = MAX_DOCUMENT_LENGTH + 100_000 if path == "/api/documents" else 200_000
                if length <= 0 or length > maximum:
                    raise ValueError("Invalid request size.")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("Request body must be a JSON object.")
                if path == "/api/citations":
                    if payload.get("integrated", True) is False:
                        self.send_json(application.save_unintegrated(payload), HTTPStatus.CREATED)
                    else:
                        self.send_json(application.capture(payload), HTTPStatus.CREATED)
                elif path == "/api/documents":
                    self.send_json(application.save_document(payload), HTTPStatus.CREATED)
                elif path == "/api/integrations":
                    self.send_json(application.start_capture(payload), HTTPStatus.ACCEPTED)
                elif path == "/api/graph/fact-positions":
                    self.send_json(application.save_fact_positions(payload))
                else:
                    self.send_json(application.save_graph_positions(payload))
            except (ValueError, json.JSONDecodeError) as error:
                self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            except Exception as error:  # Keep experimental server diagnostics visible.
                print(f"Capture failed: {error!r}", flush=True)
                self.send_json({"error": "Local analysis failed; see the server console."}, HTTPStatus.INTERNAL_SERVER_ERROR)

        def do_PUT(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if not self.require_authorization():
                return
            path = urlparse(self.path).path
            if not path.startswith("/api/citations/"):
                self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
                return
            try:
                citation_id = int(path.removeprefix("/api/citations/"))
                length = int(self.headers.get("Content-Length", "0"))
                if citation_id < 1 or length <= 0 or length > 200_000:
                    raise ValueError("Invalid request.")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("Request body must be a JSON object.")
                self.send_json(
                    application.start_update_fact(citation_id, payload),
                    HTTPStatus.ACCEPTED,
                )
            except KeyError:
                self.send_json({"error": "Fact not found"}, HTTPStatus.NOT_FOUND)
            except (ValueError, json.JSONDecodeError) as error:
                self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            except Exception as error:
                print(f"Fact update failed: {error!r}", flush=True)
                self.send_json({"error": "Could not update fact."}, HTTPStatus.INTERNAL_SERVER_ERROR)

        def do_PATCH(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            path = urlparse(self.path).path
            if path.startswith("/api/chat-annotations/"):
                try:
                    annotation_id = int(path.removeprefix("/api/chat-annotations/"))
                    length = int(self.headers.get("Content-Length", "0"))
                    if annotation_id < 1 or length <= 0 or length > 100_000:
                        raise ValueError("Invalid request.")
                    payload = json.loads(self.rfile.read(length))
                    if not isinstance(payload, dict):
                        raise ValueError("Request body must be a JSON object.")
                    note = str(payload.get("note", "")).strip()
                    if not note or len(note) > 20_000:
                        raise ValueError("Annotation must contain between 1 and 20,000 characters.")
                    self.send_json(application.store.update_chat_annotation(annotation_id, note))
                except KeyError:
                    self.send_json({"error": "Annotation not found"}, HTTPStatus.NOT_FOUND)
                except (ValueError, json.JSONDecodeError) as error:
                    self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
                except Exception as error:
                    print(f"Chat annotation update failed: {error!r}", flush=True)
                    self.send_json({"error": "Could not update annotation."}, HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            if path.startswith("/api/annotations/"):
                try:
                    annotation_id = int(path.removeprefix("/api/annotations/"))
                    length = int(self.headers.get("Content-Length", "0"))
                    if annotation_id < 1 or length <= 0 or length > 100_000:
                        raise ValueError("Invalid request.")
                    payload = json.loads(self.rfile.read(length))
                    if not isinstance(payload, dict):
                        raise ValueError("Request body must be a JSON object.")
                    note = str(payload.get("note", "")).strip()
                    if not note or len(note) > 20_000:
                        raise ValueError("Annotation must contain between 1 and 20,000 characters.")
                    self.send_json(application.store.update_document_annotation(annotation_id, note))
                except KeyError:
                    self.send_json({"error": "Annotation not found"}, HTTPStatus.NOT_FOUND)
                except (ValueError, json.JSONDecodeError) as error:
                    self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
                except Exception as error:
                    print(f"Annotation update failed: {error!r}", flush=True)
                    self.send_json({"error": "Could not update annotation."}, HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            if path == "/api/citations/labels":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if length <= 0 or length > 200_000:
                        raise ValueError("Invalid request size.")
                    payload = json.loads(self.rfile.read(length))
                    if not isinstance(payload, dict) or not isinstance(payload.get("citation_ids"), list):
                        raise ValueError("citation_ids must be an array.")
                    citation_ids = [int(value) for value in payload["citation_ids"]]
                    if any(value < 1 for value in citation_ids) or len(citation_ids) > 10_000:
                        raise ValueError("Invalid fact IDs.")
                    label = " ".join(str(payload.get("label", "")).split()).lower()
                    if not label or len(label) > 120:
                        raise ValueError("Label must contain between 1 and 120 characters.")
                    self.send_json({"citations": application.store.rename_labels(citation_ids, label)})
                except KeyError:
                    self.send_json({"error": "Fact not found"}, HTTPStatus.NOT_FOUND)
                except (TypeError, ValueError, json.JSONDecodeError) as error:
                    self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
                except Exception as error:
                    print(f"Label rename failed: {error!r}", flush=True)
                    self.send_json({"error": "Could not rename label."}, HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            if not path.startswith("/api/citations/") or not path.endswith("/draft"):
                self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
                return
            try:
                raw_id = path.removeprefix("/api/citations/").removesuffix("/draft")
                citation_id = int(raw_id)
                length = int(self.headers.get("Content-Length", "0"))
                if citation_id < 1 or length <= 0 or length > 200_000:
                    raise ValueError("Invalid request.")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("Request body must be a JSON object.")
                draft = str(payload.get("highlight", "")).strip()
                if len(draft) > MAX_TEXT_LENGTH:
                    raise ValueError(f"Text must be shorter than {MAX_TEXT_LENGTH:,} characters.")
                self.send_json(application.store.save_draft(citation_id, draft))
            except KeyError:
                self.send_json({"error": "Fact not found"}, HTTPStatus.NOT_FOUND)
            except (ValueError, json.JSONDecodeError) as error:
                self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            except Exception as error:
                print(f"Draft save failed: {error!r}", flush=True)
                self.send_json({"error": "Could not save draft."}, HTTPStatus.INTERNAL_SERVER_ERROR)

        def do_DELETE(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if not self.require_authorization():
                return
            path = urlparse(self.path).path
            if not path.startswith("/api/citations/"):
                self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
                return
            try:
                citation_id = int(path.removeprefix("/api/citations/"))
                if citation_id < 1:
                    raise ValueError("Invalid fact ID.")
                self.send_json(application.store.delete(citation_id))
            except KeyError:
                self.send_json({"error": "Fact not found"}, HTTPStatus.NOT_FOUND)
            except ValueError as error:
                self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            except Exception as error:
                print(f"Fact deletion failed: {error!r}", flush=True)
                self.send_json({"error": "Could not delete fact."}, HTTPStatus.INTERNAL_SERVER_ERROR)

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument(
        "--auth-token",
        help="Require this bearer/session token (used by the desktop shell).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    initialize_database(args.db)
    application = CaptureApplication(args.db)
    server = ThreadingHTTPServer(
        (args.host, args.port), make_handler(application, args.auth_token)
    )
    print(f"Arbrellis: http://{args.host}:{args.port}", flush=True)
    print(f"Document capture: http://{args.host}:{args.port}/capture", flush=True)
    print(f"Knowledge database: {args.db}", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
