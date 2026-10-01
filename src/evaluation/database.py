"""SQLite persistence for end-to-end agent evaluations."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS evaluation_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    finished_at TEXT,
    question_count INTEGER NOT NULL,
    model_config TEXT NOT NULL,
    export_path TEXT
);
CREATE TABLE IF NOT EXISTS question_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES evaluation_runs(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    query TEXT NOT NULL,
    answer TEXT,
    sources_count INTEGER NOT NULL DEFAULT 0,
    total_duration_seconds REAL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS discovered_questions (
    question_run_id INTEGER NOT NULL REFERENCES question_runs(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    question TEXT NOT NULL,
    PRIMARY KEY (question_run_id, position)
);
CREATE TABLE IF NOT EXISTS search_results (
    question_run_id INTEGER NOT NULL REFERENCES question_runs(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    selected INTEGER NOT NULL DEFAULT 0,
    relevance_score REAL,
    question TEXT,
    title TEXT,
    url TEXT,
    body TEXT,
    result_json TEXT NOT NULL,
    PRIMARY KEY (question_run_id, position)
);
CREATE TABLE IF NOT EXISTS scrape_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question_run_id INTEGER NOT NULL REFERENCES question_runs(id) ON DELETE CASCADE,
    source_url TEXT,
    source_title TEXT,
    relevance_score REAL,
    extracted_info TEXT,
    tool_calls_json TEXT NOT NULL,
    tool_results_json TEXT NOT NULL,
    scrape_data_json TEXT NOT NULL,
    duration_seconds REAL
);
CREATE TABLE IF NOT EXISTS stage_timings (
    question_run_id INTEGER NOT NULL REFERENCES question_runs(id) ON DELETE CASCADE,
    stage TEXT NOT NULL,
    duration_seconds REAL NOT NULL,
    PRIMARY KEY (question_run_id, stage)
);
CREATE INDEX IF NOT EXISTS idx_question_runs_run_id ON question_runs(run_id);
CREATE INDEX IF NOT EXISTS idx_search_results_score ON search_results(relevance_score);
"""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, default=str)


class EvaluationDatabase:
    """Small repository around a SQLite evaluation database."""

    def __init__(self, path: str | Path = "evaluation.sqlite3") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.executescript(SCHEMA)
        self.connection.commit()

    def start_run(self, question_count: int, model_config: dict[str, str]) -> int:
        cursor = self.connection.execute(
            "INSERT INTO evaluation_runs(question_count, model_config) VALUES (?, ?)",
            (question_count, _json(model_config)),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def record_question(self, run_id: int, position: int, query: str, result: Any, telemetry: dict[str, Any]) -> int:
        metadata = getattr(result, "metadata", {})
        cursor = self.connection.execute(
            """INSERT INTO question_runs
            (run_id, position, query, answer, sources_count, total_duration_seconds, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (run_id, position, query, result.answer, result.sources_count,
             telemetry.get("stages", {}).get("total"), _json(metadata)),
        )
        question_run_id = int(cursor.lastrowid)

        for index, question in enumerate(telemetry.get("questions_found", []), 1):
            self.connection.execute(
                "INSERT INTO discovered_questions VALUES (?, ?, ?)",
                (question_run_id, index, question),
            )

        selected_urls = {item.get("link", item.get("href", "")) for item in telemetry.get("selected_results", [])}
        for index, item in enumerate(telemetry.get("ranked_results", []), 1):
            url = item.get("link", item.get("href", ""))
            self.connection.execute(
                """INSERT INTO search_results
                (question_run_id, position, selected, relevance_score, question, title, url, body, result_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (question_run_id, index, int(url in selected_urls), item.get("relevance_score"),
                 item.get("question"), item.get("title"), url, item.get("body"), _json(item)),
            )

        for item in telemetry.get("scrape_results", []):
            self.connection.execute(
                """INSERT INTO scrape_results
                (question_run_id, source_url, source_title, relevance_score, extracted_info,
                 tool_calls_json, tool_results_json, scrape_data_json, duration_seconds)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (question_run_id, item.get("source_url"), item.get("source_title"),
                 item.get("relevance_score"), item.get("extracted_info"), _json(item.get("tool_calls", [])),
                 _json(item.get("tool_results", [])), _json(item.get("scrape_data", {})), item.get("duration_seconds")),
            )

        for stage, duration in telemetry.get("stages", {}).items():
            self.connection.execute(
                "INSERT INTO stage_timings VALUES (?, ?, ?)",
                (question_run_id, stage, duration),
            )
        self.connection.commit()
        return question_run_id

    def finish_run(self, run_id: int, export_path: str | None = None) -> None:
        self.connection.execute(
            "UPDATE evaluation_runs SET finished_at = CURRENT_TIMESTAMP, export_path = ? WHERE id = ?",
            (export_path, run_id),
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "EvaluationDatabase":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
