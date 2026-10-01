"""Run the 100-question end-to-end benchmark and persist every observation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents.orchestrator import ResearchOrchestrator  # noqa: E402
from src.evaluation.database import EvaluationDatabase  # noqa: E402


def load_questions(path: Path) -> list[str]:
    questions = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(questions, list) or len(questions) != 100:
        raise ValueError(f"Expected exactly 100 questions in {path}, got {len(questions)}")
    if not all(isinstance(question, str) and question.strip() for question in questions):
        raise ValueError("Every benchmark question must be a non-empty string")
    return questions


def run(questions_path: Path, database_path: Path, export_path: Path, limit: int | None = None) -> None:
    questions = load_questions(questions_path)
    if limit is not None:
        questions = questions[:limit]

    orchestrator = ResearchOrchestrator()
    export: dict[str, Any] = {
        "questions_path": str(questions_path),
        "models": orchestrator.models,
        "results": [],
    }

    with EvaluationDatabase(database_path) as database:
        run_id = database.start_run(len(questions), orchestrator.models)
        for position, query in enumerate(questions, 1):
            telemetry: dict[str, Any] = {}
            print(f"[{position}/{len(questions)}] {query}")
            try:
                result = orchestrator.research(query, telemetry=telemetry)
                database.record_question(run_id, position, query, result, telemetry)
                export["results"].append({
                    "position": position,
                    "query": query,
                    "answer": result.answer,
                    "sources_count": result.sources_count,
                    "metadata": result.metadata,
                    "telemetry": telemetry,
                })
            except Exception as error:
                failure = {"position": position, "query": query, "error": repr(error), "telemetry": telemetry}
                export["results"].append(failure)
                print(f"  ERROR: {error}")
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_text(json.dumps(export, ensure_ascii=True, indent=2, default=str), encoding="utf-8")
        database.finish_run(run_id, str(export_path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=ROOT / "test/evaluation_questions.json")
    parser.add_argument("--db", type=Path, default=ROOT / "evaluation.sqlite3")
    parser.add_argument("--export", type=Path, default=ROOT / "evaluation-results.json")
    parser.add_argument("--limit", type=int, help="Run only the first N questions while developing")
    args = parser.parse_args()
    run(args.questions, args.db, args.export, args.limit)


if __name__ == "__main__":
    main()
