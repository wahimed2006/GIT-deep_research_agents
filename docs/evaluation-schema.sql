-- SQLite schema for agent evaluations.
-- SQLite has no database password; protect the file with OS permissions.
-- PostgreSQL migration: replace INTEGER PRIMARY KEY AUTOINCREMENT with BIGSERIAL
-- and JSON text columns with JSONB.

PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS evaluation_runs (id INTEGER PRIMARY KEY AUTOINCREMENT, started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, finished_at TEXT, question_count INTEGER NOT NULL, model_config TEXT NOT NULL, export_path TEXT);
CREATE TABLE IF NOT EXISTS question_runs (id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER NOT NULL REFERENCES evaluation_runs(id) ON DELETE CASCADE, position INTEGER NOT NULL, query TEXT NOT NULL, answer TEXT, sources_count INTEGER NOT NULL DEFAULT 0, total_duration_seconds REAL, metadata_json TEXT NOT NULL DEFAULT '{}');
CREATE TABLE IF NOT EXISTS discovered_questions (question_run_id INTEGER NOT NULL REFERENCES question_runs(id) ON DELETE CASCADE, position INTEGER NOT NULL, question TEXT NOT NULL, PRIMARY KEY (question_run_id, position));
CREATE TABLE IF NOT EXISTS search_results (question_run_id INTEGER NOT NULL REFERENCES question_runs(id) ON DELETE CASCADE, position INTEGER NOT NULL, selected INTEGER NOT NULL DEFAULT 0, relevance_score REAL, question TEXT, title TEXT, url TEXT, body TEXT, result_json TEXT NOT NULL, PRIMARY KEY (question_run_id, position));
CREATE TABLE IF NOT EXISTS scrape_results (id INTEGER PRIMARY KEY AUTOINCREMENT, question_run_id INTEGER NOT NULL REFERENCES question_runs(id) ON DELETE CASCADE, source_url TEXT, source_title TEXT, relevance_score REAL, extracted_info TEXT, tool_calls_json TEXT NOT NULL, tool_results_json TEXT NOT NULL, scrape_data_json TEXT NOT NULL, duration_seconds REAL);
CREATE TABLE IF NOT EXISTS stage_timings (question_run_id INTEGER NOT NULL REFERENCES question_runs(id) ON DELETE CASCADE, stage TEXT NOT NULL, duration_seconds REAL NOT NULL, PRIMARY KEY (question_run_id, stage));
CREATE INDEX IF NOT EXISTS idx_question_runs_run_id ON question_runs(run_id);
CREATE INDEX IF NOT EXISTS idx_search_results_score ON search_results(relevance_score);
