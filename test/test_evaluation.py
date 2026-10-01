from pathlib import Path

from src.agents.orchestrator import ResearchOrchestrator
from src.evaluation.database import EvaluationDatabase
from src.tools.agent_response import AgentResponse


class FakeUserAgent:
    def chat(self, query, stream=True):
        return AgentResponse(content="<SUBREQUEST>\n- sous-question test\n<ENDSUBREQUEST>")

    def parse_subrequests(self, content):
        return ["sous-question test"]

    def reset(self):
        pass


class FakeAnalysisAgent:
    def web_search_score(self, query, web_search_result):
        return [{**item, "relevance_score": 91, "link": item["link"]} for item in web_search_result]

    def reset(self):
        pass


class FakeSynthesisAgent:
    def synthesize(self, query, extracted_sources, stream=False):
        return "Réponse synthétique"

    def reset(self):
        pass


class FakeScrappingAgent:
    def reset(self):
        pass


def fake_search_web(query, max_results):
    return [{
        "title": "Source test",
        "link": "https://example.test/source",
        "body": "album sortie released 2026 market cap " + "x" * 300,
    }]


def test_research_records_each_stage(monkeypatch):
    import src.agents.orchestrator as module

    orchestrator = ResearchOrchestrator.__new__(ResearchOrchestrator)
    orchestrator.models = {"scrapping": "fake"}
    orchestrator.user_agent = FakeUserAgent()
    orchestrator.analysis_agent = FakeAnalysisAgent()
    orchestrator.scrapping_agent = FakeScrappingAgent()
    orchestrator.synthesis_agent = FakeSynthesisAgent()
    monkeypatch.setattr(module, "search_web", fake_search_web)

    telemetry = {}
    result = orchestrator.research("question test", telemetry=telemetry)

    assert result.answer == "Réponse synthétique"
    assert telemetry["questions_found"] == ["sous-question test"]
    assert len(telemetry["ranked_results"]) == 1
    assert len(telemetry["selected_results"]) == 1
    assert len(telemetry["scrape_results"]) == 1
    assert {"decomposition", "search", "analysis", "scraping", "synthesis", "total"} <= set(telemetry["stages"])


def test_database_persists_observations(tmp_path: Path):
    database_path = tmp_path / "evaluation.sqlite3"
    telemetry = {
        "questions_found": ["sub"],
        "ranked_results": [{"title": "t", "link": "https://example.test", "relevance_score": 88}],
        "selected_results": [{"link": "https://example.test"}],
        "scrape_results": [{"source_url": "https://example.test", "extracted_info": "info", "tool_calls": [], "tool_results": [], "scrape_data": {}, "duration_seconds": 0.1}],
        "stages": {"total": 0.2, "synthesis": 0.01},
    }
    result = type("Result", (), {"answer": "answer", "sources_count": 1, "metadata": {}})()

    with EvaluationDatabase(database_path) as database:
        run_id = database.start_run(1, {"synthesis": "fake"})
        question_run_id = database.record_question(run_id, 1, "query", result, telemetry)
        database.finish_run(run_id)
        assert question_run_id > 0
        assert database.connection.execute("SELECT COUNT(*) FROM stage_timings").fetchone()[0] == 2
        assert database.connection.execute("SELECT selected FROM search_results").fetchone()[0] == 1
