import pytest

from field_incident_agent.llm_client import ModelClientError
from field_incident_agent.runtime import build_runtime


class NeverCalledClient:
    def extract_incident(self, text):
        raise AssertionError("Runtime construction must not call the model.")


def test_runtime_assembles_components_with_injected_client(tmp_path):
    database_path = tmp_path / "nested" / "incidents.db"
    fake_client = NeverCalledClient()

    runtime = build_runtime(database_path, client=fake_client)

    assert database_path.exists()
    assert runtime.orchestrator.client is fake_client
    assert runtime.orchestrator.service is runtime.service
    assert runtime.orchestrator.workflow is runtime.workflow

    with runtime.database.connect() as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }

    assert "sessions" in tables
    assert "incidents" in tables


def test_runtime_without_api_key_fails_before_database_creation(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    database_path = tmp_path / "incidents.db"

    with pytest.raises(ModelClientError, match="GROQ_API_KEY"):
        build_runtime(database_path)

    assert not database_path.exists()