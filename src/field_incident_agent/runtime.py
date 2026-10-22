"""Application composition root for incident intake."""

from dataclasses import dataclass
from pathlib import Path

from field_incident_agent.database import Database
from field_incident_agent.groq_client import GroqIncidentExtractionClient
from field_incident_agent.incident_service import IncidentService
from field_incident_agent.llm_client import IncidentExtractionClient
from field_incident_agent.orchestrator import IncidentOrchestrator
from field_incident_agent.text_workflow import TextIncidentWorkflow


@dataclass(frozen=True, slots=True)
class IncidentRuntime:
    """Assembled application services; authentication remains external."""

    database: Database
    service: IncidentService
    workflow: TextIncidentWorkflow
    orchestrator: IncidentOrchestrator


def build_runtime(
    database_path: str | Path,
    *,
    client: IncidentExtractionClient | None = None,
) -> IncidentRuntime:
    """Build the runtime with Groq by default or an injected test client."""
    extraction_client = (
        client if client is not None else GroqIncidentExtractionClient()
    )

    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    database = Database(path)
    database.initialize()

    service = IncidentService(database)
    workflow = TextIncidentWorkflow(service)
    orchestrator = IncidentOrchestrator(
        extraction_client,
        workflow,
        service,
    )

    return IncidentRuntime(
        database=database,
        service=service,
        workflow=workflow,
        orchestrator=orchestrator,
    )