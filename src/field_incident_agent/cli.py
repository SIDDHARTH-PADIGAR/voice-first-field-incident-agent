"""Development-only text demo using synthetic assets."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import uuid

from field_incident_agent.incident_service import IncidentValidationError
from field_incident_agent.llm_client import (
    IncidentExtractionClient,
    ModelClientError,
)
from field_incident_agent.models import Asset
from field_incident_agent.registry import AccessContext
from field_incident_agent.runtime import build_runtime


DEMO_SITE_ID = "SITE-DEMO"
DEMO_PRINCIPAL_ID = "TECH-DEMO"

DEMO_ASSETS = (
    Asset(
        "DEMO-PUMP-001",
        DEMO_SITE_ID,
        "pump",
        "Main demonstration water pump",
    ),
    Asset(
        "DEMO-PUMP-002",
        DEMO_SITE_ID,
        "pump",
        "Backup demonstration water pump",
    ),
    Asset(
        "DEMO-LIFT-001",
        DEMO_SITE_ID,
        "elevator",
        "Demonstration service elevator",
    ),
    Asset(
        "DEMO-PANEL-001",
        DEMO_SITE_ID,
        "electrical_panel",
        "Demonstration electrical panel",
    ),
)


def _ensure_demo_assets(runtime) -> None:
    """Create synthetic assets once and reject conflicting seed records."""
    with runtime.database.connect() as connection:
        rows = connection.execute(
            """
            SELECT asset_id, site_id, asset_type, display_name
            FROM assets
            WHERE asset_id LIKE 'DEMO-%'
            """
        ).fetchall()

    existing = {
        row["asset_id"]: (
            row["site_id"],
            row["asset_type"],
            row["display_name"],
        )
        for row in rows
    }

    for asset in DEMO_ASSETS:
        expected = (
            asset.site_id,
            asset.asset_type,
            asset.display_name,
        )

        if asset.asset_id in existing:
            if existing[asset.asset_id] != expected:
                raise RuntimeError(
                    f"Conflicting demo asset already exists: "
                    f"{asset.asset_id}"
                )
            continue

        runtime.service.register_asset(asset)


def main(
    argv: list[str] | None = None,
    *,
    client: IncidentExtractionClient | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description="Run the synthetic field incident intake demo."
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/demo/incident_demo.sqlite3"),
        help="SQLite database path for demo runs.",
    )
    args = parser.parse_args(argv)

    try:
        runtime = build_runtime(args.database, client=client)
    except ModelClientError as exc:
        print(f"Cannot start demo: {exc}", file=sys.stderr)
        return 2

    _ensure_demo_assets(runtime)

    access = AccessContext(
        principal_id=DEMO_PRINCIPAL_ID,
        authorized_site_ids=frozenset({DEMO_SITE_ID}),
    )

    session_id = f"SESSION-DEMO-{uuid.uuid4().hex}"
    idempotency_key = f"IDEM-DEMO-{uuid.uuid4().hex}"

    runtime.workflow.start(session_id, access)

    print("\nField Incident Intake Demo")
    print("==========================")
    print(f"Model: {getattr(runtime.orchestrator.client, 'model', 'injected client')}")
    print(f"Session: {session_id}")
    print("Use synthetic demo assets only.")
    print("\nExample asset IDs: DEMO-PUMP-001, DEMO-PUMP-002")
    print("Do not submit real sensitive incident data.")

    try:
        report = input("\nDescribe the incident: ").strip()
    except EOFError:
        print("\nNo report received.")
        return 1

    outcome = runtime.orchestrator.process_report(
        session_id,
        report,
        access,
        idempotency_key,
    )

    print(f"\nStatus: {outcome.reply.status}")
    print(outcome.reply.message)

    if outcome.extraction is not None:
        print("\nValidated extraction:")
        print(json.dumps(asdict(outcome.extraction), indent=2))

    if outcome.reply.status != "confirmation_required":
        print("\nNo normal incident ticket was created by this step.")
        print(f"Database: {args.database.resolve()}")
        return 0

    print("\nProposed incident:")
    print(json.dumps(outcome.reply.data["proposal"], indent=2))

    try:
        answer = input(
            "\nType YES to explicitly confirm ticket creation: "
        ).strip()
    except EOFError:
        answer = ""

    if answer.casefold() != "yes":
        print("Not confirmed. No incident ticket was created.")
        print(f"Database: {args.database.resolve()}")
        return 0

    try:
        result = runtime.orchestrator.confirm_proposal(
            outcome,
            access,
            idempotency_key,
            explicitly_confirmed=True,
        )
    except IncidentValidationError as exc:
        print(f"Submission rejected: {exc}")
        return 1

    print(f"\nFinal status: {result.status}")
    print(result.message)
    print(json.dumps(result.data, indent=2))
    print(f"Database: {args.database.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())