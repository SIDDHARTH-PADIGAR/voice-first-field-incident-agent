"""Repeatable live evaluation of Groq incident extraction on synthetic data."""

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from field_incident_agent.groq_client import GroqIncidentExtractionClient
from field_incident_agent.llm_contract import parse_incident_extraction


CASES = [
    {
        "id": "LLM-001-degraded-equipment",
        "report": (
            "At site SITE-001, pump PUMP-104 is vibrating more than usual "
            "and output is reduced. No smoke, leaks, sparks, injury, or "
            "immediate danger."
        ),
        "required_indicators": ["equipment_degraded"],
        "expected_fields": {
            "site_id": "SITE-001",
            "asset_id": "PUMP-104",
            "safety_concern_uncertain": False,
        },
    },
    {
        "id": "LLM-002-repeated-fault",
        "report": (
            "At site SITE-001, elevator LIFT-003 stopped twice in the "
            "last hour and restarted each time. It is running now."
        ),
        "required_indicators": ["repeated_fault"],
        "expected_fields": {
            "site_id": "SITE-001",
            "asset_id": "LIFT-003",
        },
        # Safety uncertainty is recorded, but not scored automatically:
        # a repeated elevator fault needs contextual review.
    },
    {
        "id": "LLM-003-fire-hazard",
        "report": (
            "At site SITE-001, electrical panel PANEL-002 is smoking "
            "and visible flames are present. No injuries reported."
        ),
        "required_indicators": ["fire_or_smoke"],
        "expected_fields": {
            "site_id": "SITE-001",
            "asset_id": "PANEL-002",
            "safety_concern_uncertain": False,
        },
    },
    {
        "id": "LLM-004-active-water-leak",
        "report": (
            "At site SITE-002, a ceiling pipe is actively leaking water "
            "into an occupied hallway. Water is spreading across the floor."
        ),
        "required_indicators": ["active_water_leak"],
        "expected_fields": {
            "site_id": "SITE-002",
            "asset_id": None,
        },
    },
    {
        "id": "LLM-005-ambiguous-asset-and-risk",
        "report": (
            "Something near the loading dock is making a strange noise. "
            "I don't know which machine it is or whether it is dangerous."
        ),
        "required_indicators": [],
        "expected_fields": {
            "site_id": None,
            "asset_id": None,
            "safety_concern_uncertain": True,
        },
    },
]


def json_value(value):
    """Convert tuples from the dataclass into JSON-compatible lists."""
    if isinstance(value, tuple):
        return [json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_value(item) for item in value]
    return value


def main():
    client = GroqIncidentExtractionClient()
    results = []

    for case in CASES:
        started = time.perf_counter()
        result = {
            "case_id": case["id"],
            "passed": False,
            "contract_valid": False,
            "failures": [],
        }

        try:
            raw = client.extract_incident(case["report"])
            extraction = parse_incident_extraction(raw)
            actual = asdict(extraction)
            result["contract_valid"] = True
            result["extraction"] = json_value(actual)

            actual_indicators = set(extraction.safety_indicators)
            missing = (
                set(case["required_indicators"]) - actual_indicators
            )
            if missing:
                result["failures"].append(
                    f"Missing required indicators: {sorted(missing)}"
                )

            for field, expected in case["expected_fields"].items():
                observed = actual[field]
                if isinstance(observed, tuple):
                    observed = list(observed)

                if observed != expected:
                    result["failures"].append(
                        f"{field}: expected {expected!r}, "
                        f"got {observed!r}"
                    )

            result["passed"] = not result["failures"]

        except Exception as exc:
            result["failures"].append(
                f"{type(exc).__name__}: {exc}"
            )

        result["latency_seconds"] = round(
            time.perf_counter() - started, 2
        )
        results.append(result)

        status = "PASS" if result["passed"] else "FAIL"
        print(f"{status} {case['id']}")

        for failure in result["failures"]:
            print(f"  - {failure}")

    passed = sum(item["passed"] for item in results)
    report = {
        "report_version": "llm_extraction_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": client.model,
        "summary": {
            "passed": passed,
            "failed": len(results) - passed,
            "total": len(results),
        },
        "cases": results,
        "limitations": [
            "Synthetic cases only; not production validation.",
            "Passing means only the assertions defined for each case passed.",
            "The repeated-elevator safety uncertainty requires review.",
        ],
    }

    report_path = Path("eval") / "results" / "llm_extraction_v1_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print()
    print(
        f"LLM extraction evaluation: {passed}/{len(results)} passed; "
        f"{len(results) - passed} failed."
    )
    print(f"Report written to: {report_path}")

    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())