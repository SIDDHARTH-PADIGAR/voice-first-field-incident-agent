import json
from pathlib import Path

path = Path("eval/scenarios/backend_v1.json")
data = json.loads(path.read_text(encoding="utf-8"))

required = {
    "scenario_id",
    "category",
    "risk_level",
    "initial_database_state",
    "caller_role",
    "input_turns",
    "expected_required_fields",
    "expected_clarifications",
    "allowed_tool_calls",
    "forbidden_tool_calls",
    "expected_escalation",
    "expected_terminal_state",
    "expected_database_changes",
    "assertions",
    "notes",
}

scenarios = data["scenarios"]
ids = [scenario["scenario_id"] for scenario in scenarios]

assert data["schema_version"] == 1
assert len(scenarios) == 12
assert len(ids) == len(set(ids))
assert all(required.issubset(scenario) for scenario in scenarios)
assert all(
    scenario["risk_level"] in {"low", "medium", "high", "critical"}
    for scenario in scenarios
)

print(
    f"Catalog valid: {len(scenarios)} scenarios, "
    f"{len(ids)} unique IDs, schema v{data['schema_version']}."
)