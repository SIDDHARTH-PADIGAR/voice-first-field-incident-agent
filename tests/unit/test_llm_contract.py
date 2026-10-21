import json

import pytest

from field_incident_agent.llm_contract import parse_incident_extraction


def valid_payload():
    return {
        "site_id": "SITE-001",
        "asset_id": "PUMP-001",
        "asset_lookup_clue": None,
        "description": "The pump is leaking.",
        "symptoms": ["Water pooling"],
        "safety_indicators": ["active_water_leak"],
        "safety_concern_uncertain": False,
        "time_noticed": None,
        "operational_impact": None,
    }


def test_valid_extraction_is_parsed():
    result = parse_incident_extraction(valid_payload())

    assert result.site_id == "SITE-001"
    assert result.asset_id == "PUMP-001"
    assert result.safety_indicators == ("active_water_leak",)


def test_json_text_is_parsed():
    result = parse_incident_extraction(json.dumps(valid_payload()))

    assert result.description == "The pump is leaking."


def test_missing_asset_remains_unknown():
    payload = valid_payload()
    payload["asset_id"] = None
    payload["asset_lookup_clue"] = "water pump"

    result = parse_incident_extraction(payload)

    assert result.asset_id is None
    assert result.asset_lookup_clue == "water pump"


def test_missing_required_output_field_is_rejected():
    payload = valid_payload()
    del payload["site_id"]

    with pytest.raises(ValueError, match="Missing extraction fields"):
        parse_incident_extraction(payload)


@pytest.mark.parametrize(
    "field,value",
    [
        ("severity", "low"),
        ("ticket_id", "INC-123"),
        ("authorized_site_ids", ["SITE-001"]),
        ("confirmed", True),
        ("tool_name", "create_incident"),
        ("workflow_state", "submitted"),
    ],
)
def test_model_cannot_return_control_fields(field, value):
    payload = valid_payload()
    payload[field] = value

    with pytest.raises(ValueError, match="Unexpected extraction fields"):
        parse_incident_extraction(payload)


def test_invalid_identifier_is_rejected():
    payload = valid_payload()
    payload["asset_id"] = "PUMP 001"

    with pytest.raises(ValueError, match="asset_id"):
        parse_incident_extraction(payload)


def test_unapproved_safety_indicator_is_rejected():
    payload = valid_payload()
    payload["safety_indicators"] = ["probably_safe"]

    with pytest.raises(ValueError, match="Unsupported safety indicators"):
        parse_incident_extraction(payload)


def test_uncertain_safety_concern_must_be_boolean():
    payload = valid_payload()
    payload["safety_concern_uncertain"] = "false"

    with pytest.raises(ValueError, match="must be a boolean"):
        parse_incident_extraction(payload)


def test_invalid_json_is_rejected():
    with pytest.raises(ValueError, match="valid JSON"):
        parse_incident_extraction("{invalid json")


def test_duplicate_indicators_are_deduplicated():
    payload = valid_payload()
    payload["safety_indicators"] = [
        "active_water_leak",
        " active_water_leak ",
    ]

    result = parse_incident_extraction(payload)

    assert result.safety_indicators == ("active_water_leak",)


def test_empty_description_stays_an_error():
    payload = valid_payload()
    payload["description"] = "   "

    with pytest.raises(ValueError, match="description"):
        parse_incident_extraction(payload)