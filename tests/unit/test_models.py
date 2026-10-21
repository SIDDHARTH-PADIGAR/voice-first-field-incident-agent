import pytest

from field_incident_agent.models import (
    Asset,
    AssetStatus,
    IncidentDraft,
    IncidentStatus,
    SessionState,
    Severity,
)


def test_valid_asset_is_created():
    asset = Asset(
        asset_id="PUMP-001",
        site_id="SITE-001",
        asset_type="pump",
        display_name="Main water pump",
    )

    assert asset.asset_id == "PUMP-001"
    assert asset.status is AssetStatus.ACTIVE


def test_asset_rejects_malformed_identifier():
    with pytest.raises(ValueError, match="asset_id"):
        Asset(
            asset_id="INVALID ID",
            site_id="SITE-001",
            asset_type="pump",
            display_name="Main water pump",
        )


def test_asset_rejects_empty_display_name():
    with pytest.raises(ValueError, match="display_name"):
        Asset(
            asset_id="PUMP-001",
            site_id="SITE-001",
            asset_type="pump",
            display_name="   ",
        )


def test_asset_rejects_invalid_status():
    with pytest.raises(ValueError, match="status"):
        Asset(
            asset_id="PUMP-001",
            site_id="SITE-001",
            asset_type="pump",
            display_name="Main water pump",
            status="active",
        )


def test_incomplete_report_preserves_unknown_asset():
    draft = IncidentDraft(
        session_id="SESSION-001",
        description="Pump is making an unusual noise",
    )

    assert draft.asset_id is None
    assert draft.site_id is None
    assert draft.asset_lookup_clue is None


def test_incident_draft_normalizes_text():
    draft = IncidentDraft(
        session_id="SESSION-001",
        description="  Pump is leaking  ",
        symptoms=("  Water pooling  ",),
    )

    assert draft.description == "Pump is leaking"
    assert draft.symptoms == ("Water pooling",)


def test_incident_draft_rejects_blank_description():
    with pytest.raises(ValueError, match="description"):
        IncidentDraft(
            session_id="SESSION-001",
            description="   ",
        )


def test_incident_draft_rejects_malformed_asset_id():
    with pytest.raises(ValueError, match="asset_id"):
        IncidentDraft(
            session_id="SESSION-001",
            description="Pump is leaking",
            asset_id="BAD ID",
        )


def test_incident_draft_rejects_list_for_tuple_field():
    with pytest.raises(ValueError, match="symptoms"):
        IncidentDraft(
            session_id="SESSION-001",
            description="Pump is leaking",
            symptoms=["Water pooling"],
        )


def test_incident_draft_rejects_blank_symptom():
    with pytest.raises(ValueError, match="symptoms item"):
        IncidentDraft(
            session_id="SESSION-001",
            description="Pump is leaking",
            symptoms=("   ",),
        )


def test_incident_draft_rejects_oversized_description():
    with pytest.raises(ValueError, match="description"):
        IncidentDraft(
            session_id="SESSION-001",
            description="x" * 2001,
        )


def test_domain_enums_expose_expected_values():
    assert Severity.CRITICAL.value == "critical"
    assert IncidentStatus.OPEN.value == "open"
    assert SessionState.SUBMITTED.value == "submitted"